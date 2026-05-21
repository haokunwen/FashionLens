from torch.utils.data import Dataset, Sampler
import os 
import json 
from typing import List, Dict, Any, Iterator, Optional, Tuple
import torch
import random
import torch.distributed as dist
import math
from collections import deque
from qwen_vl_utils import process_vision_info

num_constrains = {
    '1_t2i_FashionGen_train_triplet_data.json': [0, 6],
    '1_t2i_Shoes_train_triplet_data.json': [0, 0],
    '2_sketch2i_ClothesV1_train_triplet_data.json': [5, 0],
    '2_sketch2i_HAIFashion_train_triplet_data.json': [0, 0],
    '2_sketch2i_QMUL-Shoe-V2_train_triplet_data.json': [5, 0],
    '3_inshop_DeepFashion_train_triplet_data.json' : [0, 0],
    '4_street2shop_DeepFashion2_train_triplet_data.json': [0, 6],
    '5_compatibility_FashionVC_train_triplet_data.json': [0, 0],
    '5_compatibility_Polyvore_train_triplet_data.json': [0, 0],
    '6_video2img_MovingFashion_train_triplet_data.json': [0, 0],
    '7_cir_Fashion200K_train_triplet_data.json': [0, 0],
    '7_cir_FashionIQ-Dress_train_triplet_data.json': [0, 0],
    '7_cir_FashionIQ-Shirt_train_triplet_data.json': [0, 0],
    '7_cir_FashionIQ-Toptee_train_triplet_data.json': [0, 0],
    '7_cir_Shoes_train_triplet_data.json': [0, 0],
    '8_sketch&text_CSTBIR_train_triplet_data.json': [0, 0],
    '9_asfr_DARN_train_triplet_data.json': [0, 0],
    '9_asfr_FashionAI_train_triplet_data.json': [0, 0],
}

class FashionDataset(Dataset):
    def __init__(self, base_path, train_triplet_path):
        with open(os.path.join(base_path, '0_triplet_files', train_triplet_path), 'r') as f:
            self.train_data = json.load(f)
        self.base_path = base_path
        self.train_triplet_path = train_triplet_path

        self.query_image_constraint, self.target_image_constraint = num_constrains[train_triplet_path]
        self.target_image_constraint = 1
    def __len__(self):
        return len(self.train_data)
    def __getitem__(self, idx):
        data = self.train_data[idx]

        instruction_text = data['instruction_text']
        target_image = [os.path.join(self.base_path, p) for p in data['target_image']]
        
        if self.target_image_constraint > 0:
            if len(target_image) > 0:
                target_image = random.sample(target_image, min(self.target_image_constraint, len(target_image)))

        if 'query_image' in data:
            query_image = [os.path.join(self.base_path, p) for p in data['query_image']]
            if self.query_image_constraint > 0 and len(query_image) > 0:
                query_image = random.sample(query_image, min(self.query_image_constraint, len(query_image)))
        else:
            query_image = None
        
        if 'query_video' in data:
            query_video = [os.path.join(self.base_path, p) for p in data['query_video']]
        else:
            query_video = None
        
        return {
            'instruction_text': instruction_text,
            'query_image': query_image,
            'query_video': query_video,
            'target_image': target_image,
            'dataset': self.train_triplet_path
        }


class CombinedDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = [0] * len(datasets)
        for i in range(1, len(datasets)):
            self.offsets[i] = self.offsets[i-1] + self.lengths[i-1]
        self.total_len = sum(self.lengths)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        for i, offset in enumerate(self.offsets):
            if idx < offset + self.lengths[i]:
                return self.datasets[i][idx - offset]
        raise IndexError(f"Index {idx} out of range for CombinedDataset")


class InterleavingBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        datasets: List[Dataset],
        batch_size: int,
        adaptive_alpha: float = 0.9,       
        adaptive_lam: float = 0.5,         
        adaptive_temp: float = 1.0, 
        adaptive_min_prob: float = 0.02,
        probabilities: Optional[List[float]] = None,
        seed: int = 42,
        drop_last: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        if not datasets:
            raise ValueError("Dataset list cannot be empty.")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        
        self.datasets = datasets
        self.batch_size = batch_size
        self.probabilities = probabilities
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # ---------------------
        self.alpha = adaptive_alpha
        self.lam = adaptive_lam
        self.temp = adaptive_temp
        self.num_datasets = len(datasets)
        
        max_possible_min_prob = 1.0 / self.num_datasets
        if adaptive_min_prob > max_possible_min_prob:
            if rank == 0:
                print(f"[Warning] adaptive_min_prob {adaptive_min_prob} is too large for {self.num_datasets} datasets.")
                print(f"Clamping to {max_possible_min_prob} (Uniform Distribution).")
            self.min_prob = max_possible_min_prob
        else:
            self.min_prob = adaptive_min_prob

        self.dataset_lengths = [len(d) for d in datasets]
        self.task_difficulties = [0.0] * self.num_datasets
        self.adaptive_enabled = False
        # ---------------------

        # --- Distributed Init ---
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        self.num_replicas = num_replicas
        self.rank = rank
        
        self.replica_dataset_indices = [[] for _ in range(self.num_datasets)]
        self.replica_dataset_effective_lengths = [0] * self.num_datasets 
        self.total_samples_for_this_replica = 0

        for ds_idx, d_len in enumerate(self.dataset_lengths):
            full_indices = list(range(d_len))
            
            # DDP Slicing
            start = self.rank
            step = self.num_replicas
            current_replica_indices_subset = full_indices[start:d_len:step]
            
            num_samples_in_subset = len(current_replica_indices_subset)
            
            if self.drop_last:
                replica_effective_len_ds = num_samples_in_subset - (num_samples_in_subset % self.batch_size)
            else:
                replica_effective_len_ds = num_samples_in_subset
            
            self.replica_dataset_indices[ds_idx] = current_replica_indices_subset
            self.replica_dataset_effective_lengths[ds_idx] = replica_effective_len_ds
            self.total_samples_for_this_replica += replica_effective_len_ds

        # Global offsets for ConcatDataset
        self.global_index_offsets = [0] * self.num_datasets
        for i in range(1, self.num_datasets):
            self.global_index_offsets[i] = self.global_index_offsets[i-1] + self.dataset_lengths[i-1]
            
        # --- Calculate Local Batches ---
        if self.total_samples_for_this_replica == 0:
            local_num_batches = 0
        elif self.drop_last:
            local_num_batches = self.total_samples_for_this_replica // self.batch_size
        else:
            local_num_batches = math.ceil(self.total_samples_for_this_replica / self.batch_size)
        
        self.num_batches_per_replica = self._sync_num_batches(local_num_batches)
        self.recompute_probabilities()

    def _sync_num_batches(self, local_num_batches):
        if dist.is_available() and dist.is_initialized():
            count_tensor = torch.tensor([local_num_batches], dtype=torch.long, device=torch.device(f"cuda:{self.rank % torch.cuda.device_count()}"))
            dist.all_reduce(count_tensor, op=dist.ReduceOp.MIN)
            return count_tensor.item()
        return local_num_batches

    def set_adaptive_mode(self, enabled: bool):
        self.adaptive_enabled = enabled
        self.recompute_probabilities()

    def update_task_difficulty(self, dataset_idx: int, grad_norm: float):
        if dist.is_available() and dist.is_initialized():
             device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
             g_tensor = torch.tensor([grad_norm], dtype=torch.float32, device=device)
             dist.all_reduce(g_tensor, op=dist.ReduceOp.AVG) 
             grad_norm = g_tensor.item()

        old_g = self.task_difficulties[dataset_idx]
        new_g = self.alpha * old_g + (1 - self.alpha) * grad_norm
        self.task_difficulties[dataset_idx] = new_g

    def recompute_probabilities(self):
        if not self.adaptive_enabled:
            self.probabilities = None 
            return

        log_scores = []
        for i in range(self.num_datasets):
            N_k = self.dataset_lengths[i]
            G_k = self.task_difficulties[i]
            
            # log(S_k) = lambda * log(N_k) + G_k / temp
            log_scale_term = self.lam * math.log(N_k + 1e-8)
            log_difficulty_term = G_k / self.temp
            log_scores.append(log_scale_term + log_difficulty_term)
        
        # Softmax Trick
        max_log_score = max(log_scores)
        raw_scores = [math.exp(ls - max_log_score) for ls in log_scores]
        total_raw_score = sum(raw_scores)
        
        if total_raw_score == 0:
            raw_probs = [1.0 / self.num_datasets] * self.num_datasets
        else:
            raw_probs = [s / total_raw_score for s in raw_scores]

        # Floor Mechanism
        floored_probs = [max(p, self.min_prob) for p in raw_probs]
        
        # Re-normalize
        sum_floored = sum(floored_probs)
        self.probabilities = [p / sum_floored for p in floored_probs]

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.recompute_probabilities()

    def __iter__(self) -> Iterator[List[int]]:

        decision_seed = self.seed + self.epoch * 1000
        rng_decision = random.Random(decision_seed)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            shuffle_seed = self.seed + self.rank + self.epoch * 1000
        else:
            shuffle_seed = self.seed + self.rank * worker_info.num_workers + worker_info.id + self.epoch * 1000
        
        rng_shuffle = random.Random(shuffle_seed)
        
        # ------------------------------------------

        shuffled_replica_ds_indices: List[deque[int]] = []
        for ds_idx in range(self.num_datasets):
            current_ds_replica_indices = list(self.replica_dataset_indices[ds_idx]) 

            rng_shuffle.shuffle(current_ds_replica_indices) 
            shuffled_replica_ds_indices.append(deque(current_ds_replica_indices))

        batches_yielded = 0 
        
        while batches_yielded < self.num_batches_per_replica:
            
            chosen_ds_idx = -1 
            
            if self.probabilities is None:
                chosen_ds_idx = rng_decision.randint(0, self.num_datasets - 1)
            else:
                choice_idx = rng_decision.choices(range(self.num_datasets), weights=self.probabilities, k=1)[0]
                chosen_ds_idx = choice_idx

            current_ds_pool = shuffled_replica_ds_indices[chosen_ds_idx]

            if not current_ds_pool:
                new_indices = list(self.replica_dataset_indices[chosen_ds_idx])
                rng_shuffle.shuffle(new_indices)
                shuffled_replica_ds_indices[chosen_ds_idx] = deque(new_indices)
                current_ds_pool = shuffled_replica_ds_indices[chosen_ds_idx]
                
            num_samples_to_take = self.batch_size
            
            if len(current_ds_pool) < self.batch_size:
                if self.drop_last:
                      new_indices = list(self.replica_dataset_indices[chosen_ds_idx])
                      rng_shuffle.shuffle(new_indices)
                      current_ds_pool.extend(new_indices)
                      num_samples_to_take = self.batch_size
                else:
                    num_samples_to_take = len(current_ds_pool)

            batch_internal_indices = []
            for _ in range(num_samples_to_take):
                internal_idx = current_ds_pool.popleft() 
                batch_internal_indices.append(internal_idx)
            
            batch_global_indices = [idx + self.global_index_offsets[chosen_ds_idx] for idx in batch_internal_indices]
            
            yield batch_global_indices
            batches_yielded += 1

    def __len__(self):
        return self.num_batches_per_replica


class FashionCollator:
    def __init__(self, processor, min_pixel=224*224, max_pixel=256*256, mean_token="<|MEAN|>", img_token="<|IMG|>"):
        self.processor = processor
        self.mean_token = mean_token

        self.img_token = img_token
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel

    def __call__(self, batch):
        query_messages = []
        target_messages = []
        target_owner_ids = []

        query_num_images = []
        query_num_videos = []
        target_num_images = []
        target_num_videos = []
        dataset_names = []

        for i, sample in enumerate(batch):
            ### query
            query_messages_for_current_sample = [
                {"role": "system", "content": [{"type": "text", "text": "You are a fashion expert in processing fashion images."}]}
            ]
            query_messages_for_current_sample.append({
                "role": "user",
                "content": [{"type": "text", "text": sample['instruction_text']}]
            })

            num_imgs = 0
            if sample['query_image'] is not None:
                num_imgs = len(sample['query_image'])
                for img in sample['query_image']:
                    query_messages_for_current_sample[-1]['content'].append({
                        "type": "image", "image": img, "min_pixels": self.min_pixel, "max_pixels": self.max_pixel
                    })
            query_num_images.append(num_imgs)

            num_vids = 0
            if sample['query_video'] is not None:
                num_vids = len(sample['query_video'])
                for vid in sample['query_video']:
                    query_messages_for_current_sample[-1]['content'].append({
                        "type": "video", "video": vid, 'nframes':8, "resized_height": 224, "resized_width": 224
                    })
            query_num_videos.append(num_vids)

            query_messages_for_current_sample[-1]['content'].append({
                "type": "text",
                "text": "Summarize the semantic feature of the target fashion image in one word."
            })
            query_messages_for_current_sample.append({
                "role": "assistant",
                "content": [{"type": "text", "text": self.mean_token}]
            })
            query_messages.append(query_messages_for_current_sample)

            ### target
            for img in sample['target_image']:
                i_target_image_message = [
                    {"role": "system", "content": [{"type": "text", "text": "You are a fashion expert in processing fashion images."}]},  
                    {"role": "user", "content": [
                        {"type": "image", "image": img, "min_pixels": self.min_pixel, "max_pixels": self.max_pixel},
                        {"type": "text", "text": "Summarize the image in one word."}
                    ]},
                    {"role": "assistant", "content": [{"type": "text", "text": self.img_token}]}
                ]
                target_messages.append(i_target_image_message)
                target_owner_ids.append(i)
                target_num_images.append(1)
                target_num_videos.append(0)
            
            dataset_names.append(sample['dataset'])

        text_query = self.processor.apply_chat_template(query_messages, tokenize=False, add_generation_prompt=False)
        query_images, query_videos, query_video_kwargs = process_vision_info(query_messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        
        if query_videos is not None:
            query_videos, query_video_metadatas = zip(*query_videos)
            query_videos, query_video_metadatas = list(query_videos), list(query_video_metadatas)
        else:
            query_video_metadatas = None
            
        query_inputs = self.processor(text=text_query, images=query_images, videos=query_videos, video_metadata=query_video_metadatas, return_tensors="pt", do_resize=False, **query_video_kwargs, padding=True)


        text_target = self.processor.apply_chat_template(target_messages, tokenize=False, add_generation_prompt=False)
        target_images, target_videos, target_video_kwargs = process_vision_info(target_messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        if target_videos is not None:
            target_videos, target_video_metadatas = zip(*target_videos)
            target_videos, target_video_metadatas = list(target_videos), list(target_video_metadatas)
        else:
            target_video_metadatas = None
        target_inputs = self.processor(text=text_target, images=target_images, videos=target_videos, video_metadata=target_video_metadatas, return_tensors="pt", do_resize=False, **target_video_kwargs, padding=True)


        return {
            "query_inputs": query_inputs,
            "target_inputs": target_inputs, 
            "target_owner_ids": torch.tensor(target_owner_ids, dtype=torch.long), 
            "query_num_images": query_num_images,  
            "query_num_videos": query_num_videos,  
            "target_num_images": target_num_images, 
            "target_num_videos": target_num_videos, 
            'dataset_names': dataset_names
        }
