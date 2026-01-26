import os 
import json 
from torch.utils.data import Dataset, Sampler
from qwen_vl_utils import process_vision_info
import torch.distributed as dist
import torch 
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from tqdm import tqdm 
from utils import calculate_recall_at_k, distributed_concat
import torch.nn.functional as F

val_files = [
    '1_t2i_FashionGen_val_triplet_data.json',
    '1_t2i_Shoes_val_triplet_data.json',
    '2_sketch2i_ClothesV1_val_triplet_data.json',
    '2_sketch2i_HAIFashion_val_triplet_data.json',
    '2_sketch2i_QMUL-Shoe-V2_val_triplet_data.json',
    '3_inshop_DeepFashion_val_triplet_data.json',
    '4_street2shop_DeepFashion2_val_triplet_data.json',
    '5_compatibility_FashionVC_val_triplet_data.json',
    '5_compatibility_Polyvore_val_triplet_data.json',
    '6_video2img_MovingFashion_val_triplet_data.json',
    '7_cir_Fashion200K_val_triplet_data.json',
    '7_cir_FashionIQ-Dress_val_triplet_data.json',
    '7_cir_FashionIQ-Shirt_val_triplet_data.json',
    '7_cir_FashionIQ-Toptee_val_triplet_data.json',
    '7_cir_Shoes_val_triplet_data.json',
    '8_sketch&text_CSTBIR_val_triplet_data.json',
    '9_asfr_DARN_val_triplet_data.json',
    '9_asfr_FashionAI_val_triplet_data.json',
]

class Fashion_VAL_dataset_query(Dataset):
    def __init__(self, base_path, val_triplet_path, image_path):
        with open(os.path.join(base_path, '0_triplet_files', val_triplet_path), 'r') as f:
            self.val_data = json.load(f)
        self.base_path = base_path

        self.all_images = sorted(os.listdir(os.path.join(base_path, image_path)))

        if 'asfr' in val_triplet_path:
            with open(os.path.join(base_path, '0_triplet_files', val_triplet_path.replace('triplet', 'candidate')), 'r') as f:
                self.val_label_data = json.load(f)
    def __len__(self):
        return len(self.val_data)
    
    def __getitem__(self, idx):
        data = self.val_data[idx]

        instruction_text = data['instruction_text']

        if 'query_image' in data:
            query_image = [os.path.join(self.base_path, p) for p in data['query_image']][0:8]
        else:
            query_image = None 
        
        if 'query_video' in data:
            query_video = [os.path.join(self.base_path, p) for p in data['query_video']]
        else:
            query_video = None
        
        true_candidate_global_ids = []
        
        if 'target_image' in data:
            for img in data['target_image']:
                img_name = img.split('/')[-1]
                true_candidate_global_ids.append(self.all_images.index(img_name))
        
        if 'target_label' in data:
            target_label = data['target_label']
            target_images = self.val_label_data[target_label]
            for img in target_images:
                img_name = img.split('/')[-1]
                true_candidate_global_ids.append(self.all_images.index(img_name))
        
        true_candidate_global_ids = list(set(true_candidate_global_ids))

        return {
            'instruction_text': instruction_text,
            'query_image': query_image,
            'query_video': query_video,
            'target_global_ids': true_candidate_global_ids
        }
    
class Fashion_VAL_dataset_candidate(Dataset):
    def __init__(self, base_path, val_triplet_path, image_path):
        with open(os.path.join(base_path, '0_triplet_files', val_triplet_path), 'r') as f:
            self.val_data = json.load(f)
        self.base_path = base_path

        self.all_images = sorted(os.listdir(os.path.join(base_path, image_path)))

        self.candidate_images = []
        self.image_labels = []

        if 'asfr' in val_triplet_path:
            with open(os.path.join(base_path, '0_triplet_files', val_triplet_path.replace('triplet', 'candidate')), 'r') as f:
                self.val_label_data = json.load(f)
            
            for label, images in self.val_label_data.items():
                for img in images:
                    if os.path.join(self.base_path, img) not in self.candidate_images:
                        self.candidate_images.append(os.path.join(self.base_path, img))
                        self.image_labels.append(label)

        else:
            for data in self.val_data:
                target_images = data['target_image']
                for img in target_images:
                    if os.path.join(self.base_path, img) not in self.candidate_images:
                        self.candidate_images.append(os.path.join(self.base_path, img))
    def __len__(self):
        return len(self.candidate_images)
    
    def __getitem__(self, idx):
        data = self.candidate_images[idx]
        if len(self.image_labels) > 0:
            label = self.image_labels[idx]
        else:
            label = None
        
        candidate_global_id = self.all_images.index(data.split('/')[-1])

        return {
            'image': data,
            'candidate_global_id': candidate_global_id,
            'label': label
        }


class Fashion_VAL_query_Collator:
    def __init__(self, processor, min_pixel=224*224, max_pixel=256*256, mean_token="<|MEAN|>", img_token="<|IMG|>"):
        self.processor = processor
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.mean_token = mean_token
        self.img_token = img_token
    
    def __call__(self, batch):
        query_messages = []
        # batch_target_global_ids = []

        for i, sample in enumerate(batch):

            query_messages_for_current_sample = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a fashion expert in processing fashion images."
                        }
                    ]
                }
            ]
            query_messages_for_current_sample.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": sample['instruction_text']
                    }
                ]
            })

            if sample['query_image'] is not None:
                for img in sample['query_image']:
                    query_messages_for_current_sample[-1]['content'].append({
                        "type": "image",
                        "image": img,
                        "min_pixels": self.min_pixel,
                        "max_pixels": self.max_pixel
                    })
            if sample['query_video'] is not None:
                for vid in sample['query_video']:
                    query_messages_for_current_sample[-1]['content'].append({
                        "type": "video",
                        "video": vid,
                        # "min_pixels": self.min_pixel,
                        # "max_pixels": self.max_pixel,
                        # "total_pixels": self.max_pixel * 8,
                        'nframes':8,
                        "resized_height": 224, "resized_width": 224
                    })
            query_messages_for_current_sample[-1]['content'].append({
                "type": "text",
                "text": "Summarize the semantic feature of the target fashion image in one word."
            })
            
            query_messages_for_current_sample.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": self.mean_token
                    }
                ]
            })
            query_messages.append(query_messages_for_current_sample)

        

            # batch_target_global_ids.append(sample['target_global_ids'])
        all_samples_target_global_ids = [sample['target_global_ids'] for sample in batch]
        
        text_query = self.processor.apply_chat_template(query_messages, tokenize=False, add_generation_prompt=False)
        query_images, query_videos, query_video_kwargs = process_vision_info(query_messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        if query_videos is not None:
            query_videos, query_video_metadatas = zip(*query_videos)
            query_videos, query_video_metadatas = list(query_videos), list(query_video_metadatas)
        else:
            query_video_metadatas = None
        query_inputs = self.processor(text=text_query, images=query_images, videos=query_videos, video_metadata=query_video_metadatas, return_tensors="pt", do_resize=False, **query_video_kwargs, padding = True)

        return {
            'query_inputs': query_inputs,
            'target_global_ids': all_samples_target_global_ids
        }
    

class Fashion_VAL_candidate_Collator:
    def __init__(self, processor, min_pixel=224*224, max_pixel=256*256, mean_token="<|MEAN|>", img_token="<|IMG|>"):
        self.processor = processor
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.mean_token = mean_token
        self.img_token = img_token
    
    def __call__(self, batch):
        target_messages = []
        batch_candidate_global_ids = []

        for i, sample in enumerate(batch):
            target_messages_for_current_sample = [
                 {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a fashion expert in processing fashion images."
                        }
                    ]
                }
            ]

  
            target_messages_for_current_sample.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": sample['image'],
                        "min_pixels": self.min_pixel,
                        "max_pixels": self.max_pixel
                    },
                    {
                        "type": "text",
                        "text": "Summarize the image in one word."
                    }
                ]
            })
            target_messages_for_current_sample.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": self.img_token
                    }
                ]
            })

            
            target_messages.append(target_messages_for_current_sample)
            batch_candidate_global_ids.append(sample['candidate_global_id'])

        text_target = self.processor.apply_chat_template(target_messages, tokenize=False, add_generation_prompt=False)
        target_images, target_videos, target_video_kwargs = process_vision_info(target_messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        if target_videos is not None:
            target_videos, target_video_metadatas = zip(*target_videos)
            target_videos, target_video_metadatas = list(target_videos), list(target_video_metadatas)
        else:
            target_video_metadatas = None
        target_inputs = self.processor(text=text_target, images=target_images, videos=target_videos, video_metadata=target_video_metadatas, return_tensors="pt", do_resize=False, **target_video_kwargs, padding = True)

        return {
            'candidate_inputs': target_inputs,
            'candidate_global_id': torch.tensor(batch_candidate_global_ids, dtype=torch.long)
        }


def evaluate_on_val(args, model, processor, distributed, local_rank, world_size):
    if distributed:
        rank = local_rank 
        device = torch.device(f"cuda:{rank}")
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    all_results = {}
    
    for sub_task in val_files:
        dataset_name = sub_task.split('_')[2]
        task_name = sub_task.split('_')[0] + '_' + sub_task.split('_')[1]
        if 'Dress' in dataset_name:
            images_path = 'FashionIQ_images/Dress_images'
        elif 'Shirt' in dataset_name:
            images_path = 'FashionIQ_images/Shirt_images'
        elif 'Toptee' in dataset_name:
            images_path = 'FashionIQ_images/Toptee_images'
        else:
            images_path = dataset_name + '_images'
        query_dataset = Fashion_VAL_dataset_query(args.base_path, sub_task, images_path)
        candiate_data = Fashion_VAL_dataset_candidate(args.base_path, sub_task, images_path)

        if distributed:
            query_sampler = DistributedSampler(query_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
            query_dataloader = DataLoader(query_dataset, batch_size=args.eval_batch_size, sampler=query_sampler, collate_fn=Fashion_VAL_query_Collator(processor), num_workers=args.num_workers)

            candidate_sampler = DistributedSampler(candiate_data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
            candidate_dataloader = DataLoader(candiate_data, batch_size=args.eval_batch_size, sampler=candidate_sampler, collate_fn=Fashion_VAL_candidate_Collator(processor), num_workers=args.num_workers)
        else:
            query_dataloader = DataLoader(query_dataset, batch_size=args.eval_batch_size, collate_fn=Fashion_VAL_query_Collator(processor), shuffle=False, drop_last=False, num_workers=args.num_workers)
            candidate_dataloader = DataLoader(candiate_data, batch_size=args.eval_batch_size, collate_fn=Fashion_VAL_candidate_Collator(processor), shuffle=False, drop_last=False, num_workers=args.num_workers)
        
        # 提取query特征
        local_query_embeddings = []
        #local_query_true_candidate_global_ids_lists = []
        all_local_query_true_candidate_global_ids = []
        for batch in tqdm(query_dataloader, desc=f"Rank {rank} Query Embedding", disable=(rank!=0)):
            inputs = batch['query_inputs'].to(device)
            with torch.no_grad():
                query_embeds = raw_model.forward_query(inputs)[0]
                # query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)

            local_query_embeddings.append(query_embeds.cpu())
            # local_query_true_candidate_global_ids_lists.append(batch['target_global_ids'])
            for target_id_list_for_single_query in batch['target_global_ids']:
                all_local_query_true_candidate_global_ids.append(target_id_list_for_single_query)
        local_query_embeddings = torch.cat(local_query_embeddings, dim=0)

        # 提取candidate特征
        local_candidate_embeddings = []
        local_candidate_global_ids = []
        for batch in tqdm(candidate_dataloader, desc=f"Rank {rank} Candidate Embedding", disable=(rank!=0)):
            inputs = batch['candidate_inputs'].to(device)
            with torch.no_grad():
                candidate_embeds = raw_model.forward_target(inputs)
            
            local_candidate_embeddings.append(candidate_embeds.cpu())
            local_candidate_global_ids.append(batch['candidate_global_id'].cpu())

        local_candidate_embeddings = torch.cat(local_candidate_embeddings, dim=0)
        local_candidate_global_ids = torch.cat(local_candidate_global_ids, dim=0)


        if distributed:
            dist.barrier()

            # query
            global_query_embeddings = distributed_concat(local_query_embeddings.to(device)).cpu()  
            gathered_true_ids_lists_from_all_ranks = [None for _ in range(world_size)]
            #dist.all_gather_object(gathered_true_ids_lists, local_query_true_candidate_global_ids_lists)
            dist.all_gather_object(gathered_true_ids_lists_from_all_ranks, all_local_query_true_candidate_global_ids)

            # global_query_true_candidate_global_ids_lists = [item for sublist in gathered_true_ids_lists for item in sublist]
            global_query_true_candidate_global_ids_lists = [
                item for sublist in gathered_true_ids_lists_from_all_ranks for item in sublist
            ]


            # candidate
            global_candidate_embeddings = distributed_concat(local_candidate_embeddings.to(device)).cpu()
            global_candidate_ids = distributed_concat(local_candidate_global_ids.to(device)).cpu()

            if rank == 0:
                unique_candidate_id_to_index = {}
                final_global_candidate_embeddings_list = [] 

                for i, cand_id in enumerate(global_candidate_ids):
                    if cand_id not in unique_candidate_id_to_index:
                        unique_candidate_id_to_index[cand_id.item()] = len(final_global_candidate_embeddings_list)
                        final_global_candidate_embeddings_list.append(global_candidate_embeddings[i])

                global_candidate_embeddings = torch.stack(final_global_candidate_embeddings_list)
            
            else:
                global_candidate_embeddings = None
                unique_candidate_id_to_index = None
        
        else:
            global_query_embeddings = local_query_embeddings
            #global_query_true_candidate_global_ids_lists = local_query_true_candidate_global_ids_lists
            global_query_true_candidate_global_ids_lists = all_local_query_true_candidate_global_ids
            global_candidate_embeddings = local_candidate_embeddings

            unique_candidate_id_to_index = {id.item(): idx for idx, id in enumerate(local_candidate_global_ids)}
        
        # 计算Recall@K
        results = {}
        if rank == 0:

            query_true_candidate_indices_for_recall = []
            # for true_global_ids_list in global_query_true_candidate_global_ids_lists:
            #     mapped_indices = [unique_candidate_id_to_index[gid] for gid in true_global_ids_list if gid in unique_candidate_id_to_index]
            #     query_true_candidate_indices_for_recall.append(mapped_indices)

            for true_global_ids_list_for_a_query in global_query_true_candidate_global_ids_lists: 
                mapped_indices_for_a_query = []
                for gid in true_global_ids_list_for_a_query: 
                    if gid in unique_candidate_id_to_index:
                        mapped_indices_for_a_query.append(unique_candidate_id_to_index[gid])
                query_true_candidate_indices_for_recall.append(mapped_indices_for_a_query)
            
            results = calculate_recall_at_k(
                global_query_embeddings,
                global_candidate_embeddings,
                query_true_candidate_indices_for_recall,
                k_values=[1, 5, 10, 50]
            )
     
        
        if distributed:
            dist.barrier()
            if rank == 0:
                gathered_results = [results]
            else:
                gathered_results = [None]
            dist.broadcast_object_list(gathered_results, src=0)
            results = gathered_results[0]
        
        all_results[task_name + '_' + dataset_name] = results

    return all_results