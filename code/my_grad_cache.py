from typing import List, Union, Callable, Any
from contextlib import nullcontext
from itertools import repeat
from collections import UserDict
import logging
from transformers import BatchFeature

import torch
from torch import nn, Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.distributed as dist 
import time 

from torch.nn import functional as F
from torch import distributed as dist

logger = logging.getLogger(__name__)
from torch.utils.checkpoint import get_device_states, set_device_states


class RandContext:
    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(
            devices=self.fwd_gpu_devices,
            enabled=True
        )
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


class SimpleContrastiveLoss:
    def __init__(self, n_hard_negatives: int = 0, temperature: float = 1.0, *args, **kwargs):
        self.target_per_qry = n_hard_negatives + 1
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean'):

        target = torch.arange(0, x.size(0), dtype = torch.long, device=x.device)
        logits = torch.matmul(x, y.transpose(0, 1)) / self.temperature
        loss = F.cross_entropy(logits, target, reduction=reduction)
        return loss

class Z2TContrastiveLoss:
    def __init__(self, temperature: float = 1.0, *args, **kwargs):
        self.temperature = temperature
    
    def __call__(self, query, target, target_owner_ids, reduction: str = 'mean'):
        B,K,D = query.shape
        N,D = target.shape
        device = query.device

        sim_k = torch.einsum('bkd,nd->bkn', query, target)  # B x K x N
        logits = torch.logsumexp(sim_k / self.temperature, dim=1)

        query_indices = torch.arange(B, device=device).unsqueeze(1)

        pos_mask = (query_indices == target_owner_ids.unsqueeze(0)).float()  # B x N
        
        log_prob = F.log_softmax(logits, dim=1)
        mask_sum_log_prob = (pos_mask * log_prob).sum(dim=1)

        num_positives = pos_mask.sum(dim=1)
        valid_indices = num_positives > 0
        if valid_indices.sum() == 0:
                    return torch.tensor(0.0, device=device, requires_grad=True)

        loss_per_query = -mask_sum_log_prob[valid_indices] / num_positives[valid_indices]
        loss = loss_per_query.mean()

        return loss


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, n_hard_negatives: int = 0, temperature: float = 1.0, *args, **kwargs):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."

        super().__init__(n_hard_negatives=n_hard_negatives, temperature=temperature)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)

        return super().__call__(dist_x, dist_y, **kwargs)

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)


class GradCache:
    """
    Gradient Cache class. Implements input chunking, first graph-less forward pass, Gradient Cache creation, second
    forward & backward gradient computation. Optimizer step is not included. Native torch automatic mixed precision is
    supported. User needs to handle gradient unscaling and scaler update after a gradeitn cache step.
    """
    def __init__(
        self,
        model,
        temperature: float = 0.07,
        beta: float = 1e-3,
        K: int = 1,
        fp16_autocast_enabled: bool = False,
        lambda_fro: float = 1e-4, 
        lambda_orth: float = 1e-2
    ):

        self.model = model

        self.loss_inbatch = SimpleContrastiveLoss(temperature=temperature)

        self.beta = beta
        self.K = K
        self.fp16_autocast_enabled = fp16_autocast_enabled

        self._get_input_tensors_strict = False
        self.lambda_fro = lambda_fro
        self.lambda_orth = lambda_orth


    def get_input_tensors(self, model_input) -> List[Tensor]:
        """
        Recursively go through model input and grab all tensors, which are then used to record current device random
        states. This method will do its best to parse types of Tensor, tuple, list, dict and UserDict. Other types will
        be ignored unless self._get_input_tensors_strict is set to True, in which case an exception will be raised.
        :param model_input: input to model
        :return: all torch tensors in model_input
        """
        if isinstance(model_input, Tensor):
            return [model_input]

        elif isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])

        elif isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), [])

        elif self._get_input_tensors_strict:
            raise NotImplementedError(f'get_input_tensors not implemented for type {type(model_input)}')

        else:
            return []
    def forward_query_no_grad(self, query_inputs_chunks):

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        rnd_states = []
        model_reps = []
        with torch.no_grad():
            for x in query_inputs_chunks:
                rnd_states.append(RandContext(*self.get_input_tensors(x)))

                autocast_ctx = torch.amp.autocast(
                                    device_type='cuda', 
                                    dtype=torch.bfloat16, 
                                    enabled=self.fp16_autocast_enabled
                                ) if torch.cuda.is_available() else nullcontext()

                with autocast_ctx:
                    z = raw_model.forward_query(x)[0]
                    model_reps.append(z)

                
        model_reps = torch.cat(model_reps, dim=0)
        return model_reps, rnd_states


    def forward_target_no_grad(self, target_inputs_chunks):

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        rnd_states = []
        model_reps = []
        with torch.no_grad():
            for x in target_inputs_chunks:
                rnd_states.append(RandContext(*self.get_input_tensors(x)))

                autocast_ctx = torch.amp.autocast(
                                    device_type='cuda', 
                                    dtype=torch.bfloat16, 
                                    enabled=self.fp16_autocast_enabled
                                ) if torch.cuda.is_available() else nullcontext()

                with autocast_ctx:
                    img = raw_model.forward_target(x)
                    model_reps.append(img)

                
        model_reps = torch.cat(model_reps, dim=0)
        return model_reps, rnd_states


    def inbatch_build_cache(self, query_reps, target_reps, target_owner_ids=None):

        query_reps = query_reps.detach().requires_grad_()
        target_reps = target_reps.detach().requires_grad_()

        autocast_ctx = torch.amp.autocast(
            device_type='cuda',
            dtype=torch.bfloat16,
            enabled=self.fp16_autocast_enabled
        ) if torch.cuda.is_available() else nullcontext()
        with autocast_ctx:
            loss = self.loss_inbatch(query_reps, target_reps, target_owner_ids)
        
        loss.backward()

        query_cache = query_reps.grad
        target_cache = target_reps.grad

        return query_cache, target_cache, loss.detach()

    def compute_sigma_constraint(self, sigma):

        target_scale = 0.1  
        var_prior = target_scale ** 2

        sigma = torch.clamp(sigma, min=1e-6, max=10.0) 
        var_post = sigma ** 2
        
        term1 = var_post / var_prior
        term2 = -torch.log(term1 + 1e-9) 
        
        kl_div = 0.5 * (term1 + term2 - 1)
        loss_reg = kl_div.mean()
        
        return loss_reg
    def forward_backward_query_inbatch(self, query_inputs, cached_gradients, random_states, no_sync_except_last):

        if no_sync_except_last:
            sync_contexts = [self.model.no_sync for _ in range(len(query_inputs) - 1)] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(query_inputs))]

        total_reg_loss = 0.0

        for x, state, gradient, sync_context in zip(query_inputs, random_states, cached_gradients, sync_contexts):
            with sync_context():
                with state:
                    autocast_ctx = torch.amp.autocast(
                                    device_type='cuda', 
                                    dtype=torch.bfloat16, 
                                    enabled=self.fp16_autocast_enabled
                                ) if torch.cuda.is_available() else nullcontext()
                    with autocast_ctx:
                        z, _, loss_fro, loss_orth = self.model(x, mode='query')
                        surrogate = torch.sum(z * gradient.to(z.dtype))

                        reg_loss = (self.lambda_fro * loss_fro) + (self.lambda_orth * loss_orth)
                        total_chunk_loss = surrogate + reg_loss

                    total_chunk_loss.backward()
                    total_reg_loss += reg_loss.item()
        return total_reg_loss
    
    def forward_backward_target_inbatch(self, target_inputs, cached_gradients, random_states, no_sync_except_last):
        if no_sync_except_last:
            sync_contexts = [self.model.no_sync for _ in range(len(target_inputs) - 1)] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(target_inputs))]
        
        for x, state, gradient, sync_context in zip(target_inputs, random_states, cached_gradients, sync_contexts):
            with sync_context():
                with state:
                    autocast_ctx = torch.amp.autocast(
                                    device_type='cuda', 
                                    dtype=torch.bfloat16, 
                                    enabled=self.fp16_autocast_enabled
                                ) if torch.cuda.is_available() else nullcontext()
                    with autocast_ctx:
                        img = self.model(x, mode='target')
                surrogate = torch.sum(img * gradient.to(img.dtype))
                surrogate.backward()



    def chunk_qwen3vl_inputs(self, inputs, chunk_size, num_images, num_videos):

        batch_size = inputs['input_ids'].shape[0]
        chunks = []

        img_token_ranges = []
        if "image_grid_thw" in inputs and "pixel_values" in inputs and len(inputs["image_grid_thw"]) > 0:
            start_idx = 0
            for g in inputs["image_grid_thw"]:  # g = [T, H, W]
                tokens = int(g[0] * g[1] * g[2])
                img_token_ranges.append((start_idx, start_idx + tokens))
                start_idx += tokens

        vid_token_ranges = []
        if "video_grid_thw" in inputs and "pixel_values_videos" in inputs and len(inputs["video_grid_thw"]) > 0:
            start_idx = 0
            for g in inputs["video_grid_thw"]:
                tokens = int(g[0] * g[1] * g[2])
                vid_token_ranges.append((start_idx, start_idx + tokens))
                start_idx += tokens

        img_start_idx = 0
        vid_start_idx = 0

        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            chunk = {}

            for k, v in inputs.items():
                if k in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
                    continue
                if isinstance(v, torch.Tensor) and v.shape[0] == batch_size:
                    chunk[k] = v[start:end]
                else:
                    chunk[k] = v

            if img_token_ranges:
                chunk_img_idx_start = img_start_idx
                chunk_img_idx_end = img_start_idx + sum(num_images[start:end])
                if chunk_img_idx_end > chunk_img_idx_start:  
                    chunk["image_grid_thw"] = inputs["image_grid_thw"][chunk_img_idx_start:chunk_img_idx_end]
                    pixel_start = img_token_ranges[chunk_img_idx_start][0]
                    pixel_end = img_token_ranges[chunk_img_idx_end - 1][1]
                    chunk["pixel_values"] = inputs["pixel_values"][pixel_start:pixel_end]
                img_start_idx = chunk_img_idx_end

            if vid_token_ranges:
                chunk_vid_idx_start = vid_start_idx
                chunk_vid_idx_end = vid_start_idx + sum(num_videos[start:end])
                if chunk_vid_idx_end > chunk_vid_idx_start:  
                    chunk["video_grid_thw"] = inputs["video_grid_thw"][chunk_vid_idx_start:chunk_vid_idx_end]
                    pixel_start = vid_token_ranges[chunk_vid_idx_start][0]
                    pixel_end = vid_token_ranges[chunk_vid_idx_end - 1][1]
                    chunk["pixel_values_videos"] = inputs["pixel_values_videos"][pixel_start:pixel_end]
                vid_start_idx = chunk_vid_idx_end

            chunks.append(BatchFeature(chunk))

        return chunks

    def cache_step_inbatch(
        self,
        query_inputs,
        query_num_images,
        query_num_videos,
        target_inputs,
        target_num_images,
        target_num_videos,
        target_owner_ids,
        chunk_size,
        no_sync_except_last: bool = False,
    ) -> Tensor:
        """
        Run a cached step to compute gradient over the inputs.
        :param model_inputs: Input to each encoder model. Should be in similar order as the class's model.
        :param no_sync_except_last: If True, under distributed setup, for each model, only trigger gradient reduction
        across processes for the last sub-batch's forward-backward pass.
        :param loss_kwargs: Additional keyword arguments to the loss function.
        :return: The current's loss.
        """
        query_model_inputs = self.chunk_qwen3vl_inputs(query_inputs, chunk_size, query_num_images, query_num_videos)
        target_model_inputs = self.chunk_qwen3vl_inputs(target_inputs, chunk_size, target_num_images, target_num_videos)
        query_chunk_sizes = [chunk["input_ids"].shape[0] for chunk in query_model_inputs]
        target_chunk_sizes = [chunk["input_ids"].shape[0] for chunk in target_model_inputs]

        # run the first forward pass
        all_query_reps, all_query_rnd_states = self.forward_query_no_grad(query_model_inputs)
        all_target_reps, all_target_rnd_states = self.forward_target_no_grad(target_model_inputs) 

        # build cache
        target_owner_ids = target_owner_ids.to(all_target_reps.device)
        query_cache, target_cache, loss = self.inbatch_build_cache(all_query_reps, all_target_reps, target_owner_ids)

        query_grad_chunks = query_cache.split(query_chunk_sizes)
        target_grad_chunks = target_cache.split(target_chunk_sizes)

        # run the second forward pass
        reg_loss = self.forward_backward_query_inbatch(query_model_inputs, query_grad_chunks, all_query_rnd_states, no_sync_except_last)
        self.forward_backward_target_inbatch(target_model_inputs, target_grad_chunks, all_target_rnd_states, no_sync_except_last)

        return loss, reg_loss


