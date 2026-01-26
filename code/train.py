#ulimit -n 65536
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch 
torch.multiprocessing.set_sharing_strategy('file_system')
import json
import time
import math
import random
import numpy as np
from contextlib import nullcontext
from typing import List, Dict
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import gc
from model import FashionLens
from my_grad_cache import GradCache
from tqdm import tqdm
from collections import defaultdict

from dataset import FashionDataset, CombinedDataset, InterleavingBatchSampler, FashionCollator
from evaluate import evaluate_on_val 
from test import evaluate_on_test

train_files_chunk_size = {
    '1_t2i_FashionGen_train_triplet_data.json': 8,
    '1_t2i_Shoes_train_triplet_data.json': 32,
    '2_sketch2i_ClothesV1_train_triplet_data.json': 16,
    '2_sketch2i_HAIFashion_train_triplet_data.json': 16,
    '2_sketch2i_QMUL-Shoe-V2_train_triplet_data.json': 16,
    '3_inshop_DeepFashion_train_triplet_data.json': 8,
    '4_street2shop_DeepFashion2_train_triplet_data.json': 8,
    '5_compatibility_FashionVC_train_triplet_data.json': 32,
    '5_compatibility_Polyvore_train_triplet_data.json': 8,
    '6_video2img_MovingFashion_train_triplet_data.json': 2,
    '7_cir_Fashion200K_train_triplet_data.json': 8,
    '7_cir_FashionIQ-Dress_train_triplet_data.json': 32,
    '7_cir_FashionIQ-Shirt_train_triplet_data.json': 32,
    '7_cir_FashionIQ-Toptee_train_triplet_data.json': 32,
    '7_cir_Shoes_train_triplet_data.json': 32,
    '8_sketch&text_CSTBIR_train_triplet_data.json': 32,
    '9_asfr_DARN_train_triplet_data.json': 4,
    '9_asfr_FashionAI_train_triplet_data.json': 4,
}

def unwrap_model(m):
    return m.module if hasattr(m, "module") else m

def get_trainable_state_dict(model):
    trainable_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    full_state_dict = model.state_dict()
    trainable_state_dict = {k: v for k, v in full_state_dict.items() if k in trainable_param_names}
    return trainable_state_dict

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(args, rank, world_size, local_rank):

    if world_size > 1:
        setup_ddp(rank, world_size, local_rank)
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"Rank {rank}/{world_size} (Local Rank {local_rank}): Using device {device}")

    data_list = []
    dataset_name_to_idx = {}
    # Sort keys to ensure consistent indexing across processes
    sorted_keys = sorted(list(train_files_chunk_size.keys())) 
    for idx, train_triplet_path in enumerate(sorted_keys):
        data_list.append(FashionDataset(args.dataset_path, train_triplet_path))
        dataset_name_to_idx[train_triplet_path] = idx

    CombinedFashionDataset = CombinedDataset(data_list)

    model = FashionLens(args.qwen_path, torch.bfloat16).to(device)
    model_processor = model.processor
    collate_fn = FashionCollator(model_processor, 224 * 224, 224 * 224, model.mean_token, model.img_token)

    opt_params = list(unwrap_model(model).parameters())

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        if rank == 0:
            print(f"Models wrapped with DDP (find_unused_parameters=True).")
    
    grad_cache_manager = GradCache(
        model = model,
        temperature = args.temperature,
        beta = args.beta,
        K = args.train_k,
        fp16_autocast_enabled = True
    )

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, opt_params), lr=args.lr, weight_decay=args.weight_decay)

    rank_sampler = InterleavingBatchSampler(
        datasets=data_list,
        batch_size=args.batch_size,
        adaptive_alpha=args.adaptive_alpha,
        adaptive_lam=args.adaptive_lam,
        adaptive_temp=args.adaptive_temp,
        adaptive_min_prob=args.adaptive_min_prob,
        seed=42,
        num_replicas=world_size,
        rank=rank,
        drop_last=True
    )

    dataloader = DataLoader(
        CombinedFashionDataset,
        batch_sampler=rank_sampler,
        collate_fn=collate_fn,
        num_workers = args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker 
    )

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = os.path.join(args.output_dir, "training_log.json")
        best_val_path = os.path.join(args.output_dir, "best_val_results.json")
        best_test_path = os.path.join(args.output_dir, "best_test_results.json")
        best_val_recall_at_1 = -1.0
        train_log = [] 

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        if rank == 0:
            print(f"Epoch {epoch} start")
            epoch_dataset_counts = defaultdict(int)
            epoch_total_steps = 0
        if epoch < args.adaptive_warmup_epochs:
            rank_sampler.set_adaptive_mode(False) 
            if rank == 0: print(f"Adaptive Sampling: OFF (Warmup phase)")
        else:
            rank_sampler.set_adaptive_mode(True) 
            if rank == 0: print(f"Adaptive Sampling: ON")

        if hasattr(dataloader.batch_sampler, 'set_epoch'):
            dataloader.batch_sampler.set_epoch(epoch)

        step_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(rank!=0))
        for step, batch in enumerate(step_iterator):
            optimizer.zero_grad()   
            


            query_inputs = batch['query_inputs'].to(device)
            target_inputs = batch['target_inputs'].to(device) 
            target_owner_ids = batch['target_owner_ids'].to(device) 
            dataset_names = batch['dataset_names']
            if rank == 0:
                current_ds_name = dataset_names[0]
                epoch_dataset_counts[current_ds_name] += 1
                epoch_total_steps += 1

            loss, reg_loss = grad_cache_manager.cache_step_inbatch(
                query_inputs, batch['query_num_images'], batch['query_num_videos'],
                target_inputs, batch['target_num_images'], batch['target_num_videos'],
                target_owner_ids,
                train_files_chunk_size[dataset_names[0]],
                no_sync_except_last=True if dist.is_initialized() else False
            )

            current_dataset_name = dataset_names[0] 
            current_ds_idx = dataset_name_to_idx[current_dataset_name]

            raw_model = unwrap_model(model)
            anchor_grads = []
            target_params = [
                raw_model.learnable_token_MEAN, 
                raw_model.learnable_token_IMG
            ]
            for param in target_params:
                if param.grad is not None:
                    anchor_grads.append(param.grad.view(-1))
            
            if anchor_grads:
                all_grads = torch.cat(anchor_grads)
                grad_norm = torch.norm(all_grads, p=2).item()
            else:
                grad_norm = 0.0
            
            rank_sampler.update_task_difficulty(current_ds_idx, grad_norm)
           
            optimizer.step()

            del query_inputs, target_inputs
            global_step = epoch * len(dataloader) + step

            if rank == 0 and step % args.log_interval == 0:
                dataset_ratios = {
                                    k: f"{(v / epoch_total_steps) * 100:.2f}%" 
                                    for k, v in epoch_dataset_counts.items()
                                }
                dataset_difficulties = {}
                dataset_probs = {}
                for name, idx in dataset_name_to_idx.items():
        
                    diff_val = rank_sampler.task_difficulties[idx]
                    dataset_difficulties[name] = round(diff_val, 4)
      
                    if rank_sampler.probabilities is not None:
                        prob_val = rank_sampler.probabilities[idx]
                        dataset_probs[name] = f"{prob_val * 100:.2f}%"
                    else:
                        dataset_probs[name] = f"{(1.0 / len(dataset_name_to_idx)) * 100:.2f}%"
                log_entry = {
                    "mode": "train", "epoch": epoch, "step": global_step,
                    "contrastive_loss": loss.item(), "reg_loss": reg_loss,
                    "lr": optimizer.param_groups[0]['lr'],
                    "dataset_ratios": dataset_ratios,           
                    "dataset_target_probs": dataset_probs,      
                    "dataset_raw_difficulties": dataset_difficulties 
                }
                train_log.append(log_entry)
                with open(log_file, "w") as f:
                    json.dump(train_log, f, indent=4)

            if step % args.eval_steps == 0 and step != 0:
                torch.cuda.empty_cache()
                if dist.is_initialized():
                    dist.barrier()

                if rank == 0: print(f"\n[Step {global_step}] Running Validation...")
                val_results = evaluate_on_val(args, model, model_processor, world_size > 1, local_rank, world_size)

                if rank == 0:
                    log_entry = {"mode": "validation", "epoch": epoch, "step": global_step, "results": val_results}
                    train_log.append(log_entry)
                    
                    current_val_recall_at_1 = 0
                    if val_results:
                        for task_name, metrics in val_results.items():
                            current_val_recall_at_1 += metrics.get('Recall@1', 0)
                    
                    print(f"Current Val Recall@1 Sum: {current_val_recall_at_1:.4f} (Best: {best_val_recall_at_1:.4f})")

                    new_best_found = torch.tensor(0, dtype=torch.int, device=device)
                    if current_val_recall_at_1 > best_val_recall_at_1:
                        best_val_recall_at_1 = current_val_recall_at_1
                        new_best_found = torch.tensor(1, dtype=torch.int, device=device)
                        print(f"!!! New Best Model Found !!!")
                        
                        with open(best_val_path, "w") as f:
                            json.dump(val_results, f, indent=4)
                        
                        model_to_save = unwrap_model(model)
                        save_dict = get_trainable_state_dict(model_to_save)
                        torch.save({
                            'model_state_dict': save_dict,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_score': best_val_recall_at_1
                        }, os.path.join(args.output_dir, "best_model.pth"))

                    if dist.is_initialized():
                        dist.broadcast(new_best_found, src=0)
                else:
                    new_best_found = torch.tensor(0, dtype=torch.int, device=device)
                    if dist.is_initialized():
                        dist.broadcast(new_best_found, src=0)

                if new_best_found.item() == 1:
                    torch.cuda.empty_cache()
                    if rank == 0: print(f"Running Test on New Best Model...")
                    test_results = evaluate_on_test(args, model, model_processor, world_size > 1, local_rank, world_size)
                    
                    if rank == 0:
                        log_entry = {"mode": "test", "epoch": epoch, "step": global_step, "results": test_results}
                        train_log.append(log_entry)
                        with open(best_test_path, "w") as f:
                            json.dump(test_results, f, indent=4)

                if rank == 0:
                    with open(log_file, "w") as f:
                        json.dump(train_log, f, indent=4)

                torch.cuda.empty_cache()
                model.train()
        if rank == 0:
            final_ratios = {
                k: f"{(v / epoch_total_steps) * 100:.2f}%" 
                for k, v in epoch_dataset_counts.items()
            }

            log_entry = {
                "mode": "train_epoch_end_summary", 
                "epoch": epoch,
                "total_steps_in_this_epoch": epoch_total_steps,
                "final_dataset_ratios": final_ratios,
                "final_dataset_difficulties": {
                    name: round(rank_sampler.task_difficulties[dataset_name_to_idx[name]], 4)
                    for name in dataset_name_to_idx
                }
            }
            train_log.append(log_entry)
            with open(log_file, "w") as f:
                json.dump(train_log, f, indent=4)
                
        if dist.is_initialized():
            dist.barrier()

    cleanup_ddp()

def setup_ddp(rank, world_size, local_rank):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_path", type=str, default="/root/autodl-tmp/Qwen3-VL-4B-Instruct")
    parser.add_argument("--dataset_path", type=str, default="/root/autodl-tmp/Fashion_Retrieval")
    parser.add_argument("--base_path", type=str, default="/root/autodl-tmp/Fashion_Retrieval")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--beta", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--train_k", type=int, default=8)
    parser.add_argument("--test_k", type=int, default=5)
    parser.add_argument("--adaptive_min_prob", type=float, default=0.02, 
                        help="Minimum probability for each dataset to prevent skipping.")

    parser.add_argument("--adaptive_warmup_epochs", type=int, default=1)
    parser.add_argument("--adaptive_alpha", type=float, default=0.9)
    parser.add_argument("--adaptive_lam", type=float, default=0.5)
    parser.add_argument("--adaptive_temp", type=float, default=1.0)

    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    print(f"Main entry (Process {os.getpid()}): RANK={rank}, WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
    train(args, rank, world_size, local_rank)

if __name__ == "__main__":
    main()