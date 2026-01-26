import torch
import torch.distributed as dist
from typing import List, Dict

def calculate_probabilistic_recall_at_k(
    query_samples: torch.Tensor,       # [num_queries, K, dim]
    candidate_embeddings: torch.Tensor, # [num_candidates, dim]
    query_true_candidate_indices_for_recall: List[List[int]],
    k_values: List[int] = [1, 10, 50],
    temperature: float = 1.0,
    batch_size: int = 50, # Reduced batch size for safety
) -> Dict[str, float]:
    
    device = query_samples.device
    num_queries = len(query_samples)
    max_k = max(k_values)
    
    all_top_k_indices = []
    
    if torch.cuda.is_available():
        C = candidate_embeddings.cuda()
    else:
        C = candidate_embeddings

    for start_idx in range(0, num_queries, batch_size):
        end_idx = min(start_idx + batch_size, num_queries)
        
        if torch.cuda.is_available():
            Q_batch = query_samples[start_idx:end_idx].cuda()
        else:
            Q_batch = query_samples[start_idx:end_idx]
            
        # [B, K, M]
        sim_batch = torch.einsum('bkd,md->bkm', Q_batch.to(torch.float32), C.to(torch.float32))
        # sim_batch = torch.einsum('bkd,md->bkm', Q_batch, C)
        
        # LogSumExp -> [B, M]
        scores_batch = torch.logsumexp(sim_batch / temperature, dim=1)
        
        # TopK -> [B, max_k]
        _, indices_batch = torch.topk(scores_batch, k=max_k, dim=1)
        
        all_top_k_indices.append(indices_batch.cpu())
        
        del Q_batch, sim_batch, scores_batch, indices_batch
    
    top_k_indices = torch.cat(all_top_k_indices, dim=0).numpy()
    
    hits_at_k = {k: 0 for k in k_values}
    
    for i in range(num_queries):
        true_targets = set(query_true_candidate_indices_for_recall[i])
        if not true_targets: continue
            
        pred_indices = top_k_indices[i]
        first_hit_rank = float('inf')
        
        for rank, pred_idx in enumerate(pred_indices):
            if pred_idx in true_targets:
                first_hit_rank = rank + 1
                break
        
        for k in k_values:
            if first_hit_rank <= k:
                hits_at_k[k] += 1
                
    results = {}
    for k in k_values:
        results[f"Recall@{k}"] = hits_at_k[k] / num_queries
        
    return results

def calculate_recall_at_k(
    query_embeddings: torch.Tensor, 
    candidate_embeddings: torch.Tensor, 
    query_true_candidate_indices_for_recall: List[List[int]], 
    k_values: List[int] = [1, 10, 50],
    batch_size: int = 100
) -> Dict[str, float]:
    """
    Optimized Recall@K calculation with batching to avoid OOM.
    """
    num_queries = len(query_embeddings)
    max_k = max(k_values)
    all_top_k_indices = []

    if torch.cuda.is_available():
        C = candidate_embeddings.cuda()
    else:
        C = candidate_embeddings

    # Batch processing
    for start_idx in range(0, num_queries, batch_size):
        end_idx = min(start_idx + batch_size, num_queries)

        if torch.cuda.is_available():
            Q_batch = query_embeddings[start_idx:end_idx].cuda()
        else:
            Q_batch = query_embeddings[start_idx:end_idx]

        # [B, M]
        sim_batch = torch.matmul(Q_batch, C.t())
        
        # TopK [B, max_k]
        _, indices_batch = torch.topk(sim_batch, k=max_k, dim=1)
        all_top_k_indices.append(indices_batch.cpu())
        
        del Q_batch, sim_batch, indices_batch

    top_k_indices = torch.cat(all_top_k_indices, dim=0).numpy()
    
    # Calculate Metrics (CPU side)
    hits_at_k = {k: 0 for k in k_values}
    
    for i in range(num_queries):
        true_targets = set(query_true_candidate_indices_for_recall[i])
        if not true_targets:
            continue
            
        pred_indices = top_k_indices[i]
        first_hit_rank = float('inf')
        
        for rank, pred_idx in enumerate(pred_indices):
            if pred_idx in true_targets:
                first_hit_rank = rank + 1 
                break 
        
        for k in k_values:
            if first_hit_rank <= k:
                hits_at_k[k] += 1
                
    results = {}
    for k in k_values:
        results[f"Recall@{k}"] = hits_at_k[k] / num_queries
        
    return results

def distributed_concat(tensor: torch.Tensor, num_total_examples=None):
    """
    Safe concatenation across ranks allowing unequal tensor sizes.
    """
    local_size = torch.tensor([tensor.shape[0]], dtype=torch.long, device=tensor.device)
    
    all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, local_size)
    
    max_size = max([size.item() for size in all_sizes])
    
    # Pad to max size
    size_diff = max_size - local_size.item()
    if size_diff > 0:
        padding = torch.zeros((size_diff, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        tensor_padded = torch.cat([tensor, padding], dim=0)
    else:
        tensor_padded = tensor

    output_tensors = [torch.zeros_like(tensor_padded) for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor_padded)
    
    final_output = []
    for i, t in enumerate(output_tensors):
        real_size = all_sizes[i].item()
        final_output.append(t[:real_size])
    
    result = torch.cat(final_output, dim=0)
    return result