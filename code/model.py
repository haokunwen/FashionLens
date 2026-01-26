import torch 
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from typing import List, Callable
from qwen_vl_utils import process_vision_info


class SphericalHyperAdapter(nn.Module):
    def __init__(self, emb_dim, rank=32):
        super().__init__()
        self.emb_dim = emb_dim
        self.rank = rank
        
        self.hyper_net = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, (emb_dim * rank * 2) + 1) 
        )
        with torch.no_grad():
            last_layer = self.hyper_net[-1]

            nn.init.normal_(last_layer.weight, mean=0.0, std=1e-6) 
            nn.init.zeros_(last_layer.bias)

            last_layer.bias[-1] = -3.0

    def slerp(self, q, q_target, val):
        """
        q, q_target: [Batch, Dim] (Normalized)
        val: [Batch, 1] (Interpolation factor 0~1)
        """
        dot = (q * q_target).sum(dim=-1, keepdim=True)
        dot = torch.clamp(dot, -0.9995, 0.9995)
        
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)

        mask = sin_omega < 1e-6
        
        scale0 = torch.sin((1.0 - val) * omega) / (sin_omega + 1e-8)
        scale1 = torch.sin(val * omega) / (sin_omega + 1e-8)
        
        out = scale0 * q + scale1 * q_target

        if mask.any():
            linear_interp = (1 - val) * q + val * q_target
            linear_interp = F.normalize(linear_interp, dim=-1)
            out = torch.where(mask, linear_interp, out)
            
        return out

    def forward(self, query_emb):

        q_norm = F.normalize(query_emb, dim=-1)
        params = self.hyper_net(query_emb)
        
        matrix_params = params[:, :-1]
        lambda_logit = params[:, -1:]

        interp_val = torch.sigmoid(lambda_logit)
        batch_size = query_emb.size(0)
        params_A, params_B = torch.split(matrix_params, self.emb_dim * self.rank, dim=1)
        
        A = params_A.view(batch_size, self.emb_dim, self.rank)
        B = params_B.view(batch_size, self.rank, self.emb_dim)
 
        delta = (q_norm.unsqueeze(1) @ A @ B).squeeze(1)
        q_target = F.normalize(q_norm + delta, dim=-1)
    
        q_final = self.slerp(q_norm, q_target, interp_val)
        
        return q_final, interp_val 

    def get_regularization_loss(self, query_emb):

        params = self.hyper_net(query_emb)
        matrix_params = params[:, :-1] 
        
        batch_size = query_emb.size(0)

        params_A, params_B = torch.split(matrix_params, self.emb_dim * self.rank, dim=1)
        A = params_A.view(batch_size, self.emb_dim, self.rank)
        B = params_B.view(batch_size, self.rank, self.emb_dim)
        loss_fro_A = torch.norm(A, p='fro', dim=(1,2)).mean()
        loss_fro_B = torch.norm(B, p='fro', dim=(1,2)).mean()
        loss_fro = loss_fro_A + loss_fro_B
        
        # [B, R, D] @ [B, D, R] -> [B, R, R]
        gram_matrix = torch.bmm(A.transpose(1, 2), A) 

        identity = torch.eye(self.rank, device=A.device).unsqueeze(0).expand(batch_size, -1, -1)

        loss_orth = torch.norm(gram_matrix - identity, p='fro', dim=(1,2)).mean()

        return loss_fro, loss_orth


class FashionLens(nn.Module):
    def __init__(self, qwen_path: str, dtype: torch.dtype = torch.bfloat16):
        super(FashionLens, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
                    qwen_path,
                    torch_dtype = self.dtype,
                    attn_implementation = "flash_attention_2").to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
                    qwen_path,
                    padding_side = "right",
                    use_fast = True)
        
        for param in self.qwen.parameters():
            param.requires_grad = False
        
        self.mean_token = "<|MEAN|>"

        self.img_token = "<|IMG|>"
        
        self._init_embeddings()

        self.adapter = SphericalHyperAdapter(emb_dim=2560, rank=32).to(self.device).to(self.dtype)

        
        lora_config = LoraConfig(
            task_type = "CAUSAL_LM",
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            r = 8,
            lora_alpha = 16,
            lora_dropout = 0.05,
            bias = "none",
        )
        self.qwen = get_peft_model(self.qwen, lora_config)

    def _init_embeddings(self):
        num_added = self.processor.tokenizer.add_tokens([self.mean_token, self.img_token], special_tokens=True)
        if num_added > 0:
            print(f"Added {num_added} new tokens to the tokenizer.")
        self.qwen.resize_token_embeddings(len(self.processor.tokenizer))
        
        self.mean_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.mean_token)
        self.img_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.img_token)

        embedding_layer = self.qwen.get_input_embeddings()
        with torch.no_grad():
            im_end_token_id = self.processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
            im_end_embedding = embedding_layer(torch.tensor([im_end_token_id], dtype=torch.long, device=self.device))
        
        im_end_embedding = im_end_embedding[0].detach().to(self.dtype)
        self.learnable_token_IMG = nn.Parameter(im_end_embedding.clone())
        self.learnable_token_MEAN = nn.Parameter(im_end_embedding.clone())


    def forward(self, inputs, mode='query'):
        if mode == 'query':
            return self.forward_query(inputs)  
        elif mode == 'target':
            return self.forward_target(inputs) 
        else:
            raise ValueError("mode must be 'query' or 'target'")

    def forward_query(self, inputs):
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        pixel_values = inputs.pixel_values if "pixel_values" in inputs else None
        image_grid_thw = inputs.image_grid_thw if "image_grid_thw" in inputs else None
        pixel_values_videos = inputs.pixel_values_videos if "pixel_values_videos" in inputs else None
        video_grid_thw = inputs.video_grid_thw if "video_grid_thw" in inputs else None

        inputs_embeds = self.qwen.get_input_embeddings()(input_ids)

        mean_mask = (input_ids == self.mean_token_id)

        mean_embed  = self.learnable_token_MEAN.unsqueeze(0).unsqueeze(0).expand_as(inputs_embeds)

        inputs_embeds = torch.where(mean_mask.unsqueeze(-1), mean_embed, inputs_embeds)


        self.qwen.model.rope_deltas = None 
        outputs = self.qwen.model(
            inputs_embeds = inputs_embeds,
            attention_mask = attention_mask,
            pixel_values = pixel_values,
            image_grid_thw = image_grid_thw,
            pixel_values_videos = pixel_values_videos,
            video_grid_thw = video_grid_thw,
            output_hidden_states = True,
        )

        last_hidden_state = outputs.hidden_states[-1]  # B,L,D

        img_mean = last_hidden_state[mean_mask].view(last_hidden_state.shape[0], -1)

        q_final, interp_lambda = self.adapter(img_mean) 

        loss_fro, loss_orth = self.adapter.get_regularization_loss(img_mean)
        return q_final, interp_lambda, loss_fro, loss_orth


    def forward_target(self, inputs):
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        pixel_values = inputs.pixel_values if "pixel_values" in inputs else None
        image_grid_thw = inputs.image_grid_thw if "image_grid_thw" in inputs else None

        inputs_embeds = self.qwen.get_input_embeddings()(input_ids)

        img_mask = (input_ids == self.img_token_id)
        img_embed = self.learnable_token_IMG.unsqueeze(0).unsqueeze(0).expand_as(inputs_embeds)
        inputs_embeds = torch.where(img_mask.unsqueeze(-1), img_embed, inputs_embeds)
        
        self.qwen.model.rope_deltas = None 
        outputs = self.qwen.model(
            inputs_embeds = inputs_embeds,
            attention_mask = attention_mask,
            pixel_values = pixel_values,
            image_grid_thw = image_grid_thw,
            output_hidden_states = True,
        )

        last_hidden_state = outputs.hidden_states[-1]
        img_hid = last_hidden_state[img_mask].view(last_hidden_state.shape[0], -1)
        return F.normalize(img_hid, p=2, dim=-1)