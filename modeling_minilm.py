"""
modeling_minilm — MiniLM 的 HuggingFace 标准实现。

与 `minilm/model.py` 里的 `GPT` **数值完全等价**，只是换成 HF 的模块划分与权重命名：

    minilm/model.py                            本文件
    --------------------------------------     ------------------------------------------
    tok_emb.weight                         ->  model.embed_tokens.weight
    pos_emb.weight                         ->  model.embed_positions.weight
    blocks[i].ln1.{weight,bias}            ->  model.layers[i].input_layernorm.*
    blocks[i].attn.qkv.weight  (融合 QKV)   ->  model.layers[i].self_attn.{q,k,v}_proj.weight  （切片拆开）
    blocks[i].attn.proj.weight             ->  model.layers[i].self_attn.o_proj.weight
    blocks[i].ln2.{weight,bias}            ->  model.layers[i].post_attention_layernorm.*
    blocks[i].mlp.fc.weight                ->  model.layers[i].mlp.fc1.weight
    blocks[i].mlp.proj.weight              ->  model.layers[i].mlp.fc2.weight
    ln_f.{weight,bias}                     ->  model.norm.*
    lm_head.weight（与 tok_emb 共享）       ->  lm_head.weight（tie_word_embeddings=True，不重复存盘）

架构保持「GPT-2 式」：Pre-LN + GELU(4x) + 可学习位置嵌入 + 权重共享 + Linear 无 bias。
因此除了能用自己的 `MiniLMForCausalLM` 加载，还能一键转成 `model_type=gpt2`
（见 scripts/export_hf.py --layout gpt2），被**原生 transformers** 直接加载。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache

try:                                            # transformers>=4.50 需要显式继承
    from transformers.generation.utils import GenerationMixin
except Exception:                               # 老版本里 PreTrainedModel 已包含
    GenerationMixin = object
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

try:                                            # 本地包内相对导入
    from .configuration_minilm import MiniLMConfig
except ImportError:                             # trust_remote_code / 单文件加载
    from configuration_minilm import MiniLMConfig

ACT2FN = {
    "gelu": F.gelu,
    "gelu_new": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
    "silu": F.silu,
}


class MiniLMAttention(nn.Module):
    """多头因果自注意力（SDPA，单卡/多卡自动走 FlashAttention / mem-efficient）。"""

    def __init__(self, config: MiniLMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim,
                                bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim,
                                bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size,
                                bias=config.attention_bias)
        self.o_proj._residual_scaled = True      # 残差投影：初始化时缩小 std

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,   # (B, KV) 1/0
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        kv_len = k.shape[2]
        past_len = kv_len - q_len

        attn_mask = None
        need_causal = q_len > 1
        need_pad = attention_mask is not None and bool((attention_mask == 0).any())
        if need_causal or need_pad or past_len > 0:
            attn_mask = torch.zeros(q_len, kv_len, dtype=q.dtype, device=q.device)
            if need_causal:
                # 位置 i（绝对位置 past_len+i）只能看到 <= 它的 key
                future = torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device).triu(
                    diagonal=past_len + 1)
                attn_mask = attn_mask.masked_fill(future, torch.finfo(q.dtype).min)
            if need_pad:
                pad = (1.0 - attention_mask.to(q.dtype)).unsqueeze(1).unsqueeze(1)  # (B,1,1,KV)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0) + pad * torch.finfo(q.dtype).min
            else:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            scale=self.scaling,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(out)


class MiniLMMLP(nn.Module):
    def __init__(self, config: MiniLMConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.fc2._residual_scaled = True         # 残差投影：初始化时缩小 std
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))


class MiniLMDecoderLayer(nn.Module):
    def __init__(self, config: MiniLMConfig, layer_idx: int):
        super().__init__()
        self.self_attn = MiniLMAttention(config, layer_idx)
        self.mlp = MiniLMMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, attention_mask, past_key_values, use_cache)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class MiniLMPreTrainedModel(PreTrainedModel):
    config_class = MiniLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _supports_sdpa = True
    _supports_attention_backend = True
    _no_split_modules = ["MiniLMDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            # 残差输出投影（attn.o_proj / mlp.fc2）按 GPT-2 惯例缩放：
            # std = 0.02 / sqrt(2 * n_layers)，与自研 GPT 的初始化逐位一致，
            # 否则残差分支初值过大，训练早期 loss 明显偏高。
            if getattr(module, "_residual_scaled", False):
                std = std / math.sqrt(2 * self.config.num_hidden_layers)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


class MiniLMModel(MiniLMPreTrainedModel):
    """Transformer 主体：embedding + N×decoder layer + final norm。"""

    def __init__(self, config: MiniLMConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=None)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [MiniLMDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_hidden_states = bool(output_hidden_states)

        bsz, seq_len = input_ids.shape
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = past_key_values.get_seq_length()
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + seq_len,
                                        dtype=torch.long, device=input_ids.device).unsqueeze(0)

        hidden_states = self.dropout(
            self.embed_tokens(input_ids) + self.embed_positions(position_ids))

        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = layer(hidden_states, attention_mask, past_key_values, use_cache)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )


class MiniLMForCausalLM(MiniLMPreTrainedModel, GenerationMixin):
    """因果语言模型：`model` + 输出头，与 embed_tokens 共享权重。"""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniLMConfig):
        super().__init__(config)
        self.model = MiniLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = outputs.last_hidden_state
        # 训练/评估要全量 logits；推理只要最后几个位置（省显存）
        slice_len = hidden_states.shape[1] if logits_to_keep == 0 else logits_to_keep
        logits = self.lm_head(hidden_states[:, -slice_len:, :]).float()

        loss = None
        if labels is not None:
            # HF 语义：logits[t] 预测 labels[t+1]（内部右移一位），与预训练/SFT 口径一致
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = past_key_values.get_seq_length()
        if past_len > 0:
            input_ids = input_ids[:, past_len:]

        position_ids = None
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids[:, -input_ids.shape[1]:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return past_key_values.reorder_cache(beam_idx) if hasattr(past_key_values, "reorder_cache") else past_key_values
