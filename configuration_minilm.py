"""
configuration_minilm — MiniLM 的 HuggingFace 标准配置类。

与 `minilm/model.py:ModelConfig` 一一对应，但字段名换成 HF 通用命名
（hidden_size / num_hidden_layers / num_attention_heads / intermediate_size ...），
这样 `config.json` 与 Llama / Qwen / GPT-2 等开源模型的结构完全同构，
`AutoConfig.from_pretrained(dir, trust_remote_code=True)` 可直接读取。
"""
from __future__ import annotations

from transformers import PretrainedConfig


class MiniLMConfig(PretrainedConfig):
    model_type = "minilm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32768,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        max_position_embeddings: int = 1024,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.use_cache = use_cache
        self.head_dim = hidden_size // num_attention_heads

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
