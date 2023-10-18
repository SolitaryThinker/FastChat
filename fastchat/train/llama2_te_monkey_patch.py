import warnings
from typing import Optional, Tuple

import torch
from flash_attn import __version__ as flash_attn_version
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_kvpacked_func,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaModel,
    rotate_half,
)

def __init__(self, config: LlamaConfig):
    super().__init__(config)
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size

    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
    self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
    self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    self.gradient_checkpointing = False
    # Initialize weights and apply final processing
    self.post_init()
