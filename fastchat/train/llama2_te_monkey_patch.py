import warnings
from typing import Optional, Tuple

import torch
from torch import nn
from flash_attn import __version__ as flash_attn_version
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_kvpacked_func,
)
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaModel,
    LlamaDecoderLayer,
    LlamaMLP,
    rotate_half,
)
from transformers.models.llama.configuration_llama import LlamaConfig

import transformer_engine.pytorch as te

# monkey patch for LlamaAttention
def LlamaAttention__init__(self, config: LlamaConfig):
    super(LlamaAttention, self).__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta

    if (self.head_dim * self.num_heads) != self.hidden_size:
        raise ValueError(
            f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
            f" and `num_heads`: {self.num_heads})."
        )
    self.q_proj = te.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
    self.k_proj = te.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
    self.v_proj = te.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
    self.o_proj = te.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
    # self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
    # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
    # self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
    # self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
    self._init_rope()



# monkey patch for LlamaMLP
def LlamaMLP__init__(self, config):
    super(LlamaMLP, self).__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.gate_proj = te.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.up_proj = te.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.down_proj = te.Linear(self.intermediate_size, self.hidden_size, bias=False)
    # self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    # self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    # self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    self.act_fn = ACT2FN[config.hidden_act]

# Disable the transformation of the attention mask in LlamaModel as flash attention
# takes a boolean key_padding_mask. Fills in the past kv length for use in forward.
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat(
            (
                torch.full(
                    (input_shape[0], past_key_values_length),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ),
            dim=-1,
        )

    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples

    return attention_mask

def replace_llama_with_te():
    # LlamaDecoderLayer.__init__ = __init__
    # LlamaDecoderLayer.forward = forward
    LlamaAttention.__init__ = LlamaAttention__init__
    LlamaMLP.__init__ = LlamaMLP__init__

def test():
    print("in test of te")
    # from fastchat.train.llama_flash_attn_monkey_patch import forward as fastchat_forward
    from transformers.models.llama.configuration_llama import LlamaConfig

    config = LlamaConfig(
        hidden_size=1024,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=8,
        max_position_embeddings=16,
    )
    device = torch.device("cuda")
    model = LlamaModel(config)
    attn = LlamaAttention(config).to(device).half()
    # decoder = LlamaDecoderLayer(config).to(device)
    decoder = LlamaMLP(config).to(device)

    # LlamaDecerLayer.__init__ = __init__
    # LlamaMLP.__init__ = __init__
    # test_decoder = LlamaDecoderLayer(config).to(device)
    # test_decoder = LlamaDecoderLayer(config).to(device)


    bsz, hs, seqlen = 2, config.hidden_size, config.max_position_embeddings
    position_ids = torch.arange(seqlen, dtype=torch.long, device=device).view(
        -1, seqlen
    )
    print("after")

    mask = torch.full((bsz, seqlen), True, dtype=torch.bool, device=device)
    for i in range(4):
        hidden = torch.rand((bsz, seqlen, hs), dtype=torch.float32, device=device)
        if i:
            mask[0, -i:] = False
            mask[1, :i] = False

        lmask = model._prepare_decoder_attention_mask(mask, hidden.shape[:2], hidden, 0)

        # ref = decoder.forward(hidden, attention_mask=lmask, position_ids=position_ids)

        # decoder.layernorm_mlp = te.LayerNormMLP(
            # config.hidden_size,
            # config.intermediate_size,
            # eps=config.rms_norm_eps,
            # bias=False,
            # normalization='RMSNorm',
            # activation='swiglu',
            # init_method=lambda x: decoder.mlp.down_proj.weight,
            # output_layer_init_method=lambda x: decoder.mlp.down_proj.weight,
            # )
        ref = decoder.forward(hidden)
        # ref = decoder.forward(hidden, position_ids=position_ids)
        # ref = ref[0]
        # with torch.no_grad():

        w = decoder.gate_proj.weight.data
        decoder.gate_proj = te.Linear(
                config.hidden_size,
                config.intermediate_size,
                bias=False,
                )
        decoder.gate_proj.weight.data = w

        w = decoder.up_proj.weight.data
        decoder.up_proj = te.Linear(
                config.hidden_size,
                config.intermediate_size,
                bias=False,
                )
        decoder.up_proj.weight.data = w

        w = decoder.down_proj.weight.data
        decoder.down_proj = te.Linear(
                config.intermediate_size,
                config.hidden_size,
                bias=False,
                )
        decoder.down_proj.weight.data = w

        # ref, _, _ = attn.forward(
            # hidden, attention_mask=lmask, position_ids=position_ids
        # )

        # fast, _, _ = fastchat_forward(
            # attn, hidden, attention_mask=mask, position_ids=position_ids
        # )

        lmask = _prepare_decoder_attention_mask(
            model, mask, hidden.shape[:2], hidden, 0
        )
        # test = forward(test_decoder, hidden, attention_mask=mask,
                # position_ids=position_ids)
        # test = forward(test_decoder, hidden,
                # position_ids=position_ids)
        test = decoder.forward(hidden)

        # test = test[0]
        # print(test)
        # print(ref)
        # test, _, _ = forward(
            # attn, hidden, attention_mask=lmask, position_ids=position_ids
        # )

        print(f"Mean(abs(ref)) = {torch.mean(torch.abs(ref))}")
        # print(f"Mean(abs(ref - fast)) = {torch.mean(torch.abs(ref - fast))}")
        print(f"Mean(abs(ref - test)) = {torch.mean(torch.abs(ref - test))}")
        # print(f"Mean(abs(fast - test)) = {torch.mean(torch.abs(fast - test))}")
        # print(f"allclose(fast, test) = {torch.allclose(fast, test)}")
        print(f"allclose(ref, test) = {torch.allclose(ref, test)}")

if __name__ == '__main__':
    test()
