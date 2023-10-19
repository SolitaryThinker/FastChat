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
    LlamaFlashAttention2,
    LlamaModel,
    rotate_half,
)
from transformers.models.llama.configuration_llama import LlamaConfig


# monkey patch for LlamaDecoderLayer
def __init__(self, config: LlamaConfig):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.self_attn = (
        LlamaAttention(config=config)
        if not getattr(config, "_flash_attn_2_enabled", False)
        else LlamaFlashAttention2(config=config)
    )
    # self.mlp = LlamaMLP(config)
    self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    # self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.layernorm_mlp = te.LayerNormMLP(
            config.hidden_size,
            eps=config.rms_norm_eps,
            bias=False,
            normalization='RMSNorm',
            activation='swiglu')

# monkey patch for LlamaDecoderLayer
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        padding_mask=padding_mask,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    # hidden_states = self.post_attention_layernorm(hidden_states)
    # hidden_states = self.mlp(hidden_states)

    # FIXME looking in to is_first_microbatch
    hidden_states = self.layernorm_mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs

def replace_llama_with_te():
    LlamaDecoderLayer.__init__ = __init__
    LlamaDecoderLayer.forward = forward

def test():
    print("in test of te")
    # from fastchat.train.llama_flash_attn_monkey_patch import forward as fastchat_forward
    # from transformers.models.llama.configuration_llama import LlamaConfig

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
    decoder = LlamaDecoderLayer(config).to(device).half()
    bsz, hs, seqlen = 2, config.hidden_size, config.max_position_embeddings
    position_ids = torch.arange(seqlen, dtype=torch.long, device=device).view(
        -1, seqlen
    )

    mask = torch.full((bsz, seqlen), True, dtype=torch.bool, device=device)
    for i in range(4):
        hidden = torch.rand((bsz, seqlen, hs), dtype=torch.float16, device=device)
        if i:
            mask[0, -i:] = False
            mask[1, :i] = False

        lmask = model._prepare_decoder_attention_mask(mask, hidden.shape[:2], hidden, 0)

        ref, _, _ = decoder.forward(decoder, hidden, attention_mask=lmask,
                position_ids=position_ids)

        # ref, _, _ = attn.forward(
            # hidden, attention_mask=lmask, position_ids=position_ids
        # )

        # fast, _, _ = fastchat_forward(
            # attn, hidden, attention_mask=mask, position_ids=position_ids
        # )

        lmask = _prepare_decoder_attention_mask(
            model, mask, hidden.shape[:2], hidden, 0
        )
        test, _, _ = forward(decoder, hidden, attention_mask=lmask,
                position_ids=position_ids)
        # test, _, _ = forward(
            # attn, hidden, attention_mask=lmask, position_ids=position_ids
        )

        print(f"Mean(abs(ref)) = {torch.mean(torch.abs(ref))}")
        # print(f"Mean(abs(ref - fast)) = {torch.mean(torch.abs(ref - fast))}")
        print(f"Mean(abs(ref - test)) = {torch.mean(torch.abs(ref - test))}")
        # print(f"Mean(abs(fast - test)) = {torch.mean(torch.abs(fast - test))}")
        # print(f"allclose(fast, test) = {torch.allclose(fast, test)}")
        print(f"allclose(ref, test) = {torch.allclose(ref, test)}")

if __name__ == '__main__':
    test()
