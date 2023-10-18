# Replace layers with Nvidia's TransformerEngine to make it more performant and
# enable fp8 training


from fastchat.train.llama2_te_monkey_patch import (
    replace_llama_with_te,
)

replace_llama_with_te()
replace_llama_attn_with_flash_attn()

from fastchat.train.train import train

if __name__ == "__main__":
    train()
