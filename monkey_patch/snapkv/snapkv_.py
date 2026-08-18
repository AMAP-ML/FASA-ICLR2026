from importlib.metadata import version
import warnings
import transformers
from monkey_patch.oracle.llama_oracle_static import LLAMA_ATTENTION_CLASSES_flash
from monkey_patch.oracle.qwen_oracle_static import QWEN2_ATTENTION_CLASSES_flash 
from monkey_patch.oracle.mistral_oracle_static import MISTRAL_ATTENTION_CLASSES_flash
from monkey_patch.snapkv.llama_snapkv_4_43 import LlamaFlashAttention2_forward
from monkey_patch.snapkv.qwen_snapkv_4_43 import Qwen2FlashAttention2_forward
from monkey_patch.snapkv.mistral_snapkv_4_43 import MistralFlashAttention2_forward

def replace_llama():
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = LlamaFlashAttention2_forward
    transformers.models.llama.modeling_llama.LLAMA_ATTENTION_CLASSES = LLAMA_ATTENTION_CLASSES_flash


def replace_qwen():
    transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = Qwen2FlashAttention2_forward
    transformers.models.qwen2.modeling_qwen2.QWEN2_ATTENTION_CLASSES = QWEN2_ATTENTION_CLASSES_flash

def replace_mistral():
    transformers.models.mistral.modeling_mistral. MistralFlashAttention2.forward = MistralFlashAttention2_forward
    transformers.models.mistral.modeling_mistral.MISTRAL_ATTENTION_CLASSES = MISTRAL_ATTENTION_CLASSES_flash
