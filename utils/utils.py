import torch
import dataclasses
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig

@dataclasses.dataclass
class ModelConfig:
  model_path: str = dataclasses.field(default=None)  # For backward compatibility
  target_model_path: str = dataclasses.field(default=None)
  draft_model_path: str = dataclasses.field(default=None)
  dtype: str = dataclasses.field(default="float16")
  device: str = dataclasses.field(default="cuda:0")
  use_flash_attention: bool = dataclasses.field(default=True)


def _get_attn_implementation(use_flash_attention: bool):
    """Get attention implementation, with graceful fallback if flash-attn not available."""
    if not use_flash_attention:
        return None
    
    try:
        # Try importing flash-attn to check if it's available
        import flash_attn  # noqa: F401
        # Check if we're on CUDA (flash-attn requires CUDA)
        if torch.cuda.is_available():
            return "flash_attention_2"
        else:
            print("Warning: CUDA not available, flash-attention requires CUDA")
            return None
    except (ImportError, ModuleNotFoundError):
        print("Warning: flash-attn not available, falling back to default attention")
        print("  Install with: pip install flash-attn --no-build-isolation")
        return None


def load_model_and_tokenizer(model_cfg: ModelConfig):
    """Load model and tokenizer from pretrained model path.
    Returns: (target_model, draft_model, tokenizer)
    - target_model: full-precision model
    - draft_model: quantized model (4-bit weights)
    """
    device = torch.device(model_cfg.device)
    
    # Determine dtype
    if model_cfg.dtype == "float16":
        dtype = torch.float16
    elif model_cfg.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    
    # Get attention implementation
    attn_implementation = _get_attn_implementation(model_cfg.use_flash_attention)
    if attn_implementation:
        print(f"Using {attn_implementation} for faster attention computation")
    
    # Prepare model kwargs
    model_kwargs = {
        "device_map": model_cfg.device,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    
    # Determine model paths (support both old and new API)
    target_path = model_cfg.target_model_path or model_cfg.model_path
    draft_path = model_cfg.draft_model_path or model_cfg.model_path
    
    # Load full-precision target model
    target_model = LlamaForCausalLM.from_pretrained(
        target_path,
        torch_dtype=dtype,
        **model_kwargs
    )
    
    # Load draft model (FP16, not quantized for classic speculative decoding)
    draft_kwargs = {**model_kwargs}
    try:
        draft_model = LlamaForCausalLM.from_pretrained(
            draft_path,
            torch_dtype=dtype,
            **draft_kwargs
        )
    except Exception as e:
        # If flash attention fails, fall back to default
        if attn_implementation:
            print(f"Warning: Flash attention with draft model failed: {e}")
            print("Falling back to default attention for draft model")
            draft_kwargs.pop("attn_implementation", None)
            draft_model = LlamaForCausalLM.from_pretrained(
                draft_path,
                torch_dtype=dtype,
                **draft_kwargs
            )
        else:
            raise
    
    # Load tokenizer (use target model's tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(target_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return target_model, draft_model, tokenizer
