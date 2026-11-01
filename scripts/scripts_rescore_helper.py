"""Judge loader with a selectable attention implementation (shared by the diagnostic scripts)."""
from __future__ import annotations


def load_judge_with(jcfg, attn: str = "sdpa"):
    """attn: 'sdpa' (PyTorch picks the kernel), 'sdpa-math' (SDPA with the fused flash and
    memory-efficient kernels disabled, i.e. the reference math kernel), or 'eager'."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if attn == "sdpa-math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        attn = "sdpa"
    dtype = getattr(torch, jcfg["dtype"])
    base = AutoModelForCausalLM.from_pretrained(jcfg["base_model"], revision=jcfg.get("base_model_revision"),
                                                dtype=dtype, attn_implementation=attn)
    model = PeftModel.from_pretrained(base, jcfg["adapter"], revision=jcfg.get("adapter_revision")).merge_and_unload().to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(jcfg["adapter"], revision=jcfg.get("adapter_revision"), padding_side="left", truncation_side="left")
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    return model, tok
