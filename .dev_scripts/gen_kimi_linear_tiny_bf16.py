#!/usr/bin/env python3
"""Generate a tiny bf16 Kimi Linear checkpoint for load / smoke tests.

Usage:
  PYTHONPATH=. python .dev_scripts/gen_kimi_linear_tiny_bf16.py [out_dir]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "work_dirs" / "kimi_linear_tiny_bf16"
    out_dir.mkdir(parents=True, exist_ok=True)

    from xtuner.v1.model.moe.kimi_linear import KimiLinearTinyConfig

    cfg = KimiLinearTinyConfig()
    # Prefer CPU meta-free init for tiny model
    model = cfg.build()
    model.to(dtype=torch.bfloat16)
    model.init_weights() if hasattr(model, "init_weights") else None

    # Materialize parameters if still on meta
    for p in model.parameters():
        if p.device.type == "meta":
            raise RuntimeError("Model still on meta device; build without meta context")

    if hasattr(model, "save_hf"):
        model.save_hf(str(out_dir))
    else:
        # Fallback: dump state dict
        torch.save(model.state_dict(), out_dir / "pytorch_model.bin")

    # Standalone kimi_linear config.json for from_hf / AutoConfig consumers
    text_cfg = {
        "model_type": "kimi_linear",
        "architectures": ["KimiLinearForCausalLM"],
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "moe_intermediate_size": cfg.moe_intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.attention.num_attention_heads,
        "num_key_value_heads": cfg.attention.num_attention_heads,
        "num_experts": cfg.n_routed_experts,
        "num_experts_per_token": cfg.num_experts_per_tok,
        "num_shared_experts": cfg.n_shared_experts,
        "first_k_dense_replace": cfg.first_k_dense_replace,
        "hidden_act": cfg.hidden_act,
        "activation_situ_beta": cfg.activation_situ_beta,
        "activation_situ_linear_beta": cfg.activation_situ_linear_beta,
        "routed_expert_hidden_size": cfg.routed_expert_hidden_size,
        "latent_moe_use_norm": cfg.latent_moe_use_norm,
        "attn_res_block_size": cfg.attn_res_block_size,
        "rms_norm_eps": cfg.rms_norm_eps,
        "max_position_embeddings": cfg.max_position_embeddings,
        "pad_token_id": cfg.pad_token_id,
        "eos_token_id": cfg.eos_token_id,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "q_lora_rank": cfg.attention.q_lora_rank,
        "kv_lora_rank": cfg.attention.kv_lora_rank,
        "qk_nope_head_dim": cfg.attention.qk_nope_head_dim,
        "qk_rope_head_dim": cfg.attention.qk_rope_head_dim,
        "v_head_dim": cfg.attention.v_head_dim,
        "mla_use_nope": True,
        "mla_use_output_gate": cfg.attention.use_output_gate,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": True,
        "routed_scaling_factor": 1.0,
        "use_grouped_topk": False,
        "num_expert_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "linear_attn_config": {
            "kda_layers": cfg.kda_layers,
            "full_attn_layers": cfg.full_attn_layers,
            "num_heads": cfg.linear_attention.num_heads if cfg.linear_attention else 4,
            "head_dim": cfg.linear_attention.head_dim if cfg.linear_attention else 32,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        "auto_map": {
            "AutoConfig": "configuration_kimi_k3.KimiLinearConfig",
            "AutoModelForCausalLM": "modeling_kimi_linear.KimiLinearForCausalLM",
        },
    }
    (out_dir / "config.json").write_text(json.dumps(text_cfg, indent=2), encoding="utf-8")

    # Copy remote-code stubs so trust_remote_code loaders can resolve model_type
    src = repo / ".cache" / "kimi_k3_hf"
    for name in (
        "configuration_kimi_k3.py",
        "modeling_kimi_linear.py",
        "tokenization_kimi.py",
        "tokenizer_config.json",
        "tiktoken.model",
    ):
        p = src / name
        if p.exists():
            shutil.copy2(p, out_dir / name)

    print(f"Wrote tiny bf16 Kimi Linear checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
