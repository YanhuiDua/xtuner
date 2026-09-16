# Kimi-K3 text tower (`kimi_linear`) integration design

> Status: **phase-1 + AttnRes forward landed**; 8-GPU smoke script available. Numerical parity still deferred.  
> Scope: text-only, random bf16 tiny checkpoint, load + AttnRes forward, no numerical parity, no multimodal.

## 1. Reference and classification

| Property | Value |
|---|---|
| Hub | [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) |
| Top-level HF `model_type` | `kimi_k3` (VLM compose) — **out of scope this phase** |
| Text tower HF `model_type` | `kimi_linear` |
| Text architecture | `KimiLinearForCausalLM` |
| Remote code (cached under `.cache/kimi_k3_hf/`) | `configuration_kimi_k3.py`, `modeling_kimi_linear.py`, tokenizer |
| Bucket | **MoE LLM**, hybrid attention, `trust_remote_code` |
| Published scale | 2.8T total / ~104B act, 93 layers, 896 experts (top-16), shared 2 |
| Hub weights | **MXFP4** (`compressed-tensors`) — not training-suitable |
| This phase weights | **Synthetic bf16** (see §4) — not the Hub MXFP4 dump |

Full published text dims (from Hub `config.json` → `text_config`):

| Field | Value |
|---|---|
| `hidden_size` | 7168 |
| `num_hidden_layers` | 93 |
| `first_k_dense_replace` | 1 |
| Attention mix | 69 KDA + 24 Gated MLA (Hub lists; XTuner schedule is the matching formula below) |
| MLA | `q_lora_rank=1536`, `kv_lora_rank=512`, `qk_nope=128`, `qk_rope=64`, `v=128`, `mla_use_nope=True`, **`mla_use_output_gate=True`** |
| MoE | 896 experts, top-16, shared 2, `topk_method=noaux_tc`, sigmoid, **LatentMoE** (`routed_expert_hidden_size=3584`, `latent_moe_use_norm=True`) |
| Act | **`situ`** (`activation_situ_beta=4`, `activation_situ_linear_beta=25`) |
| AttnRes | `attn_res_block_size=12` |
| Vocab / ctx | 163840 / 1048576 |

## 2. Reuse map vs gaps

Closest existing ports: hybrid MoE text in `moe/qwen3_5_text.py` (GatedDeltaNet + gated MHA) and `moe/deepseek_v3.py` (MLA + NoAux router).

| HF piece | Closest XTuner | Gap for Kimi |
|---|---|---|
| Hybrid layer dispatch (`kda` vs MLA) | `MoE.build_layers` + `layers_type` / `linear_attention` | **Landed**: formula `full` iff 1-based `i % 4 == 0` or last layer (same as Hub 69/24); not driven by HF lists |
| MLA | `MLAConfig` / `MultiLatentAttention` | **Landed**: `use_output_gate` / `use_nope` |
| KDA (`KimiDeltaAttention`) | `GatedDeltaNet` (Qwen3.5 style) | **Landed** as separate `kda.py` (not GDN); Ulysses SP `forward_for_sp` present |
| Router `noaux_tc` + sigmoid | `NoAuxRouterConfig` | Likely reusable; verify group/topk fields |
| Shared experts | MoE shared expert path | OK |
| LatentMoE down/up (+norm) | `MoEDecoderLayer` | **Landed**: ctor `routed_expert_hidden_size` + `_project_to/from_experts` |
| SiTU-GLU | `ops/act_fn.py` | **Landed**: `situ_glu` / `get_mlp_act_fn`; fused `native_situ` for experts |
| AttnRes (`attn_res_block_size`) | Implemented | Cross-layer `block_residual` via `_call_decoder_layer` / `_seed_block_residual`; output mixer in `_post_decoder_stack` |
| HF MoE training | XTuner MoE training path | Hub HF code **raises** in training MoE; XTuner must own train path |

**Implication:** this is not a thin Config+`to_hf_key_list` port. Several module/ops pieces are required even for a correct *structure* load.

## 3. Proposed code layout (three seams)

### 3.1 Model / config — `xtuner/v1/model/moe/kimi_linear/`

- `KimiLinear(MoE)` — override `to_hf_key_list` (and layout hooks if LatentMoE / KDA renames need them). HF sparse block is named `block_sparse_moe` (not always under `mlp.`); mapping must match Hub text-tower keys when saving a **standalone** `kimi_linear` dir. AttnRes output modules via `_configure_model_specific_layers`; stack hooks `_call_decoder_layer` / `_post_decoder_stack`.
- `KimiLinearConfig(MoEConfig)` — `from_hf` via `AutoConfig(..., trust_remote_code=True)` asserting `model_type == "kimi_linear"` **or** accepting a `kimi_k3` dir and reading `text_config` only. `hf_config` builds a serializable `PretrainedConfig` for `save_hf` (remote-code `auto_map`). Hybrid `layers_type` uses the formula in §2 (not HF `kda_layers` lists).
- Size configs:
  - `KimiLinearTinyConfig` — **phase-1 smoke** (see §4).
  - `KimiLinearK3TextConfig` — published dims (structure registration; full random weight dump optional / deferred — too large).
- Decoder: `decoder_layer.py` — `KimiDenseDecoderLayer` / `KimiMoEDecoderLayer` (AttnRes only; MLP/experts via base `build_*` + LatentMoE / SiTU ctor args).

Register in `xtuner/v1/model/__init__.py`:

- alias e.g. `"kimi-linear-tiny"`, `"kimi-k3-text"`;
- `get_model_config_from_hf`: `kimi_linear` → `KimiLinearConfig.from_hf`; optionally `kimi_k3` → extract text tower only (document clearly).

### 3.2 Module layer (minimal for load)

| New / extend | Location | Notes |
|---|---|---|
| `KimiDeltaAttention` / `KDAConfig` | `xtuner/v1/module/attention/kda.py` | Separate from Qwen3.5 `GatedDeltaNet`; FLA `causal_conv1d` + `chunk_kda`; SP via `forward_for_sp`. |
| MLA `output_gate` | `MLAConfig` + `MultiLatentAttention` | Flag + `g_proj`; default off for DeepSeek. |
| SiTU | `xtuner/v1/ops/act_fn.py` | `situ_glu` for dense/shared MLP; fused `native_situ` for routed experts. |
| LatentMoE | `MoEDecoderLayer` | `routed_expert_{down,up}_proj` + optional RMSNorm; `build_experts` at latent dim. |
| AttnRes | `kimi_linear/decoder_layer.py` + `MoE._post_decoder_stack` | Carry `block_residual` (incl. micro-batch); seed `(T, 0, H)`; output mixer before final norm. |
| Shared FLA gated RMSNorm | `ops/gated_deltanet/fused_rms_norm_gated.py` | Shared by GDN and KDA (DTensor `to_local`). |

Ops: reuse fla / causal_conv1d paths where KDA calls the same kernels; add `XTUNER_HF_IMPL` branches only when numerical parity starts (deferred).

### 3.3 Chat template

Out of scope for multimodal. For text smoke load, tokenizer copy from Hub remote files is enough; **full `CHAT_TEMPLATE_MAP` entry can wait** until SFT is requested (user: chat/多模态不做).

## 4. Synthetic bf16 checkpoint plan

Hub MXFP4 is not used. Per user: **random bf16 weights, saved locally**.

**Do not** materialize a full 2.8T bf16 dump for phase 1 (multi-TB, impractical on 8 GPUs even with FSDP for a first smoke).

| Artifact | Purpose |
|---|---|
| `KimiLinearTinyConfig` | e.g. 4 layers (mix of KDA + MLA), hidden 512–1024, few experts (8, top-2), shared 1, small LatentMoE dim, `attn_res_block_size=2`, small vocab |
| Script | e.g. `.dev_scripts/gen_kimi_linear_tiny_bf16.py` — build XTuner (or HF tiny with same remote code) → `save_hf` / safetensors → local dir under e.g. `work_dirs/kimi_linear_tiny_bf16/` |
| Optional | Same script can emit a **config-only** full-size `KimiLinearK3TextConfig` JSON for registration tests without tensors |

Phase-1 success = `from_hf(tiny_dir)` loads under XTuner, FSDP/8-GPU init path does not crash on a short forward (tolerance / bitwise **not** required).

## 5. Phase-1 acceptance (this PR series)

Must:

1. Design confirmed (this doc).
2. Tiny bf16 checkpoint generated and saved.
3. `KimiLinear` + config registered; `get_model_config_from_hf` works on tiny dir.
4. Smoke: single-rank or 8-GPU `Trainer`/engine **load + 1 step forward** (loss need not match HF).
5. Basic `save_hf` / reload of tiny checkpoint (tensor presence; byte-equal optional later).

Explicitly deferred:

- Bitwise / numerical parity (`XTUNER_HF_IMPL`, decoder-layer matrix in skill §6–7).
- Full-size random or Hub MXFP4 load.
- Vision / projector / compose `kimi_k3`.
- Chat template + SFT/RL recipes.
- EP/fp8/compile feature matrix (skill §8); KDA SP is implemented but not covered by dedicated tests.
- Behavioral unit coverage beyond structure + FSDP smoke (AttnRes mixer on/off, LatentMoE dims, SiTU vs SiLU, KDA SP).

### 8-GPU smoke

```bash
bash examples/v1/scripts/run_kimi_linear_tiny_8gpu.sh
# or
torchrun --standalone --nproc-per-node=8 \
  examples/v1/scripts/smoke_kimi_linear_tiny.py \
  --ckpt work_dirs/kimi_linear_tiny_bf16 --seq-len 64
```

---

Phase-1 §6 decisions were confirmed; AttnRes full forward (incl. micro-batch hooks) and 8-GPU smoke follow-up are implemented.
