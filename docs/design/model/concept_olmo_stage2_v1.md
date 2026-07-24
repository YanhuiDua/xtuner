# Concept OLMo Stage-2 V1 integration design

## 1. Reference checkpoint and classification

The integration target is the local HuggingFace remote-code checkpoint at
`/mnt/shared-storage-user/llmrazor-share/chenchiyu/ncp/hf_model_remote/concept_olmo_stage_2_V1`.
The checkpoint itself is not copied into the repository.

| Property | Value |
|---|---|
| HuggingFace `model_type` | `conceptlm_v22_vq` |
| HuggingFace architecture | `ConceptLMV22VQForCausalLM` |
| Bucket | dense, text-only, `trust_remote_code` |
| Parameters | 8,938,363,793 |
| Checkpoint tensors | 722 BF16 tensors and two empty U8 `_extra_state` tensors |
| Vocabulary / hidden size | 100,278 / 4,096 |
| Maximum sequence length | 8,192 |
| Reference environment | `pt29`: PyTorch 2.9.1, Transformers 5.2.0 |

The weights are training-suitable BF16 weights. A read-only reference forward
on one H200 with input IDs `[1, 2, 3, 4, 5, 6, 7, 8]` produced loss
`6.564164161682129`; the last-token logits began with
`[4.90625, -6.21875, 2.796875, -1.8515625, -2.78125, 0.185546875,
-3.796875, -6.03125]`. Peak allocated memory was 16.69 GiB. HuggingFace reports
the two `_extra_state` tensors as unexpected, but XTuner will preserve them for
a byte-complete `save_hf` round trip.

This is not a conventional OLMo decoder and cannot be represented by the
current `Dense` body alone. Its forward graph is:

1. token embedding;
2. 16-layer token encoder;
3. per-document token-to-concept reduction (chunk size 4, mean pooling);
4. 8-layer concept predictor and 32-way residual VQ projection;
5. token/concept fusion;
6. 16-layer token decoder with encoder/concept cross-depth routes;
7. final RMSNorm and untied LM head.

Each transformer layer uses fused, head-interleaved QKV, full-hidden-vector Q/K
RMSNorm, RoPE (`theta=500000`), a fused SwiGLU input projection, and OLMo-style
post-attention/post-MLP normalization. Three out of every four layers use a
left window of 4096 tokens; every fourth layer uses global causal attention.
There is no KV cache: generation recomputes the complete prefix.

## 2. Code decomposition

The port stays in XTuner's model/module/ops seams.

### 2.1 Model and config

Add `xtuner/v1/model/dense/concept_olmo.py` with:

- `ConceptLMV22VQ(BaseModel)`: orchestration, packed-sequence concept
  transforms, LM-head/loss integration, checkpoint key mapping, FSDP stage
  wrapping, and the top-level forward;
- `ConceptLMV22VQConfig(TransformerConfig)`: remote-code `from_hf`, active
  ConceptLM fields, supported-variant validation, and `hf_config = None`;
- `ConceptOlmoStage2V1Config`: the concrete published checkpoint dimensions and
  feature switches.

The direct `BaseModel` subclass is intentional. Subclassing `Dense` would
require replacing its constructor, layer builder, forward, and sharding logic,
while retaining none of its decoder-only structure.

`from_hf` uses `AutoConfig.from_pretrained(..., trust_remote_code=True)`, asserts
`model_type == "conceptlm_v22_vq"`, and maps all fields consumed by the XTuner
implementation. The remote model uses the legacy `rotary_base` name; the shared
`RopeParametersConfig.from_hf_config` helper will gain compatibility for that
name so RoPE is still parsed through the standard helper. Because there is no
built-in Transformers config, `hf_config` remains `None`; `save_hf` therefore
copies the remote config, tokenizer, and Python side files from `_hf_path`.

### 2.2 Reusable modules

Add:

- `xtuner/v1/module/attention/concept_olmo.py`: fused QKV projection with
  `[head, Q/K/V, dim]` storage, full-hidden Q/K RMSNorm, existing XTuner RoPE,
  `get_attn_impl_fn`, packed causal masks, and the existing Ulysses all-to-all
  path for SP;
- `xtuner/v1/module/decoder_layer/concept_olmo_decoder_layer.py`: fused SwiGLU
  MLP plus the exact OLMo post-norm residual order;
- `xtuner/v1/module/concept_olmo.py`: Depth-DD, residual routes, two-route add,
  quantizer, and encoder/concept/decoder stage units.

All trainable matrix operations use `build_linear`. With FP8 disabled it is the
same `nn.Linear` numerical path as the remote reference; with FP8 enabled the
same modules participate in XTuner's Float8 handler instead of growing a second
implementation.

Stage units group one transformer layer with the routing modules executed next
to it. This provides a correct activation-checkpoint/FSDP boundary and avoids
leaving the cross-depth routing graph outside recomputation.

### 2.3 Ops compatibility

The remote reference and FlashAttention treat a left window of 4096 as an
inclusive distance. XTuner's eager/flex mask currently uses a strict `< 4096`
test. Align eager/flex to FlashAttention's inclusive boundary at the ops layer
and add a boundary regression test. This is required for bitwise HF parity at
sequences longer than the window and keeps the correction shared by all model
implementations.

## 3. Checkpoint layout

Stage-oriented XTuner names map one-to-one to the existing checkpoint; tensor
bytes do not need reshaping or transposition.

| XTuner region | HuggingFace region |
|---|---|
| `embed_tokens.*` | `embedding.word_embeddings.*` |
| `encoder_stages.N.transformer.*` | `encoder.layers.N.*` |
| `encoder_stages.N.depth_dd.*` | `dd_encoder_self_dd.depth_dds.N.*` |
| `concept_stages.N.transformer.*` | `concept_predictor.hlm_block.layers.N.*` |
| `concept_stages.N.depth_dd.*` | `concept_predictor.concept_self_dd.depth_dds.N.*` |
| `concept_stages.N.read_encoder_route.*` | `concept_predictor.concept_read_encoder_routes.N.*` |
| `decoder_stages.N.transformer.*` | `decoder.layers.N.*` |
| `decoder_stages.N.decoder_dd.*` | `dd_two_route_add.decoder_dds.N.*` |
| `decoder_stages.N.concept_route.*` | `dd_two_route_add.concept_routes.N.*` |
| `decoder_stages.N.read_encoder_route.*` | `decoder_read_encoder_routes.N.*` |
| `decoder_stages.N.read_concept_route.*` | `decoder_read_concept_routes.N.*` |

The remaining embedding, codebook, norms, prediction heads, fusion parameters,
gate logits, and LM head have direct one-to-one mappings. The two empty U8
`_extra_state` entries are registered as persistent buffers on the two final
RMSNorm modules so the saved index and tensors remain byte-equal to the source.

## 4. Packed sequences and sequence parallelism

The remote code ignores `attention_mask` and only demonstrates a rectangular HF
batch. XTuner training packs documents into one token row, so concept pooling,
shifting, and concept-to-token expansion must use `SequenceContext.cu_seq_lens`
and reset at every document boundary. The port will derive a second concept
`SequenceContext` whose cumulative lengths and positions are computed per
document. No pooling, route, or shift may cross a packed-document boundary.

Baseline packed correctness is tested by comparing a packed forward with the
same documents forwarded separately.

For SP, transformer attention reuses XTuner's Ulysses path. The token-to-concept
boundary performs an SP all-gather only for the encoder states being reduced,
builds the per-document concept sequence, then repartitions it with its derived
concept context. Concept outputs required by the token decoder are gathered and
expanded back to the local token offsets. This favors a simple correct baseline
over a model-specific fused collective; SP `{1, 2, 4, 8}` is compared against
the single-rank BF16 baseline.

## 5. FSDP, compilation, FP8, and offload

- FSDP wraps each of the 16 encoder, 8 concept, and 16 decoder stage units
  independently, then separately wraps embedding, output head, shared norms,
  codebook/prediction heads, and the root.
- Activation recomputation uses those same 40 stage boundaries. The explicit
  cross-depth state tensors remain architectural outputs, while attention/MLP
  intermediates are recomputed.
- `torch.compile` is applied per stage after the eager baseline passes. Any
  graph breaks are localized to packed-context construction outside stage
  kernels.
- FP8 covers the large fused QKV, fused SwiGLU, output projections, prediction
  heads, and route linears through `build_linear`; norms, gates, and codebook
  parameters stay at their required precision.
- `XTUNER_ACTIVATION_OFFLOAD=1` wraps each stage with a separate
  encoder/concept/decoder offload group, including the cross-depth state tensors
  that dominate this architecture's retained activation memory.
- Expert parallelism and EP-only intra-layer micro-batching are not applicable:
  this checkpoint has no experts or router. Sequence parallelism is the dense
  parallel feature exercised instead.

Each switch is tested against the committed eager BF16 baseline, and the maximum
absolute/relative error is recorded.

## 6. Registration and user entry points

`xtuner/v1/model/__init__.py` will:

1. import and export `ConceptLMV22VQConfig` and
   `ConceptOlmoStage2V1Config`;
2. register alias `concept-olmo-stage2-v1`;
3. dispatch `model_type == "conceptlm_v22_vq"` to
   `ConceptLMV22VQConfig.from_hf`.

Add `ci/config/concept_olmo_stage2_v1.py`. Both checkpoint and tokenizer paths
come from `CONCEPT_OLMO_STAGE2_V1_PATH`; the dataset comes from `ALPACA_PATH`.
No path is hard-coded in tests or runnable configs.

## 7. Validation matrix

Baseline tests in `tests/model/test_concept_olmo_dense.py` cover:

1. config parsing, alias, and `model_type` dispatch;
2. meta construction and selective HF loading;
3. bitwise forward/backward parity for one global and one sliding transformer
   layer (`output`, CE loss, and `dL/dx`);
4. route, chunk, VQ, and fusion helper parity against the remote implementation;
5. whole-model 8.94B forward/logit parity (medium-model rule: forward only);
6. packed-document isolation;
7. FSDP forward parity;
8. full `save_hf` byte round trip, including the two U8 entries and remote-code
   side files;
9. SP, compiled/eager, FP8/BF16, and activation-offload comparisons.

The drop-in CI config is run for approximately 50 real optimization steps. The
loss trajectory and every optimization's residual error are recorded in the
handoff/PR notes.

## 8. Commit plan

1. design document;
2. shared RoPE/window compatibility fixes and their unit tests;
3. model/module/config/registration baseline;
4. baseline parity, packed-sequence, FSDP/save tests, and training config;
5. SP support and comparison tests;
6. compile support and comparison tests;
7. FP8 support and comparison tests;
8. activation-offload support and comparison tests.

The baseline is completed and committed sequentially. Independent optimization
work starts only after the baseline parity suite is green.
