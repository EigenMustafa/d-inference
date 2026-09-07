# Paged KV quantization

> Last updated: 2026-09-07 · commit `0b46b1618` + working tree

Reference for the optional packed full-attention KV formats in the Swift provider.
Native storage remains the default for every model; model weight precision does
not select a KV format. The [design record](../design/paged-kv-quantization-strategy.md)
contains the research rationale.

## Provider selection

| Setting | Default | Contract and source |
|---|---|---|
| `[backend] engine_v2_kv_quantization` | `"native"` | `native`, `int4`, `k8v4`, or `int8`; decoded in `provider-swift/Sources/ProviderCore/Config/ProviderConfig.swift`, `BackendSettings` |
| `[backend] engine_v2_kv_quantization_by_model` | Empty table | Exact model ID overrides the global choice; parsing uses `EngineV2KVQuantizationPolicy.parseSelection` in `provider-swift/Sources/ProviderCore/Inference/EngineV2KVQuantizationPolicy.swift` |
| Paged requirement | Required for every packed format | Refuses construction if the resolved backend is not paged, or no supported owning full-attention layer exists; `EngineV2KVQuantizationPolicy.requireResolvedBackend`, `EngineV2Factory+SegmentedBackend.swift` |
| Model geometry | Checked at construction | Supported head dimensions are 64, 128, 256 and 512; group and rotation divisibility are checked by `PagedKVQuantizationConfig.validate` in `libs/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/Paged/PagedKVQuantization.swift` |
| Retired `[backend] kv_quant` | No effect | Still emits the existing retired-setting warning; it does not enable the new storage format (`ProviderConfig.swift`, `RetiredCodingKeys`) |

## Physical format

The public choices use group size 64 and FP32 affine scale and offset metadata
for every token-local K or V group. The ratios below compare only full-attention
K/V payload with BF16 payload; page tails, temporary workspaces, native windows,
recurrent state, weights and allocator overhead are additional costs.

| Choice | K/V code bits | Effective bits per value including metadata | BF16 payload reduction | Source |
|---|---|---|---|---|
| `native` | Native dtype | Native dtype | 1× for BF16; dtype-dependent otherwise | `PagedKVGroupKey.bytesPerToken` |
| `int4` | 4 / 4 | 5 | 3.2× | `EngineV2KVQuantizationSelection.configuration`, `PagedKVQuantizedRowLayout` |
| `k8v4` | 8 / 4 | 7 averaged across K/V | 16/7× | Same |
| `int8` | 8 / 8 | 9 | 16/9× | Same |

| Component | Representation and contract | Source |
|---|---|---|
| Packed row | Codes, FP32 scales, then FP32 offsets; byte offsets and byte strides | `PagedKVQuantizedRowLayout` |
| Segment | All K rows followed by all V rows; each region orders `[page, KV head, token]` | `PagedKVSegments.swift`, `PagedKVStorageLayout.swift` |
| K/Q transform | Fixed signed normalized Walsh-Hadamard after model RoPE; block size is `min(128, headDim)` for public choices | `PagedKVQuantizationConfig.resolvedRotationBlockSize`, `PagedQuantizedMetal.swift` |
| V transform | No rotation | `PagedQuantizedTransfers.swift` |
| Attention compute | Packed values unpack inside attention; native input/output dtype with FP32 accumulation | `PagedQuantizedMetal.swift`, `PagedQuantizedPrefill.swift` |
| Sliding windows and recurrent state | Existing native storage | `PagedKVGroupKey` in `PagedKVGroup.swift`, `KVAdmissionStorageLayout.swift` |
| Versioned identity | Example `affine-v1-k4v4-g64-f32-h128-s1` binds bit widths, grouping, metadata and transform | `PagedKVQuantizationConfig.identity` |

## Capacity and concurrency

| Quantity | Contract | Source |
|---|---|---|
| `kv_bytes_per_token` | Literal physical growing KV rate, including packed metadata; native compute dtype is separate | `EngineV2KVSizing.swift`, `KVAdmissionStorageLayout.swift` |
| Request charge | Growing storage plus page/window/fixed overhead and a checked temporary-workspace bound | `AdmissionV2.swift`, `QuantizedWorkspaceProjection.swift` |
| Workspace retirement | Request credit cannot be reused while submitted GPU work still owns its workspace; retirement follows real stream completion | `AdmissionWorkspaceFloor.swift`, `EngineLoopV2+QuantizedScratch.swift`, `EngineLoopV2.finalize` |
| Overlap bound | Engine prefill may overlap a pure decode successor; MTP verification calls are priced together and do not chain another step | `WorkspaceOverlapPolicy.swift`, `EngineLoopV2.swift` |
| Wire token budgets | Remain raw prompt-plus-generation tokens; physical overhead reduces the maximum rather than inflating used tokens into byte-equivalent units | `EngineV2Bridge+Capacity.swift`, `EngineV2RoutingCapacity.swift`, `coordinator/protocol/messages.go` |
| In-flight coordinator requests | Reservations not yet present in a heartbeat debit both routing admission and public slot capacity | `coordinator/registry/slot_token_budget.go`, `freeMemoryAdmits`, `ModelCapacitySnapshot` |
| Concurrent request limit | Configured compute limit remains a ceiling; physical admission may advertise fewer serviceable requests | `EngineV2RoutingCapacity.project`, `EngineV2Bridge+Capacity.swift` |
| Cold model admission | Existing padded weight estimates and activation floors remain authoritative; packed warm-cache gains are not guessed for unloaded models | `EngineV2Factory+Production.swift`, `coordinator/registry/servability.go` |

## Prefix and checkpoint compatibility

| Surface | Contract | Source |
|---|---|---|
| Resident paged prefix | Reuses the exact packed pages under the original storage charge | `Paged/PrefixBlocks/EngineLoopV2+PagedPrefix.swift` |
| Complete checkpoint K/V | UInt8 tensors retain packed codes and FP32 metadata exactly, without dequantization or requantization | `Prefix/CheckpointStorageTensorLayout.swift`, `PagedCheckpointTensorSource.swift`, `PagedCheckpointStorage.swift` |
| Manifest | Quantized layouts and `kvQuantization` must match before allocation; native manifests omit the optional field and retain their existing layout | `CompleteCheckpointContract.swift`, `CompleteCheckpointManifestCoding.swift`, `CompleteCheckpointCodec.swift` |
| Persistent identity | Storage format is included in the provider checkpoint identity; incompatible native/packed checkpoints cannot alias | `provider-swift/Sources/ProviderCore/Inference/CompleteCheckpointStorageIdentity.swift` |
| Legacy native snapshot | Disabled for packed storage; complete packed checkpoints and resident page reuse retain their own paths | `EngineV2.swift`, `PagedKVBackend.makeSequenceState` |

## Benchmark surfaces

| Surface | Contract | Source |
|---|---|---|
| `--kv-quantization` | `native`, `int4`, `k8v4`, or `int8`; reports the actual resolved identity | `BenchmarkCommand+KVQuantization.swift`, `BenchmarkKVBackend.swift` |
| `--model-directory` | Exact local artifact override for teacher-forced, quality, sweep, scheduler-prefill and arrival-invariance modes; model hashes bracket measurement | `BenchmarkCommand+ModelDirectory.swift`, `BenchmarkArtifactIdentity.swift` |
| `--teacher-forced-input` | Fixed token context/continuation, repeated ordinary-forward controls, MTP and prefix cache off | `TeacherForcedBenchmark.swift`; [input reference](../provider/cli-reference.md#teacher-forced-scores) |
| `--kv-quality-input` | Bounded natural greedy generation, explicit token prompts, actual EOS IDs, raw tokens/text and supplemental production-parser serving content; absent expected text remains ungraded | `KVQualityInput.swift`, `KVQualityBenchmark.swift`, `KVQualityCollection.swift`, `BenchmarkServingContent.swift` |
| `--scheduler-prefill` | Actual production scheduler/backend prefill measurement | `SchedulerPrefillBenchmark.swift` |
| `--sweep` | Decode uses the selected production format; native-model prefill microbenchmarks are explicitly omitted for packed selections | `ThroughputSweep+PrefillExecution.swift` |

No benchmark status alone establishes broad model quality or a release decision.
Memory reduction, request capacity, prefill latency and decode throughput are
separate measurements.
