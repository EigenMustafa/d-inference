# Coordinator and native admission validation

2026-09-07. Experimental uncommitted worktree only; no production changes.

Latest scoped-workspace source: **126 unique native tests pass**:

- `kv-workspace-step-arena-attempt1.log`: 33 tests in six suites cover packed kernels, shared step arenas, exact storage admission, workspace ownership, projection/envelope proofs, and engine transitions.
- `kv-workspace-step-arena-broader-attempt1.log`: 53 XCTest tests and 29 Swift Testing tests cover first-token/deadline work projections, scheduler admission, native/packed complete checkpoints, native dtype controls, checkpoint transfer and process memory ownership.
- `kv-workspace-step-arena-physical-isolated-attempt1.log`: 11 physical-admission tests pass alone, including the process-global memory comparison.

`workspace-step-arena-components.json` records the final scoped decomposition, source hashes and base commits. Each component total is checked against the native diagnostic log. At a four-request engine cap, conservative packed full-KV plus workspace charges for one 32K/128K request are 0.569/1.305 GiB for Qwen3.6, 0.678/1.544 GiB for GPT-OSS and 0.586/1.388 GiB for Gemma4. Corresponding native full-KV storage is 0.625/2.5, 1.5/6 and 0.625/2.5 GiB. This comparison excludes weights, native windows, recurrent state, page tails, detached owners and other slots; all remain separately charged. It is admission arithmetic, not measured peak memory, concurrency, throughput or model quality.

Earlier full native regression baseline: **114 tests pass** before the broadcast/workspace optimization:

- `kv-quant-admission-integrated-attempt2.log`: 53 XCTest tests and 50 Swift Testing tests pass. Covers packed kernels, native dtype controls, complete native/packed checkpoints, exact storage rates, workspace ownership, deadline projections and process memory ownership.
- `kv-quant-paged-admission-isolated.log`: 11 physical-admission tests pass in their required isolated process. The source's native-owner test explicitly requires its suite to run alone because it compares process-global `Memory.activeMemory`.
- `kv-quant-admission-integrated-attempt1.log`: retained failed combined invocation. Its one failure was that process-global memory comparison while other suites allocated concurrently. The expectation was not changed; the full physical suite passes alone.

Earlier focused logs remain alongside these final records. The intentionally fragmented 256-segment test required a fixture grant increase from 16 MiB to 64 MiB: the existing physical allocator accounting refused its seeded backing at 25,589,368 bytes. This changes only the test fixture. The final test still asserts at least 256 physical segments, performs a real 128-token packed prefill, verifies its actual reserved scratch against the prospective bound, and checks the output.

The first workspace optimization subsequently passes **30 tests in five suites** (`kv-workspace-broadcast-integrated-attempt2.log`): packed storage/attention, shared topology arenas and strided inputs, exact admission, workspace ownership, aggregate bounds, oversized prefill and actual prefill-to-chained-decode transitions. The first build attempt is retained; it stopped on two missing inner `try` expressions in new test macros, before runtime tests. It did not expose a production-code failure. These focused results do not replace the earlier broader regression scope.

`workspace-broadcast-components.json` decomposes this tested projection by native output, partial/meta arrays, GPU and host metadata, and other owners. It records source hashes and base commits. These are analytic admission bounds matching native diagnostic output, not measured model heap usage. The captured allocator policy gives `B(4)=7`, `B(96)=191`, `B(136)=271` and maximum additional allocation bytes 49,150; 16 KiB is not a minimum allocation. The retained earlier diagnostic shows the pre-optimization cost for comparison. Real-model query dtype, quality and performance remain separate evidence.

Coordinator verification executed independently:

```text
go test ./registry ./protocol
go test -race ./registry -run 'TestQuantizedKV|TestModelCapacityQuantized' -count=1
```

Both pass. The original reproduction of `TestModelCapacityQuantizedPrivateGrantCountsPending` failed with `Ready:true` and 4096 remaining tokens after reserving the entire private grant. The shared `remainingSlotTokenBudget` helper fixes the public capacity feed while preserving reservation semantics.

## Capacity contract and call map

| Concern | Existing authority / path |
|---|---|
| Warm format and budget | Heartbeat canonicalization and capacity sequence ordering publish `BackendCapacity.Slots`; `KVBytesPerToken` remains the resolved storage rate, including quantization metadata. |
| Request admission | `snapshotProviderIntoPLockedEx` + `fillSnapshotPendingAndPool` -> `buildCandidateWithReason` -> `freeMemoryAdmits`; per-slot pending and pooled byte limits both bind. |
| Concurrent commit | Primary reservation and `ReserveNextFromPlan` rebuild the candidate under the provider lock before inserting the pending request. |
| Queue wake-up | Heartbeat calls the canonical queue drain; suppressed saturated passes get a trailing drain after the 20 ms window. Queue assignment calls `ReserveProviderEx`. |
| Compute limits | Provider-reported per-model concurrency, the solo-TPS quality cap, provider-wide safety cap and TTFT gates remain independent of memory savings. |
| Cold models | Native conservative estimates and padded weight/free-for-load checks remain unchanged. Quantization on another resident model does not establish a cold model's rate. |
| Pending loads | Load planner and cold-spill selection reject another pending load. Reservations are serialized; disconnect, timeout and load-status paths remove pending-load state. |
| Account limits | Request/token-rate and balance admission remain unchanged. |

Raw-token de-duplication requires `ActiveTokenBudgetUsed` and `MaxTokensPotential` to stay in raw committed-token units. Current fixed/window/workspace/physical-floor overhead instead reduces the reported maximum. Encoding byte-equivalent overhead as extra used tokens would cancel unrelated fresh coordinator reservations.

## Native ownership

`KVAdmissionStorageLayout` resolves packed K/V bytes including metadata separately from compute dtype. Shared borrowers own no KV bytes; windows retain native rates. Admission's estimates, page-rounded allocations, projected operations, rollback and full-KV rate consume the same immutable table.

`QuantizedWorkspaceProjection` separately prepays supported prefill, decode and serial-MTP workspaces. Generic callers retain two arbitrary waves. EngineV2 explicitly opts into its proved overlap contract: maximum of prefill plus decode, two decodes, and one multi-column MTP round. Only pure decode can be a chained successor; an MTP round cannot be a chain base. The chunk ceiling includes pool, ordinary prefill, solo stripe and batch-step limits; the decode ceiling covers the entire rectangular maximum batch so a long row cannot borrow insufficient credit from short peers. Borrowing attention layers still contribute compute workspace.

Engine builders declare a complete step's maximum query width and context frontier before graph construction. One fixed partial/meta arena is allocated per unique query-head/head-dimension/partition geometry and cannot grow. A real fence joins each caller's KV dependency with the arena's previous merge before reuse across layers or MTP columns. The scope detaches into that step's leases; a chained successor creates separate arenas. If F/D are summed per-call nonarena prefill/decode costs and A is one arena per geometry, the engine prepays `max(F+D+2A, 2D+2A, serialCalls*D+A)`. Generic unscoped forwards retain separate per-call arena pricing. Native outputs remain conservatively priced at up to FP32 width; the activation reserve is not lowered or substituted.

The provider's aggregate getter bounds the sum of prospective allowances for every split of a raw token total into the available request slots. A nonnegative cost envelope separately sums partitions, capped queries, query blocks, and partition-block products. Exhaustive small splits and real-geometry allocation comparisons test the proof; concentrated, balanced and long-plus-short production-size distributions test the resulting capacity ceiling and binary-search monotonicity. Existing and retired commitments remain separately charged. No marginal storage rate is redefined as scratch bytes.

Actual workspace leases consume that credit once. `AdmissionWorkspaceFloor` retains retired credit while GPU leases remain outstanding so a new request cannot reuse it prematurely. Request release, unreserve, detached generations, unscheduled teacher rows and checkpoint transfer/rollback retain the same ownership contract. Native scratch cannot consume target-KV physical slack.

The workspace bound and heartbeat envelope are conservative. They do not establish model-quality acceptance or a throughput gain, and they do not justify increasing compute caps or changing the default cache policy.
