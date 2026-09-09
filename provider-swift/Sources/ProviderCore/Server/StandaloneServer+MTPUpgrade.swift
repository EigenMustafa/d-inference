import Foundation
import MLX

/// Retains the old target exactly once while an unregistered, minimal-grant
/// replacement owns only new assistant and KV resources.
final class StagedStandaloneMTPUpgrade: @unchecked Sendable {
    let modelID: String
    let original: StandaloneServer.CachedSlot
    let replacement: ProviderEngineBundle
    let sizing: SlotSizingSnapshot
    let lease: PendingModelLoadLease

    init(modelID: String, original: StandaloneServer.CachedSlot,
         replacement: ProviderEngineBundle, sizing: SlotSizingSnapshot,
         lease: PendingModelLoadLease) {
        self.modelID = modelID
        self.original = original
        self.replacement = replacement
        self.sizing = sizing
        self.lease = lease
    }
}

extension StandaloneServer {
    func startMTPUpgradeMonitor() {
        guard mtpUpgradeMonitorTask == nil else { return }
        mtpUpgradeMonitorTask = Task { [weak self] in
            var nextAttempt: [String: ContinuousClock.Instant] = [:]
            var lastOutcome: [String: MTPIdleUpgrade.Outcome] = [:]
            while !Task.isCancelled {
                guard let self else { return }
                let candidates = await self.pendingMTPUpgradeModels()
                nextAttempt = nextAttempt.filter { candidates.contains($0.key) }
                for modelID in candidates where !Task.isCancelled {
                    if let next = nextAttempt[modelID], ContinuousClock.now < next { continue }
                    if lastOutcome[modelID] == nil {
                        await self.logMTPUpgrade("checking/downloading verified assistant; target remains available", modelID: modelID)
                    }
                    let outcome = await MTPIdleUpgrade.run(
                        prepare: { try await self.prepareMTPUpgrade(modelID) },
                        commitIfIdle: { try await self.commitMTPUpgradeIfIdle($0) },
                        discard: { await self.discardMTPUpgrade($0) })
                    if outcome != lastOutcome[modelID], outcome != .installed {
                        await self.logMTPUpgrade("upgrade outcome=\(outcome); retaining current engine", modelID: modelID)
                    }
                    lastOutcome[modelID] = outcome
                    // A single monitor deduplicates GPU preparation. Fetches
                    // are independently bounded/deduplicated by the funnel.
                    nextAttempt[modelID] = .now.advanced(by:
                        outcome == .notReady ? .seconds(15) : .seconds(300))
                }
                do { try await Task.sleep(for: .seconds(Int.random(in: 10...15))) }
                catch { return }
            }
        }
    }

    private func logMTPUpgrade(_ message: String, modelID: String) {
        standaloneLogger.info("mtp: model=\(modelID) \(message)")
    }

    private func pendingMTPUpgradeModels() -> [String] {
        guard lifecycleState == .running,
            SpecDecArtifactFunnel.killSwitchEnabled(environment: ProcessInfo.processInfo.environment)
        else { return [] }
        return slots.compactMap { modelID, slot in
            guard modelID == "gemma-4-26b-qat-4bit", !slot.bundle.mtpStatus.active,
                config.mtpMode.enablesMTP(
                    forModelType: slot.modelType, embeddedArtifactDeclared: false, modelID: modelID),
                !evictingModels.contains(modelID), models.contains(where: { $0.id == modelID })
            else { return nil }
            return modelID
        }.sorted()
    }

    func prepareMTPUpgrade(_ modelID: String) async throws -> StagedStandaloneMTPUpgrade? {
        guard pendingMTPUpgradeModels().contains(modelID), !isLoadingAny,
            let original = slots[modelID],
            let info = models.first(where: { $0.id == modelID }),
            let directory = ModelScanner.resolveLocalPath(modelID: modelID)
        else { return nil }
        // Cache misses only schedule the funnel-owned fetch and return. The
        // current engine remains registered and accepts all ordinary traffic.
        let preparation = await specDecPreparation(
            modelId: modelID, modelInfo: info, modelDirectory: directory)
        guard let artifact = preparation.artifact, !isLoadingAny,
            slots[modelID]?.bridge === original.bridge,
            pendingMTPUpgradeModels().contains(modelID)
        else { return nil }
        isLoadingAny = true
        let grant = Int(clamping: EngineV2KVSizing.minimumServiceableGrantBytes)
        guard let lease = await kvBudget.claimPendingLoad(
            requestID: "mtp-upgrade:\(modelID):\(UUID().uuidString)",
            weightBytes: artifact.residentBytes, minimumKVBytes: UInt64(grant))
        else {
            standaloneLogger.warning("mtp: model=\(modelID) assistant staging deferred: insufficient memory; retaining target engine")
            await finishMTPUpgradeLoad()
            throw MTPIdleUpgrade.PreparationError.insufficientMemory
        }
        let preparationStarted = ContinuousClock.now
        var prepared: EngineV2PreparedModel?
        var replacement: ProviderEngineBundle?
        do {
            try Task.checkCancellation()
            guard await kvBudget.recheckPendingLoad(lease) else { throw CancellationError() }
            prepared = try await EngineV2SlotFactory.prepareProductionModel(
                modelId: modelID, isVLM: original.isVLM, modelDirectory: directory,
                container: original.container, specDecPreparation: preparation,
                assistantLoader: v2TestHooks?.assistantLoader ?? ProductionProviderMTPAssistantLoader(),
                emitTelemetry: v2TestHooks?.emitTelemetry,
                logInfo: { standaloneLogger.info("\($0)") }, logWarning: { standaloneLogger.warning("\($0)") })
            guard let prepared, prepared.mtpStatus.active,
                slots[modelID]?.bridge === original.bridge,
                pendingMTPUpgradeModels().contains(modelID)
            else { throw CancellationError() }
            guard await kvBudget.reducePendingLoad(lease, remainingWeightBytes: 0),
                await kvBudget.recheckPendingLoad(lease)
            else { throw CancellationError() }
            let sizing = original.sizing.replacingAuxiliaryWeightBytes(prepared.assistantBytes)
            v2TestHooks?.onCacheEligibleWeightHash?(original.cacheEligibleWeightHash)
            replacement = try await EngineV2SlotFactory.makeProductionBundle(
                modelId: modelID, modelType: original.modelType, isVLM: original.isVLM,
                modelDirectory: directory, container: original.container, tokenizer: original.tokenizer,
                sizing: sizing, kvBytesCapacity: grant,
                maxConcurrentRequests: engineV2MaxConcurrent(forModel: modelID), kvBudget: kvBudget,
                activationReserveBytes: resolvedActivationReserveBytes,
                kvBackendConfig: config.engineV2KVBackend,
                kvBackendConfigByModel: config.engineV2KVBackendByModel,
                prefillDeadlineMode: config.prefillDeadlineMode,
                weightHash: original.cacheEligibleWeightHash,
                specDecPreparation: preparation, preparedModel: prepared,
                emitTelemetry: v2TestHooks?.emitTelemetry,
                makeEngineOverride: v2TestHooks?.makeEngine,
                logInfo: { standaloneLogger.info("\($0)") },
                logWarning: { standaloneLogger.warning("\($0)") })
            try Task.checkCancellation()
            let replacement = replacement!
            let active: Bool
            if v2TestHooks != nil { active = replacement.mtpStatus.active }
            else { active = await replacement.bridge.mtpStatusSnapshot().active }
            guard active, KVHeadroomProbe.postBuildServeable(
                kvBackendKind: replacement.bridge.kvBackendKind,
                pagedPoolBytes: await replacement.bridge.kvBackendPoolBytes(),
                activationReserveBytes: resolvedActivationReserveBytes)
            else { throw CancellationError() }
            standaloneLogger.info("mtp: model=\(modelID) verified replacement prepared in \(String(describing: preparationStarted.duration(to: .now))); waiting for natural idle")
            await finishMTPUpgradeLoad()
            return StagedStandaloneMTPUpgrade(modelID: modelID, original: original,
                replacement: replacement, sizing: sizing, lease: lease)
        } catch {
            if let replacement { await replacement.bridge.shutdown(); replacement.releaseAssistant() }
            prepared?.assistant?.release()
            MLX.Memory.clearCache()
            await kvBudget.finishPendingLoad(lease)
            standaloneLogger.warning("mtp: model=\(modelID) optional preparation failed: \(String(describing: error)); retaining target engine")
            await finishMTPUpgradeLoad()
            throw error
        }
    }

    func commitMTPUpgradeIfIdle(_ staged: StagedStandaloneMTPUpgrade) async throws -> Bool {
        let modelID = staged.modelID
        try Task.checkCancellation()
        guard slots[modelID]?.bridge === staged.original.bridge,
            pendingMTPUpgradeModels().contains(modelID)
        else { throw CancellationError() }
        guard slotReservations[modelID, default: 0] == 0, !isLoadingAny else { return false }
        let capacity = await staged.original.bridge.capacitySnapshot()
        try Task.checkCancellation()
        guard capacity.activeRequests == 0, capacity.waitingRequests == 0,
            capacity.kvBytesReserved == 0 else { return false }
        guard slots[modelID]?.bridge === staged.original.bridge,
            pendingMTPUpgradeModels().contains(modelID)
        else { throw CancellationError() }
        guard slotReservations[modelID, default: 0] == 0, !isLoadingAny else { return false }
        isLoadingAny = true
        // No suspension between the last owner check and the admission gate.
        // Work arriving during publication waits; existing work is never drained.
        mtpUpgradeTransitions.insert(modelID)
        defer { finishMTPUpgradeTransition(modelID) }
        slots[modelID] = CachedSlot(
            bundle: staged.replacement, container: staged.original.container,
            tokenizer: staged.original.tokenizer, modelType: staged.original.modelType,
            isVLM: staged.original.isVLM, sizing: staged.sizing,
            lastUsedAt: staged.original.lastUsedAt,
            cacheEligibleWeightHash: staged.original.cacheEligibleWeightHash)
        // Publication is committed. Shutdown of the old idle engine releases
        // its pool before the minimal replacement grant is grown.
        await staged.original.bridge.shutdown()
        staged.original.bundle.releaseAssistant()
        MLX.Memory.clearCache()
        await kvBudget.finishPendingLoad(staged.lease)
        await resliceGrowSurvivors()
        await finishMTPUpgradeLoad()
        standaloneLogger.info("mtp: verified assistant installed at idle boundary for \(modelID)")
        return true
    }

    func discardMTPUpgrade(_ staged: StagedStandaloneMTPUpgrade) async {
        await staged.replacement.bridge.shutdown()
        staged.replacement.releaseAssistant()
        MLX.Memory.clearCache()
        await kvBudget.finishPendingLoad(staged.lease)
    }

    private func finishMTPUpgradeLoad() async {
        isLoadingAny = false
        releaseLoadGateWaiters()
        await applyDeferredModelsIfNeeded()
    }

    func waitForMTPUpgrade(_ modelID: String) async {
        while mtpUpgradeTransitions.contains(modelID) {
            await withCheckedContinuation { mtpUpgradeWaiters[modelID, default: []].append($0) }
        }
    }

    private func finishMTPUpgradeTransition(_ modelID: String) {
        mtpUpgradeTransitions.remove(modelID)
        let waiters = mtpUpgradeWaiters.removeValue(forKey: modelID) ?? []
        for waiter in waiters { waiter.resume() }
    }
}
