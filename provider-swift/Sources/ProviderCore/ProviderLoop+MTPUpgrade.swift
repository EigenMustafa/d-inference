import Foundation
import MLX

/// Retains the old target exactly once while an unregistered, minimal-grant
/// replacement owns only new assistant and KV resources.
final class StagedProviderMTPUpgrade: @unchecked Sendable {
    let modelID: String
    let original: ProviderLoop.ModelSlot
    let replacement: ProviderEngineBundle
    let sizing: SlotSizingSnapshot
    let lease: PendingModelLoadLease

    init(modelID: String, original: ProviderLoop.ModelSlot,
         replacement: ProviderEngineBundle, sizing: SlotSizingSnapshot,
         lease: PendingModelLoadLease) {
        self.modelID = modelID
        self.original = original
        self.replacement = replacement
        self.sizing = sizing
        self.lease = lease
    }
}

extension ProviderLoop {
    var mtpStagingBytes: UInt64 {
        mtpStagingReservations.extraBytes(residentTargets: Set(modelSlots.values.map { ObjectIdentifier($0.engineV2) }))
    }

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
        logger.info("mtp: model=\(modelID) \(message)")
    }

    private func pendingMTPUpgradeModels() -> [String] {
        guard !isShuttingDown, !state.refusingNewWork,
            SpecDecArtifactFunnel.killSwitchEnabled(environment: ProcessInfo.processInfo.environment)
        else { return [] }
        return modelSlots.compactMap { modelID, slot in
            guard modelID == "gemma-4-26b-qat-4bit", !slot.engineBundle.mtpStatus.active,
                loopConfig.config.backend.mtpMode.enablesMTP(
                    forModelType: slot.modelType, embeddedArtifactDeclared: false, modelID: modelID),
                !modelsUnloading.contains(modelID), !isRefusedByRetirement(modelID)
            else { return nil }
            return modelID
        }.sorted()
    }

    func prepareMTPUpgrade(_ modelID: String) async throws -> StagedProviderMTPUpgrade? {
        guard pendingMTPUpgradeModels().contains(modelID), !isLoadingAny,
            let original = modelSlots[modelID],
            let info = advertisedModels[modelID],
            let directory = ModelScanner.resolveLocalPath(modelID: modelID)
        else { return nil }
        // Cache misses only schedule the funnel-owned fetch and return. The
        // current engine remains registered and accepts all ordinary traffic.
        let preparation = await specDecPreparation(
            modelId: modelID, modelInfo: info, modelDirectory: directory, logStatus: false)
        guard let artifact = preparation.artifact, !isLoadingAny,
            modelSlots[modelID]?.engineV2 === original.engineV2,
            pendingMTPUpgradeModels().contains(modelID)
        else { return nil }
        isLoadingAny = true
        defer { isLoadingAny = false; releaseLoadGateWaiters() }
        let grant = Int(clamping: EngineV2KVSizing.minimumServiceableGrantBytes)
        guard let lease = await kvBudget.claimPendingLoad(
            requestID: "mtp-upgrade:\(modelID):\(UUID().uuidString)",
            weightBytes: artifact.residentBytes, minimumKVBytes: UInt64(grant))
        else {
            logger.warning("mtp: model=\(modelID) assistant staging deferred: insufficient memory; retaining target engine")
            throw MTPIdleUpgrade.PreparationError.insufficientMemory
        }
        await acquireResliceGate()
        mtpStagingReservations.reserve(lease, target: ObjectIdentifier(original.engineV2),
            targetBytes: UInt64(max(0, original.sizing.weightsBytes)),
            assistantBytes: artifact.residentBytes, kvBytes: UInt64(grant))
        releaseResliceGate()
        let preparationStarted = ContinuousClock.now
        var prepared: EngineV2PreparedModel?
        var replacement: ProviderEngineBundle?
        do {
            try Task.checkCancellation()
            guard await kvBudget.recheckPendingLoad(lease) else { throw CancellationError() }
            let logger = self.logger
            prepared = try await EngineV2SlotFactory.prepareProductionModel(
                modelId: modelID, isVLM: original.isVLM, modelDirectory: directory,
                container: original.container, specDecPreparation: preparation,
                assistantLoader: engineV2SlotHooks?.assistantLoader ?? ProductionProviderMTPAssistantLoader(),
                emitTelemetry: engineV2SlotHooks?.emitTelemetry,
                logInfo: { logger.info($0) }, logWarning: { logger.warning($0) })
            guard let prepared, prepared.mtpStatus.active,
                modelSlots[modelID]?.engineV2 === original.engineV2,
                pendingMTPUpgradeModels().contains(modelID)
            else { throw CancellationError() }
            guard await kvBudget.reducePendingLoad(lease, remainingWeightBytes: 0),
                await kvBudget.recheckPendingLoad(lease)
            else { throw CancellationError() }
            let sizing = original.sizing.replacingAuxiliaryWeightBytes(prepared.assistantBytes)
            replacement = try await makeEngineV2BundleForSlot(
                modelId: modelID, modelType: original.modelType, isVLM: original.isVLM,
                modelDirectory: directory, container: original.container, tokenizer: original.tokenizer,
                sizing: sizing, kvBytesCapacity: grant, specDecPreparation: preparation,
                preparedModel: prepared, cacheEligibleWeightHash: original.cacheEligibleWeightHash,
                registerInRuntime: false)
            try Task.checkCancellation()
            let replacement = replacement!
            let active: Bool
            if engineV2SlotHooks != nil { active = replacement.mtpStatus.active }
            else { active = await replacement.bridge.mtpStatusSnapshot().active }
            guard active, KVHeadroomProbe.postBuildServeable(
                kvBackendKind: replacement.bridge.kvBackendKind,
                pagedPoolBytes: await replacement.bridge.kvBackendPoolBytes(),
                activationReserveBytes: resolvedActivationReserveBytes)
            else { throw CancellationError() }
            logger.info("mtp: model=\(modelID) verified replacement prepared in \(preparationStarted.duration(to: .now)); waiting for natural idle")
            return StagedProviderMTPUpgrade(modelID: modelID, original: original,
                replacement: replacement, sizing: sizing, lease: lease)
        } catch {
            if let replacement { await replacement.bridge.shutdown(); replacement.releaseAssistant() }
            prepared?.assistant?.release()
            MLX.Memory.clearCache()
            mtpStagingReservations.release(lease)
            await kvBudget.finishPendingLoad(lease)
            logger.warning("mtp: model=\(modelID) optional preparation failed: \(error); retaining target engine")
            throw error
        }
    }

    func commitMTPUpgradeIfIdle(_ staged: StagedProviderMTPUpgrade) async throws -> Bool {
        let modelID = staged.modelID
        try Task.checkCancellation()
        guard modelSlots[modelID]?.engineV2 === staged.original.engineV2,
            pendingMTPUpgradeModels().contains(modelID)
        else { throw CancellationError() }
        guard !requestToModel.values.contains(modelID), !hasLocalReservation(modelID), !isLoadingAny else {
            return false
        }
        let capacity = await staged.original.engineV2.capacitySnapshot()
        guard capacity.activeRequests == 0, capacity.waitingRequests == 0,
            capacity.kvBytesReserved == 0 else { return false }
        await acquireResliceGate()
        defer { releaseResliceGate() }
        try Task.checkCancellation()
        guard modelSlots[modelID]?.engineV2 === staged.original.engineV2,
            pendingMTPUpgradeModels().contains(modelID)
        else { throw CancellationError() }
        guard !requestToModel.values.contains(modelID), !hasLocalReservation(modelID), !isLoadingAny else {
            return false
        }
        // No suspension between the last owner check and the admission gate.
        // Work arriving during publication waits; existing work is never drained.
        mtpUpgradeTransitions.insert(modelID)
        defer { finishMTPUpgradeTransition(modelID) }
        await engineV2Runtime.register(modelId: modelID, bridge: staged.replacement.bridge)
        modelSlots[modelID] = ModelSlot(
            engineBundle: staged.replacement, container: staged.original.container,
            tokenizer: staged.original.tokenizer, sizing: staged.sizing,
            cacheEligibleWeightHash: staged.original.cacheEligibleWeightHash,
            isVLM: staged.original.isVLM, modelType: staged.original.modelType,
            lastInferenceAt: staged.original.lastInferenceAt)
        // Publication is committed. Shutdown of the old idle engine releases
        // its pool before the minimal replacement grant is grown.
        await staged.original.engineV2.shutdown()
        staged.original.engineBundle.releaseAssistant()
        MLX.Memory.clearCache()
        mtpStagingReservations.release(staged.lease)
        await kvBudget.finishPendingLoad(staged.lease)
        await resliceGrowSurvivorsLocked()
        syncWarmModelState()
        await updateAggregateCapacity()
        logger.info("mtp: verified assistant installed at idle boundary for \(modelID)")
        return true
    }

    func discardMTPUpgrade(_ staged: StagedProviderMTPUpgrade) async {
        await staged.replacement.bridge.shutdown()
        staged.replacement.releaseAssistant()
        MLX.Memory.clearCache()
        mtpStagingReservations.release(staged.lease)
        await kvBudget.finishPendingLoad(staged.lease)
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
