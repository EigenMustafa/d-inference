import Foundation
import MLXLMCommon
import Testing
@testable import ProviderCore
import ProviderCoreFoundation

private let upgradeModelID = "gemma-4-26b-qat-4bit"

private final class UpgradeScriptedEngine: CBv2Engine, @unchecked Sendable {
    private let lock = NSLock()
    private var bytes: Int
    private var busy = false
    private var stops = 0
    private let shutdownBarrier: UpgradeBarrier?
    init(bytes: Int, shutdownBarrier: UpgradeBarrier? = nil) {
        self.bytes = bytes
        self.shutdownBarrier = shutdownBarrier
    }
    var shutdownCount: Int { lock.withLock { stops } }
    func setBusy(_ value: Bool) { lock.withLock { busy = value } }
    func submit(_ request: CBv2Request) throws -> AsyncStream<CBv2Event> { AsyncStream { $0.finish() } }
    func cancel(_ id: CBv2RequestID) {}
    func capacity() -> CBv2CapacitySnapshot {
        lock.withLock { .init(activeRequests: busy ? 1 : 0, waitingRequests: 0,
            kvBytesInUse: 0, kvBytesCapacity: bytes, activeTokens: 0, stepsExecuted: 0) }
    }
    func updateKVBytesCapacity(_ bytes: Int) { lock.withLock { self.bytes = bytes } }
    func shutdown() async {
        lock.withLock { stops += 1 }
        await shutdownBarrier?.wait()
    }
}

private final class UpgradeScriptedFactory: @unchecked Sendable {
    enum Failure: Error { case injected }
    private let lock = NSLock()
    private var fail = false
    private var built: [UpgradeScriptedEngine] = []
    func failBuild() { lock.withLock { fail = true } }
    var latest: UpgradeScriptedEngine? { lock.withLock { built.last } }
    func make(_ bytes: Int) throws -> UpgradeScriptedEngine {
        try lock.withLock {
            if fail { throw Failure.injected }
            let engine = UpgradeScriptedEngine(bytes: bytes)
            built.append(engine)
            return engine
        }
    }
}

private struct ProviderUpgradeFixture {
    let loop: ProviderLoop
    let runtime: EngineV2Runtime
    let original: EngineV2Bridge
    let originalEngine: UpgradeScriptedEngine
    let factory: UpgradeScriptedFactory
    let artifact: SpecDecArtifact
    let createdScannerPath: URL?

    static func make(shutdownBarrier: UpgradeBarrier? = nil) async throws -> Self {
        let artifact = try mtpFloorArtifact()
        // Never overwrite a user's existing checkpoint. If absent, provide
        // only an isolated scanner entry; slot hooks supply all model work.
        var createdScannerPath: URL?
        if ModelScanner.resolveLocalPath(modelID: upgradeModelID) == nil {
            let hub = try #require(ModelScanner.defaultCacheDirectory())
            let model = hub.appendingPathComponent("models--\(upgradeModelID)")
            let ownedModel = !FileManager.default.fileExists(atPath: model.path)
            let snapshot = model.appendingPathComponent("snapshots/mtp-upgrade-test-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
            try Data("{}".utf8).write(to: snapshot.appendingPathComponent("config.json"))
            createdScannerPath = ownedModel ? model : snapshot
        }
        let loop = try mtpFloorLoop(models: [ModelInfo(id: upgradeModelID,
            modelType: "gemma4", sizeBytes: 1, estimatedMemoryGb: 1)],
            mtpDrafterPath: artifact.directory.path)
        let runtime = EngineV2Runtime()
        let factory = UpgradeScriptedFactory()
        await loop.setEngineV2RuntimeForTesting(runtime)
        await loop.setEngineV2SlotHooksForTesting(.init(physicalMemoryBytes: 64 << 30,
            assistantLoader: MTPFloorAssistantLoader(),
            makeEngine: { _, bytes in try factory.make(bytes) }))
        let engine = UpgradeScriptedEngine(bytes: 1 << 30, shutdownBarrier: shutdownBarrier)
        let bridge = EngineV2Bridge(engine: engine, modelId: upgradeModelID,
            tokenizer: TokenizerHandle(MTPFloorTokenizer()), eosTokenIds: [])
        await runtime.register(modelId: upgradeModelID, bridge: bridge)
        await loop.installModelSlotForTesting(modelId: upgradeModelID,
            container: mtpFloorContainer(), tokenizer: TokenizerHandle(MTPFloorTokenizer()),
            engineV2: bridge, sizing: mtpFloorSizing(weightsGiB: 1), modelType: "gemma4")
        return Self(loop: loop, runtime: runtime, original: bridge, originalEngine: engine,
            factory: factory, artifact: artifact, createdScannerPath: createdScannerPath)
    }

    func checkOriginal() async {
        #expect(await loop.slotBridgeForTesting(modelId: upgradeModelID) === original)
        #expect(await runtime.bridge(forModel: upgradeModelID) === original)
        #expect(originalEngine.shutdownCount == 0)
    }
    func clean() async {
        await loop.beginShutdownForTesting()
        if let bridge = await runtime.unregister(modelId: upgradeModelID) { await bridge.shutdown() }
        await loop.removeModelSlotForTesting(modelId: upgradeModelID)
        cleanFiles()
    }
    func cleanFiles() {
        try? FileManager.default.removeItem(at: artifact.directory)
        if let createdScannerPath { try? FileManager.default.removeItem(at: createdScannerPath) }
    }
}

private extension ProviderLoop {
    func setUpgradeCoordinatorPin(_ pinned: Bool) {
        requestToModel["upgrade-test-request"] = pinned ? upgradeModelID : nil
    }
    func upgradeResliceWaiterCount() -> Int { resliceGateWaiters.count }
    func upgradeAdmissionWaiterCount() -> Int { mtpUpgradeWaiters[upgradeModelID]?.count ?? 0 }
}

@Suite("ProviderLoop assistant upgrade integration", .serialized)
struct ProviderLoopMTPUpgradeTests {
    init() { _ = LiveInferenceFixtures.ensureMetallibColocated() }

    @Test("coordinator pins, local acquisition and engine capacity block the real publication")
    func realReservationsKeepOldEngineUntilIdle() async throws {
        let fixture = try await ProviderUpgradeFixture.make()
        defer { fixture.cleanFiles() }
        let staged = try #require(try await fixture.loop.prepareMTPUpgrade(upgradeModelID))
        #expect(await fixture.loop.outstandingKVReservationBytesForTesting() > 0)
        await fixture.checkOriginal()
        await fixture.loop.setUpgradeCoordinatorPin(true)
        #expect(try await !fixture.loop.commitMTPUpgradeIfIdle(staged))
        await fixture.loop.setUpgradeCoordinatorPin(false)
        let acquired = try await fixture.loop.acquireModelForLocal(upgradeModelID)
        #expect(acquired.engineV2Bridge === fixture.original)
        #expect(try await !fixture.loop.commitMTPUpgradeIfIdle(staged))
        await acquired.releaseToken.fire()
        fixture.originalEngine.setBusy(true)
        #expect(try await !fixture.loop.commitMTPUpgradeIfIdle(staged))
        fixture.originalEngine.setBusy(false)
        await fixture.checkOriginal()
        #expect(try await fixture.loop.commitMTPUpgradeIfIdle(staged))
        #expect(await fixture.loop.slotBridgeForTesting(modelId: upgradeModelID) === staged.replacement.bridge)
        #expect(await fixture.runtime.bridge(forModel: upgradeModelID) === staged.replacement.bridge)
        #expect(await fixture.loop.slotMTPStatusForTesting(modelId: upgradeModelID)?.active == true)
        #expect(fixture.originalEngine.shutdownCount == 1)
        #expect(await fixture.loop.outstandingKVReservationBytesForTesting() == 0)
        await fixture.clean()
    }

    @Test("stale original after unload and reload cannot replace the new runtime")
    func staleOriginalCannotPublish() async throws {
        let fixture = try await ProviderUpgradeFixture.make()
        defer { fixture.cleanFiles() }
        let staged = try #require(try await fixture.loop.prepareMTPUpgrade(upgradeModelID))
        await fixture.runtime.unregister(modelId: upgradeModelID)
        await fixture.loop.removeModelSlotForTesting(modelId: upgradeModelID)
        await fixture.original.shutdown()
        let reloadedEngine = UpgradeScriptedEngine(bytes: 1 << 30)
        let reloaded = EngineV2Bridge(engine: reloadedEngine, modelId: upgradeModelID,
            tokenizer: TokenizerHandle(MTPFloorTokenizer()), eosTokenIds: [])
        await fixture.runtime.register(modelId: upgradeModelID, bridge: reloaded)
        await fixture.loop.installModelSlotForTesting(modelId: upgradeModelID,
            container: mtpFloorContainer(), tokenizer: TokenizerHandle(MTPFloorTokenizer()),
            engineV2: reloaded, sizing: mtpFloorSizing(weightsGiB: 1), modelType: "gemma4")
        await #expect(throws: CancellationError.self) {
            _ = try await fixture.loop.commitMTPUpgradeIfIdle(staged)
        }
        await fixture.loop.discardMTPUpgrade(staged)
        #expect(await fixture.loop.slotBridgeForTesting(modelId: upgradeModelID) === reloaded)
        #expect(await fixture.runtime.bridge(forModel: upgradeModelID) === reloaded)
        #expect(reloadedEngine.shutdownCount == 0)
        #expect(fixture.factory.latest?.shutdownCount == 1)
        #expect(await fixture.loop.outstandingKVReservationBytesForTesting() == 0)
        await fixture.clean()
    }

    @Test("local admission waits through publication and acquires the replacement bridge")
    func admissionDuringPublicationSeesCoherentReplacement() async throws {
        let shutdown = UpgradeBarrier()
        let fixture = try await ProviderUpgradeFixture.make(shutdownBarrier: shutdown)
        defer { fixture.cleanFiles() }
        let staged = try #require(try await fixture.loop.prepareMTPUpgrade(upgradeModelID))
        let commit = Task { try await fixture.loop.commitMTPUpgradeIfIdle(staged) }
        await shutdown.observeEntry()
        #expect(await fixture.loop.slotBridgeForTesting(modelId: upgradeModelID) === staged.replacement.bridge)
        #expect(await fixture.runtime.bridge(forModel: upgradeModelID) === staged.replacement.bridge)
        let admission = Task { try await fixture.loop.acquireModelForLocal(upgradeModelID) }
        for _ in 0..<2_000 {
            if await fixture.loop.upgradeAdmissionWaiterCount() > 0 { break }
            try await Task.sleep(for: .milliseconds(1))
        }
        #expect(await fixture.loop.upgradeAdmissionWaiterCount() > 0)
        #expect(await !fixture.loop.hasLocalReservation(upgradeModelID))
        await shutdown.release()
        #expect(try await commit.value)
        let acquired = try await admission.value
        #expect(acquired.engineV2Bridge === staged.replacement.bridge)
        await acquired.releaseToken.fire()
        #expect(await fixture.loop.outstandingKVReservationBytesForTesting() == 0)
        await fixture.clean()
    }

    @Test("actual preparation build failure releases its lease and keeps the original registered")
    func failedBuildUnwindsActualPreparation() async throws {
        let fixture = try await ProviderUpgradeFixture.make()
        defer { fixture.cleanFiles() }
        fixture.factory.failBuild()
        await #expect(throws: UpgradeScriptedFactory.Failure.self) {
            _ = try await fixture.loop.prepareMTPUpgrade(upgradeModelID)
        }
        await fixture.checkOriginal()
        #expect(await fixture.loop.outstandingKVReservationBytesForTesting() == 0)
        await fixture.clean()
    }

    @Test("cancellation or shutdown while queued on the reslice gate cannot publish", arguments: [false, true])
    func interruptionWhileWaitingForGate(shutdown: Bool) async throws {
        let fixture = try await ProviderUpgradeFixture.make()
        defer { fixture.cleanFiles() }
        let staged = try #require(try await fixture.loop.prepareMTPUpgrade(upgradeModelID))
        await fixture.loop.acquireResliceGateForTesting()
        let task = Task {
            await MTPIdleUpgrade.run(prepare: { staged },
                commitIfIdle: { try await fixture.loop.commitMTPUpgradeIfIdle($0) },
                discard: { await fixture.loop.discardMTPUpgrade($0) }, pause: {})
        }
        for _ in 0..<2_000 {
            if await fixture.loop.upgradeResliceWaiterCount() > 0 { break }
            try await Task.sleep(for: .milliseconds(1))
        }
        let waiting = await fixture.loop.upgradeResliceWaiterCount() > 0
        if shutdown { await fixture.loop.beginShutdownForTesting() }
        else { task.cancel() }
        await fixture.loop.releaseResliceGateForTesting()
        #expect(waiting, "test must interrupt the real actor while queued for reslice")
        #expect(await task.value == .cancelled)
        await fixture.checkOriginal()
        #expect(fixture.factory.latest?.shutdownCount == 1)
        #expect(await fixture.loop.outstandingKVReservationBytesForTesting() == 0)
        await fixture.clean()
    }
}
