import Foundation
import MLXLMCommon
import Testing
@testable import ProviderCore
import ProviderCoreFoundation

private let standaloneUpgradeModelID = "gemma-4-26b-qat-4bit"

private final class StandaloneUpgradeScriptedEngine: CBv2Engine, @unchecked Sendable {
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

private final class StandaloneUpgradeScriptedFactory: @unchecked Sendable {
    enum Failure: Error { case injected }
    private let lock = NSLock()
    private var fail = false
    private var built: [StandaloneUpgradeScriptedEngine] = []
    func failBuild() { lock.withLock { fail = true } }
    var latest: StandaloneUpgradeScriptedEngine? { lock.withLock { built.last } }
    func make(_ bytes: Int) throws -> StandaloneUpgradeScriptedEngine {
        try lock.withLock {
            if fail { throw Failure.injected }
            let engine = StandaloneUpgradeScriptedEngine(bytes: bytes)
            built.append(engine)
            return engine
        }
    }
}


private actor StandaloneUpgradeSlowCatalog: SpecDecCatalogLooking {
    let gate: UpgradeBarrier
    init(_ gate: UpgradeBarrier) { self.gate = gate }
    func cachedModel(id: String) -> CatalogModel? { nil }
    func model(id: String) async throws -> CatalogModel? { await gate.wait(); return nil }
}

private struct StandaloneUpgradeFixture {
    let server: StandaloneServer
    let original: EngineV2Bridge
    let originalEngine: StandaloneUpgradeScriptedEngine
    let factory: StandaloneUpgradeScriptedFactory
    let artifact: SpecDecArtifact
    let createdScannerPath: URL?

    static func make(shutdownBarrier: UpgradeBarrier? = nil, useLocalAssistant: Bool = true) async throws -> Self {
        let artifact = try mtpFloorArtifact()
        var createdScannerPath: URL?
        if ModelScanner.resolveLocalPath(modelID: standaloneUpgradeModelID) == nil {
            let hub = try #require(ModelScanner.defaultCacheDirectory())
            let model = hub.appendingPathComponent("models--\(standaloneUpgradeModelID)")
            let ownedModel = !FileManager.default.fileExists(atPath: model.path)
            let snapshot = model.appendingPathComponent("snapshots/standalone-mtp-upgrade-test-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
            try Data("{}".utf8).write(to: snapshot.appendingPathComponent("config.json"))
            createdScannerPath = ownedModel ? model : snapshot
        }
        let server = StandaloneServer(config: .init(mtpMode: .auto, mtpDrafterPath: useLocalAssistant ? artifact.directory.path : nil),
            models: [ModelInfo(id: standaloneUpgradeModelID, modelType: "gemma4", sizeBytes: 1, estimatedMemoryGb: 1)])
        let factory = StandaloneUpgradeScriptedFactory()
        await server.setV2TestHooksForTesting(.init(physicalMemoryBytes: 64 << 30,
            assistantLoader: MTPFloorAssistantLoader(), makeEngine: { _, bytes in try factory.make(bytes) }))
        let engine = StandaloneUpgradeScriptedEngine(bytes: 1 << 30, shutdownBarrier: shutdownBarrier)
        let bridge = EngineV2Bridge(engine: engine, modelId: standaloneUpgradeModelID,
            tokenizer: TokenizerHandle(MTPFloorTokenizer()), eosTokenIds: [])
        await server.installStandaloneUpgradeFixture(bridge)
        return Self(server: server, original: bridge, originalEngine: engine,
            factory: factory, artifact: artifact, createdScannerPath: createdScannerPath)
    }

    func checkOriginal() async {
        #expect(await server.upgradeBridge() === original)
        #expect(originalEngine.shutdownCount == 0)
    }
    func clean() async {
        await server.cleanStandaloneUpgradeFixture()
        try? FileManager.default.removeItem(at: artifact.directory)
        if let createdScannerPath { try? FileManager.default.removeItem(at: createdScannerPath) }
    }
}

private extension StandaloneServer {
    func installStandaloneUpgradeFixture(_ bridge: EngineV2Bridge) {
        lifecycleState = .running
        slots[standaloneUpgradeModelID] = CachedSlot(bundle: .init(targetOnly: bridge),
            container: mtpFloorContainer(), tokenizer: TokenizerHandle(MTPFloorTokenizer()),
            modelType: "gemma4", isVLM: false, sizing: mtpFloorSizing(weightsGiB: 1),
            lastUsedAt: .now, cacheEligibleWeightHash: String(repeating: "a", count: 64))
    }
    func cleanStandaloneUpgradeFixture() async {
        lifecycleState = .stopping
        for slot in slots.values { await slot.bridge.shutdown(); slot.bundle.releaseAssistant() }
        slots.removeAll()
        await specDecFunnel.shutdown()
        lifecycleState = .stopped
    }
    func upgradeBridge() -> EngineV2Bridge? { slots[standaloneUpgradeModelID]?.bridge }
    func upgradeAdmissionWaiterCount() -> Int { mtpUpgradeWaiters[standaloneUpgradeModelID]?.count ?? 0 }
    func upgradeWeightHash() -> String? { slots[standaloneUpgradeModelID]?.cacheEligibleWeightHash }
    func resliceForUpgradeAccountingTest() async {
        isLoadingAny = true
        await resliceGrowSurvivors()
        isLoadingAny = false
        releaseLoadGateWaiters()
    }
    func removeUpgradeTarget() { slots.removeValue(forKey: standaloneUpgradeModelID) }
}

@Suite("Standalone assistant upgrade integration", .serialized)
struct StandaloneMTPUpgradeTests {
    init() { _ = LiveInferenceFixtures.ensureMetallibColocated() }

    @Test("local reservations and engine work preserve the old engine until idle")
    func busyThenIdle() async throws {
        let fixture = try await StandaloneUpgradeFixture.make()
        let staged = try #require(try await fixture.server.prepareMTPUpgrade(standaloneUpgradeModelID))
        let stagingBytes = await fixture.server.mtpStagingBytes
        #expect(stagingBytes == fixture.artifact.residentBytes + EngineV2KVSizing.minimumServiceableGrantBytes)
        await fixture.server.resliceForUpgradeAccountingTest()
        let expectedGrant = UnifiedMemoryCap.kvBudgetBytes(physicalBytes: 64 << 30,
            residentWeightBytes: (1 << 30) + stagingBytes,
            activationReserveBytes: await fixture.server.resolvedActivationReserveBytes,
            configReserveBytes: 0)
        #expect(await fixture.server.debugEngineKVGrant(modelId: standaloneUpgradeModelID) == Int(expectedGrant))
        let acquired = try await fixture.server.acquireModel(standaloneUpgradeModelID)
        #expect(acquired.engineV2Bridge === fixture.original)
        #expect(try await !fixture.server.commitMTPUpgradeIfIdle(staged))
        await acquired.releaseToken.fire()
        fixture.originalEngine.setBusy(true)
        #expect(try await !fixture.server.commitMTPUpgradeIfIdle(staged))
        fixture.originalEngine.setBusy(false)
        await fixture.checkOriginal()
        #expect(try await fixture.server.commitMTPUpgradeIfIdle(staged))
        #expect(await fixture.server.upgradeBridge() === staged.replacement.bridge)
        #expect(await fixture.server.upgradeWeightHash() == String(repeating: "a", count: 64))
        #expect(fixture.originalEngine.shutdownCount == 1)
        #expect(await fixture.server.debugOutstandingKVReservationBytes() == 0)
        #expect(await fixture.server.mtpStagingBytes == 0)
        await fixture.clean()
    }

    @Test("arriving admission waits for old idle shutdown and receives only the replacement")
    func admissionWaitsForCutover() async throws {
        let barrier = UpgradeBarrier()
        let fixture = try await StandaloneUpgradeFixture.make(shutdownBarrier: barrier)
        let staged = try #require(try await fixture.server.prepareMTPUpgrade(standaloneUpgradeModelID))
        let commit = Task { try await fixture.server.commitMTPUpgradeIfIdle(staged) }
        await barrier.observeEntry()
        let admission = Task { try await fixture.server.acquireModel(standaloneUpgradeModelID) }
        for _ in 0..<2_000 {
            if await fixture.server.upgradeAdmissionWaiterCount() > 0 { break }
            try await Task.sleep(for: .milliseconds(1))
        }
        #expect(await fixture.server.upgradeAdmissionWaiterCount() > 0)
        #expect(await fixture.server.debugSlotReservationCount(modelId: standaloneUpgradeModelID) == 0)
        await barrier.release()
        #expect(try await commit.value)
        let acquired = try await admission.value
        #expect(acquired.engineV2Bridge === staged.replacement.bridge)
        await acquired.releaseToken.fire()
        await fixture.clean()
    }

    @Test("failed real staging keeps target and releases both reservation ledgers")
    func buildFailure() async throws {
        let fixture = try await StandaloneUpgradeFixture.make()
        fixture.factory.failBuild()
        await #expect(throws: StandaloneUpgradeScriptedFactory.Failure.self) {
            _ = try await fixture.server.prepareMTPUpgrade(standaloneUpgradeModelID)
        }
        await fixture.checkOriginal()
        #expect(await fixture.server.debugOutstandingKVReservationBytes() == 0)
        #expect(await fixture.server.mtpStagingBytes == 0)
        let acquired = try await fixture.server.acquireModel(standaloneUpgradeModelID)
        #expect(acquired.engineV2Bridge === fixture.original)
        await acquired.releaseToken.fire()
        await fixture.clean()
    }

    @Test("cancellation while busy discards the candidate and keeps target serving")
    func cancellationWhileBusy() async throws {
        let fixture = try await StandaloneUpgradeFixture.make()
        let staged = try #require(try await fixture.server.prepareMTPUpgrade(standaloneUpgradeModelID))
        let pause = UpgradeBarrier()
        fixture.originalEngine.setBusy(true)
        let task = Task {
            await MTPIdleUpgrade.run(prepare: { staged },
                commitIfIdle: { try await fixture.server.commitMTPUpgradeIfIdle($0) },
                discard: { await fixture.server.discardMTPUpgrade($0) },
                pause: { await pause.wait() })
        }
        await pause.observeEntry()
        task.cancel()
        await pause.release()
        #expect(await task.value == .cancelled)
        await fixture.checkOriginal()
        #expect(fixture.factory.latest?.shutdownCount == 1)
        #expect(await fixture.server.debugOutstandingKVReservationBytes() == 0)
        #expect(await fixture.server.mtpStagingBytes == 0)
        await fixture.clean()
    }

    @Test("unloaded target stays accounted while retained by stale preparation")
    func staleTargetAccounting() async throws {
        let fixture = try await StandaloneUpgradeFixture.make()
        let staged = try #require(try await fixture.server.prepareMTPUpgrade(standaloneUpgradeModelID))
        let before = await fixture.server.mtpStagingBytes
        await fixture.server.removeUpgradeTarget()
        #expect(await fixture.server.mtpStagingBytes == before + (1 << 30))
        await #expect(throws: CancellationError.self) {
            _ = try await fixture.server.commitMTPUpgradeIfIdle(staged)
        }
        await fixture.server.discardMTPUpgrade(staged)
        #expect(await fixture.server.upgradeBridge() == nil)
        #expect(await fixture.server.debugOutstandingKVReservationBytes() == 0)
        #expect(await fixture.server.mtpStagingBytes == 0)
        await fixture.original.shutdown()
        await fixture.clean()
    }

    @Test("deferred serving-set removal completes after idle publication")
    func deferredRemovalAtCutover() async throws {
        let barrier = UpgradeBarrier()
        let fixture = try await StandaloneUpgradeFixture.make(shutdownBarrier: barrier)
        let staged = try #require(try await fixture.server.prepareMTPUpgrade(standaloneUpgradeModelID))
        let commit = Task { try await fixture.server.commitMTPUpgradeIfIdle(staged) }
        await barrier.observeEntry()
        #expect(await fixture.server.setModels([]))
        #expect(await fixture.server.hasDeferredModelsUpdateForTesting())
        await barrier.release()
        #expect(try await commit.value)
        #expect(await !fixture.server.hasDeferredModelsUpdateForTesting())
        #expect(await fixture.server.models.isEmpty)
        #expect(await fixture.server.mtpUpgradeTransitions.isEmpty)
        await fixture.clean()
    }

    @Test("slow failed assistant fetch leaves real standalone acquisitions available")
    func slowFetchKeepsServing() async throws {
        let fixture = try await StandaloneUpgradeFixture.make(useLocalAssistant: false)
        let gate = UpgradeBarrier()
        let funnel = SpecDecArtifactFunnel(resolver: SpecDecResolver(),
            catalog: StandaloneUpgradeSlowCatalog(gate))
        await fixture.server.setSpecDecFunnelForTesting(funnel)
        #expect(try await fixture.server.prepareMTPUpgrade(standaloneUpgradeModelID) == nil)
        await gate.observeEntry()
        for _ in 0..<3 {
            let acquired = try await fixture.server.acquireModel(standaloneUpgradeModelID)
            #expect(acquired.engineV2Bridge === fixture.original)
            await acquired.releaseToken.fire()
        }
        await fixture.checkOriginal()
        #expect(await fixture.server.mtpStagingBytes == 0)
        await gate.release()
        await fixture.clean()
    }

    @Test("independent standalone providers activate separately without interrupting the busy provider")
    func independentProviders() async throws {
        let first = try await StandaloneUpgradeFixture.make()
        let second = try await StandaloneUpgradeFixture.make()
        async let firstPrepared = first.server.prepareMTPUpgrade(standaloneUpgradeModelID)
        async let secondPrepared = second.server.prepareMTPUpgrade(standaloneUpgradeModelID)
        let (firstValue, secondValue) = try await (firstPrepared, secondPrepared)
        let firstStaged = try #require(firstValue)
        let secondStaged = try #require(secondValue)
        let acquired = try await first.server.acquireModel(standaloneUpgradeModelID)
        #expect(try await !first.server.commitMTPUpgradeIfIdle(firstStaged))
        #expect(try await second.server.commitMTPUpgradeIfIdle(secondStaged))
        await first.checkOriginal()
        await acquired.releaseToken.fire()
        #expect(try await first.server.commitMTPUpgradeIfIdle(firstStaged))
        #expect(await first.server.mtpStagingBytes == 0)
        #expect(await second.server.mtpStagingBytes == 0)
        await second.clean()
        await first.clean()
    }

}
