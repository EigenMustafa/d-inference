import Foundation
import Testing
@testable import ProviderCore

private actor UpgradeBarrier {
    var entered = false
    var released = false
    var waiters: [CheckedContinuation<Void, Never>] = []
    var observers: [CheckedContinuation<Void, Never>] = []
    func wait() async {
        entered = true
        let observing = observers; observers.removeAll()
        for observer in observing { observer.resume() }
        if !released { await withCheckedContinuation { waiters.append($0) } }
    }
    func observeEntry() async {
        if !entered { await withCheckedContinuation { observers.append($0) } }
    }
    func release() {
        released = true
        let current = waiters; waiters.removeAll()
        for waiter in current { waiter.resume() }
    }
}

private actor UpgradeServingFixture {
    enum Failure: Error { case preparation, stale }
    var serving = 0
    var requests = 0
    var busy = true
    var stale = false
    var failPreparation = false
    var discarded = 0
    func serve() -> Int { requests += 1; return serving }
    func setBusy(_ value: Bool) { busy = value }
    func setStale() { stale = true }
    func setFailure() { failPreparation = true }
    func prepare() throws -> Int? {
        if failPreparation { throw Failure.preparation }
        return 1
    }
    func commit(_ candidate: Int) throws -> Bool {
        if stale { throw Failure.stale }
        if busy { return false }
        serving = candidate
        return true
    }
    func discard(_ candidate: Int) { discarded += 1 }
}

@Suite("MTP optional idle upgrade lifecycle")
struct MTPIdleUpgradeTests {
    @Test func slowPreparationKeepsTwoIndependentProvidersServing() async {
        let providers = [UpgradeServingFixture(), UpgradeServingFixture()]
        let fetch = UpgradeBarrier()
        let busy = UpgradeBarrier()
        let tasks = providers.map { provider in
            Task {
                await MTPIdleUpgrade.run(
                    prepare: { await fetch.wait(); return try await provider.prepare() },
                    commitIfIdle: { try await provider.commit($0) },
                    discard: { await provider.discard($0) },
                    pause: { await busy.wait() })
            }
        }
        await fetch.observeEntry()
        for provider in providers { #expect(await provider.serve() == 0) }
        await fetch.release()
        await busy.observeEntry()
        for provider in providers {
            #expect(await provider.serve() == 0)
            await provider.setBusy(false)
        }
        await busy.release()
        for task in tasks { #expect(await task.value == .installed) }
        for provider in providers {
            #expect(await provider.serve() == 1)
            #expect(await provider.discarded == 0)
        }
    }

    @Test func failedFetchAndBusyTimeoutNeverWithdrawOriginal() async {
        let provider = UpgradeServingFixture()
        await provider.setFailure()
        let failed = await MTPIdleUpgrade.run(
            prepare: { try await provider.prepare() },
            commitIfIdle: { try await provider.commit($0) },
            discard: { await provider.discard($0) })
        #expect(failed == .failed)
        #expect(await provider.serve() == 0)
        #expect(await provider.discarded == 0)
        let busy = UpgradeServingFixture()
        let deferred = await MTPIdleUpgrade.run(maximumIdleChecks: 2,
            prepare: { try await busy.prepare() },
            commitIfIdle: { try await busy.commit($0) },
            discard: { await busy.discard($0) }, pause: {})
        #expect(deferred == .deferred)
        #expect(await busy.serve() == 0)
        #expect(await busy.discarded == 1)
    }

    @Test func cancellationAndStaleGenerationDiscardUnpublishedReplacement() async {
        for cancel in [true, false] {
            let provider = UpgradeServingFixture()
            let gate = UpgradeBarrier()
            let task = Task {
                await MTPIdleUpgrade.run(
                    prepare: { try await provider.prepare() },
                    commitIfIdle: { try await provider.commit($0) },
                    discard: { await provider.discard($0) }, pause: { await gate.wait() })
            }
            await gate.observeEntry()
            if cancel { task.cancel() } else { await provider.setStale() }
            await provider.setBusy(false)
            await gate.release()
            #expect(await task.value == (cancel ? .cancelled : .failed))
            #expect(await provider.serve() == 0)
            #expect(await provider.discarded == 1)
        }
    }
}
