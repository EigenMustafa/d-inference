import Foundation

/// Shared lifecycle for an optional replacement. Preparation never withdraws
/// the serving engine. Only the caller's atomic idle commit can publish it.
enum MTPIdleUpgrade {
    enum Outcome: Equatable { case notReady, installed, deferred, cancelled, failed }

    static func run<Candidate: Sendable>(
        maximumIdleChecks: Int = 120,
        prepare: @Sendable () async throws -> Candidate?,
        commitIfIdle: @Sendable (Candidate) async throws -> Bool,
        discard: @Sendable (Candidate) async -> Void,
        pause: @Sendable () async throws -> Void = { try await Task.sleep(for: .milliseconds(500)) }
    ) async -> Outcome {
        let candidate: Candidate
        do {
            try Task.checkCancellation()
            guard let prepared = try await prepare() else { return .notReady }
            candidate = prepared
        } catch is CancellationError { return .cancelled }
        catch { return .failed }
        var outcome = Outcome.deferred
        do {
            for _ in 0..<max(0, maximumIdleChecks) {
                try Task.checkCancellation()
                if try await commitIfIdle(candidate) { return .installed }
                try await pause()
            }
        } catch is CancellationError { outcome = .cancelled }
        catch { outcome = .failed }
        await discard(candidate)
        return outcome
    }
}
