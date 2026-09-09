import Foundation

/// Static fleet grants exclude an unpublished replacement's assistant/KV and
/// any original target retained after concurrent unload. Actual allocation is
/// guarded separately by the process ledger and pending-load lease.
struct MTPStagingReservations {
    private struct Entry {
        let target: ObjectIdentifier
        let targetBytes: UInt64
        let replacementBytes: UInt64
    }
    private var entries: [ProcessMemoryLedger.Owner: Entry] = [:]

    func extraBytes(residentTargets: Set<ObjectIdentifier>) -> UInt64 {
        var total: UInt64 = 0
        var countedTargets = residentTargets
        for entry in entries.values {
            total = Self.adding(total, entry.replacementBytes)
            if countedTargets.insert(entry.target).inserted {
                total = Self.adding(total, entry.targetBytes)
            }
        }
        return total
    }

    mutating func reserve(_ lease: PendingModelLoadLease,
                         target: ObjectIdentifier, targetBytes: UInt64,
                         assistantBytes: UInt64, kvBytes: UInt64) {
        entries[lease.owner] = Entry(target: target, targetBytes: targetBytes,
            replacementBytes: Self.adding(assistantBytes, kvBytes))
    }

    mutating func release(_ lease: PendingModelLoadLease) {
        entries.removeValue(forKey: lease.owner)
    }

    static func adding(_ lhs: UInt64, _ rhs: UInt64) -> UInt64 {
        let (sum, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? .max : sum
    }
}
