package registry

// CacheReceiptReason is a closed, privacy-safe diagnostic vocabulary. Never
// put a nonce, scope, prompt hash, provider ID or model-supplied string in it.
type CacheReceiptReason string

const (
	CacheReceiptAccepted              CacheReceiptReason = "accepted"
	CacheReceiptInvalid               CacheReceiptReason = "invalid_shape"
	CacheReceiptInactive              CacheReceiptReason = "routing_inactive"
	CacheReceiptProviderMissing       CacheReceiptReason = "provider_missing"
	CacheReceiptProtocol              CacheReceiptReason = "unsupported_protocol"
	CacheReceiptCapabilityUnavailable CacheReceiptReason = "capability_unavailable"
	CacheReceiptCapabilityFenced      CacheReceiptReason = "capability_fenced"
	CacheReceiptAttemptUnavailable    CacheReceiptReason = "attempt_missing_or_expired"
	CacheReceiptAttemptBinding        CacheReceiptReason = "attempt_binding_mismatch"
	CacheReceiptDuplicateLookup       CacheReceiptReason = "duplicate_lookup"
	CacheReceiptLookupNotSeen         CacheReceiptReason = "lookup_not_accepted"
	CacheReceiptCapabilityChanged     CacheReceiptReason = "capability_changed"
	CacheReceiptConnectionChanged     CacheReceiptReason = "connection_changed"
	CacheReceiptIdentityMismatch      CacheReceiptReason = "identity_mismatch"
	CacheReceiptPromptMismatch        CacheReceiptReason = "prompt_anchor_mismatch"
	CacheReceiptMatchedMismatch       CacheReceiptReason = "matched_anchor_mismatch"
	CacheReceiptReadyMismatch         CacheReceiptReason = "ready_anchor_mismatch"
	CacheReceiptNonAdvancingReady     CacheReceiptReason = "nonadvancing_ready"
	CacheReceiptSequence              CacheReceiptReason = "stale_sequence"
	CacheReceiptRouteKey              CacheReceiptReason = "route_key_unavailable"
)

type CacheReceiptResult struct {
	Accepted bool
	Reason   CacheReceiptReason
	mismatch bool
}

func rejectCacheReceipt(reason CacheReceiptReason) CacheReceiptResult {
	return CacheReceiptResult{Reason: reason}
}
func mismatchCacheReceipt(reason CacheReceiptReason) CacheReceiptResult {
	return CacheReceiptResult{Reason: reason, mismatch: true}
}
