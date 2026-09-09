package api

import (
	"context"
	"time"

	"github.com/eigeninference/d-inference/coordinator/registry"
)

// Called only after verifyProviderAttestation has verified the live SE evidence.
// The store contributes reputation/counters and staged proofs, never live trust.
func (s *Server) restorePersistedProviderState(p *registry.Provider, serial, seKey string) {
	if s.store == nil || (serial == "" && seKey == "") {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	started := time.Now()
	rec, err := s.store.GetProviderForRestore(ctx, serial, seKey, p.ID)
	if err != nil {
		s.logger.Warn("provider state lookup failed", "provider_id", p.ID,
			"duration_ms", time.Since(started).Milliseconds(), "error", err)
		return
	}
	if rec != nil {
		s.registry.RestoreProviderState(p, rec)
	}
}
