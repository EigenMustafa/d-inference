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
	excluded := s.registry.ProviderIDs()
	for ctx.Err() == nil {
		rec, err := s.store.GetProviderForRestore(ctx, serial, seKey, excluded)
		if err != nil {
			s.logger.Warn("provider state lookup failed", "provider_id", p.ID,
				"duration_ms", time.Since(started).Milliseconds(), "error", err)
			return // keep incomplete identity unpublished; never shadow valid history
		}
		if rec != nil && s.registry.GetProvider(rec.ID) != nil {
			// A session may register after the exclusion snapshot. Retry without it;
			// no registry lock is held over database IO or reputation restoration.
			excluded = append(excluded, rec.ID)
			continue
		}
		if rec != nil {
			if err := s.registry.RestoreProviderState(p, rec); err != nil {
				s.logger.Warn("provider reputation restore failed", "provider_id", p.ID, "error", err)
				return
			}
		}
		p.CompleteProviderStateRestore()
		return
	}
}
