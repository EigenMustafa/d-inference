package api

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/eigeninference/d-inference/coordinator/protocol"
	"github.com/eigeninference/d-inference/coordinator/registry"
	"github.com/eigeninference/d-inference/coordinator/store"
)

type restoreTrackingStore struct {
	store.Store
	lookups int
}

func (s *restoreTrackingStore) ListProviderRecords(context.Context) ([]store.ProviderRecord, error) {
	panic("startup must not scan all historical providers")
}
func (s *restoreTrackingStore) GetProviderForRestore(ctx context.Context, serial, key, exclude string) (*store.ProviderRecord, error) {
	s.lookups++
	return s.Store.GetProviderForRestore(ctx, serial, key, exclude)
}

func TestProviderRestoreLoadsAfterStartupAndNeverResurrectsHardware(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	st := &restoreTrackingStore{Store: store.NewMemory(store.Config{})}
	reg := registry.New(logger)
	reg.SetStore(st)
	srv := NewServer(reg, st, ServerConfig{}, logger)
	defer srv.Close()
	// History earned after Server creation must be visible at reconnect.
	chain, _ := json.Marshal([][]byte{[]byte("staged-only")})
	rec := store.ProviderRecord{ID: "prior", SerialNumber: "serial", SEPublicKey: "key", LastSeen: time.Now(), TrustLevel: string(registry.TrustHardware), Attested: true, MDAVerified: true, MDACertChain: chain, LifetimeTokensGenerated: 1234, AccountID: "owner"}
	if err := st.UpsertProvider(context.Background(), rec); err != nil {
		t.Fatal(err)
	}
	if err := st.UpsertReputation(context.Background(), "prior", store.ReputationRecord{TotalJobs: 12, SuccessfulJobs: 10}); err != nil {
		t.Fatal(err)
	}
	p := reg.Register("current", nil, &protocol.RegisterMessage{})
	p.SetAttested(true, registry.TrustSelfSigned)
	srv.restorePersistedProviderState(p, "serial", "key")
	p.Mu().Lock()
	defer p.Mu().Unlock()
	if p.TrustLevel != registry.TrustSelfSigned || p.MDAVerified || len(p.MDACertChain) != 0 {
		t.Fatalf("resurrected live hardware proof: trust=%s MDA=%v", p.TrustLevel, p.MDAVerified)
	}
	if p.Stats.TokensGenerated != 1234 || p.Reputation.TotalJobs != 12 || p.AccountID != "owner" {
		t.Fatalf("lost durable counters/account/reputation")
	}
}

func TestProviderRestoreNotReachedWithoutValidAttestation(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	st := &restoreTrackingStore{Store: store.NewMemory(store.Config{})}
	reg := registry.New(logger)
	srv := &Server{registry: reg, store: st, logger: logger}
	for _, evidence := range []json.RawMessage{nil, json.RawMessage(`{"bad":`)} {
		p := reg.Register(string(evidence)+"p", nil, &protocol.RegisterMessage{})
		srv.verifyProviderAttestation(p.ID, p, &protocol.RegisterMessage{Attestation: evidence})
	}
	if st.lookups != 0 {
		t.Fatal("looked up durable state before live attestation verification")
	}
}
