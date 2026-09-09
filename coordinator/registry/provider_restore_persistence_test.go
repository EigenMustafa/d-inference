package registry

import (
	"context"
	"io"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/eigeninference/d-inference/coordinator/attestation"
	"github.com/eigeninference/d-inference/coordinator/protocol"
	"github.com/eigeninference/d-inference/coordinator/store"
)

type orderedRestoreStore struct {
	store.Store
	entered chan store.ProviderRecord
	release chan struct{}
	written chan store.ProviderRecord
	once    sync.Once
}

func (s *orderedRestoreStore) UpsertProvider(ctx context.Context, rec store.ProviderRecord) error {
	blocked := false
	s.once.Do(func() { blocked = true; s.entered <- rec })
	if blocked {
		select {
		case <-s.release:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	err := s.Store.UpsertProvider(ctx, rec)
	s.written <- rec
	return err
}
func waitPersisted(t *testing.T, ch <-chan store.ProviderRecord) store.ProviderRecord {
	t.Helper()
	select {
	case rec := <-ch:
		return rec
	case <-time.After(2 * time.Second):
		t.Fatal("persist did not finish")
		return store.ProviderRecord{}
	}
}

func TestProviderIncompleteIdentityRemainsUnpublishedAfterDisconnect(t *testing.T) {
	base := store.NewMemory(store.Config{})
	if err := base.UpsertProvider(context.Background(), store.ProviderRecord{ID: "history", SerialNumber: "serial", SEPublicKey: "se", LastSeen: time.Now().Add(-time.Hour), LifetimeTokensGenerated: 700}); err != nil {
		t.Fatal(err)
	}
	st := &orderedRestoreStore{Store: base, entered: make(chan store.ProviderRecord, 1), release: make(chan struct{}), written: make(chan store.ProviderRecord, 4)}
	r := New(slog.New(slog.NewTextHandler(io.Discard, nil)))
	r.SetStore(st)
	// Reproduce the dangerous state: the initial async snapshot runs AFTER SE
	// evidence arrived but BEFORE history restoration completed.
	p := &Provider{ID: "pending", stateRestorePending: true, AttestationResult: &attestation.VerificationResult{SerialNumber: "serial", PublicKey: "se"}, Status: StatusOnline}
	r.PersistProvider(p)
	first := waitPersisted(t, st.entered)
	if first.SerialNumber != "" || first.SEPublicKey != "" {
		t.Fatal("partial state published restore identity")
	}
	p.mu.Lock()
	p.Status = StatusOffline
	p.mu.Unlock()
	r.PersistProvider(p) // disconnect persistence can finish after removal from live IDs
	close(st.release)
	for i := 0; i < 2; i++ {
		rec := waitPersisted(t, st.written)
		if rec.SerialNumber != "" || rec.SEPublicKey != "" {
			t.Fatal("disconnected pending row became a restore candidate")
		}
	}
	got, err := base.GetProviderForRestore(context.Background(), "serial", "se", nil)
	if err != nil || got == nil || got.ID != "history" {
		t.Fatalf("partial row shadowed history: %+v %v", got, err)
	}
}

func TestProviderInitialPersistCannotOverwriteCompletedRestore(t *testing.T) {
	base := store.NewMemory(store.Config{})
	st := &orderedRestoreStore{Store: base, entered: make(chan store.ProviderRecord, 1), release: make(chan struct{}), written: make(chan store.ProviderRecord, 4)}
	r := New(slog.New(slog.NewTextHandler(io.Discard, nil)))
	r.SetStore(st)
	p := r.Register("current", nil, &protocol.RegisterMessage{})
	waitPersisted(t, st.entered) // initial write is blocked before committing
	p.mu.Lock()
	if !p.stateRestorePending {
		t.Fatal("registration did not protect incomplete restoration")
	}
	p.AttestationResult = &attestation.VerificationResult{SerialNumber: "serial", PublicKey: "se"}
	p.Stats.TokensGenerated = 700
	p.AccountID = "owner"
	p.mu.Unlock()
	p.CompleteProviderStateRestore()
	r.PersistProvider(p)
	close(st.release)
	waitPersisted(t, st.written)
	waitPersisted(t, st.written)
	got, err := base.GetProviderRecord(context.Background(), "current")
	if err != nil || got.SerialNumber != "serial" || got.SEPublicKey != "se" || got.LifetimeTokensGenerated != 700 || got.AccountID != "owner" {
		t.Fatalf("late initial persist clobbered complete state: %+v %v", got, err)
	}
}
