package store

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestProviderRestoreSelectsLatestPriorIdentity(t *testing.T) {
	for name, backend := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			st := NewCached(backend, DefaultCacheConfig()) // exercise decorator forwarding
			now := time.Now().UTC().Truncate(time.Microsecond)
			// Deliberately insert in a different order from last_seen, including a
			// newest in-progress registration with no inherited counters yet.
			rows := []ProviderRecord{
				{ID: "newest-prior", SerialNumber: "serial", SEPublicKey: "key", LastSeen: now, LifetimeTokensGenerated: 700, AccountID: "owner"},
				{ID: "old", SerialNumber: "serial", SEPublicKey: "key", LastSeen: now.Add(-time.Hour), LifetimeTokensGenerated: 5},
				{ID: "current", SerialNumber: "serial", SEPublicKey: "key", LastSeen: now.Add(time.Minute)},
				{ID: "key-only", SerialNumber: "different-serial", SEPublicKey: "fallback", LastSeen: now.Add(time.Hour)},
			}
			for _, p := range rows {
				p.Hardware = json.RawMessage(`{}`)
				p.Models = json.RawMessage(`[]`)
				p.RegisteredAt = now
				if err := st.UpsertProvider(context.Background(), p); err != nil {
					t.Fatal(err)
				}
			}
			for _, tc := range []struct{ serial, key, exclude, want string }{
				{"serial", "fallback", "current", "newest-prior"}, // serial takes priority
				{"missing", "fallback", "current", "key-only"},
				{"", "key", "current", "newest-prior"},
				{"", "", "current", ""},
				{"unknown", "unknown", "current", ""},
			} {
				got, err := st.GetProviderForRestore(context.Background(), tc.serial, tc.key, tc.exclude)
				if err != nil {
					t.Fatal(err)
				}
				if tc.want == "" {
					if got != nil {
						t.Fatalf("unexpected match: %+v", got)
					}
					continue
				}
				if got == nil || got.ID != tc.want {
					t.Fatalf("%+v: got %+v", tc, got)
				}
				if tc.want == "newest-prior" && (got.LifetimeTokensGenerated != 700 || got.AccountID != "owner") {
					t.Fatalf("lost state: %+v", got)
				}
			}
		})
	}
}

func TestPostgresRestoreIndexes(t *testing.T) {
	s := testPostgresStore(t)
	ctx := context.Background()
	// Both queries must have usable ordered indexes; an explicit no-seqscan
	// session checks index applicability, not production cardinality estimates.
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Release()
	if _, err := conn.Exec(ctx, "SET enable_seqscan = off"); err != nil {
		t.Fatal(err)
	}
	defer conn.Exec(ctx, "RESET enable_seqscan")
	for _, col := range []string{"serial_number", "se_public_key"} {
		rows, err := conn.Query(ctx, `EXPLAIN (FORMAT JSON) SELECT id FROM providers WHERE `+col+` = 's' AND `+col+` <> '' AND id <> 'current' ORDER BY last_seen DESC,id DESC LIMIT 1`)
		if err != nil {
			t.Fatal(err)
		}
		var plan string
		for rows.Next() {
			if err := rows.Scan(&plan); err != nil {
				t.Fatal(err)
			}
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			t.Fatal(err)
		}
		expected := "idx_providers_restore_serial"
		if col == "se_public_key" {
			expected = "idx_providers_restore_se_key"
		}
		if !strings.Contains(plan, expected) {
			t.Fatalf("missing %s: %s", expected, plan)
		}
	}
}

func TestListProviderRecordsRejectsPartialScan(t *testing.T) {
	databaseURL := newWithdrawableTestDatabase(t)
	s, err := NewPostgres(context.Background(), Config{DatabaseURL: databaseURL})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	now := time.Now()
	for i, id := range []string{"good", "bad"} {
		if err := s.UpsertProvider(context.Background(), ProviderRecord{ID: id, Hardware: json.RawMessage(`{}`), Models: json.RawMessage(`[]`), RegisteredAt: now, LastSeen: now.Add(-time.Duration(i) * time.Minute)}); err != nil {
			t.Fatal(err)
		}
	}
	// Deliberately malformed private test schema: the first row scans, then the
	// next cannot decode. Returning the already-read row would be silent data loss.
	if _, err := s.pool.Exec(context.Background(), `ALTER TABLE providers ALTER COLUMN failed_challenges TYPE TEXT USING failed_challenges::text; UPDATE providers SET failed_challenges='invalid' WHERE id='bad'`); err != nil {
		t.Fatal(err)
	}
	rows, err := s.ListProviderRecords(context.Background())
	if err == nil || rows != nil {
		t.Fatalf("partial success: rows=%v error=%v", rows, err)
	}
}
