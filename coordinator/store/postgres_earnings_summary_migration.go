package store

import (
	"context"
	"fmt"
	"log/slog"
	"time"
)

const earningsSummaryMigrationID = "backfill_earnings_summary_v1"

// Preserve existing live summaries; supply absent keys once. The committed
// marker prevents repeating both full-table GROUP BY scans on every restart.
func (s *PostgresStore) migrateEarningsSummary(ctx context.Context) error {
	started := time.Now()
	applied, err := s.applyEarningsSummaryMigration(ctx)
	result := "already_applied"
	if applied {
		result = "backfilled_missing_keys"
	}
	if err != nil {
		result = "failed"
	}
	slog.Info("postgres migration completed", "migration", earningsSummaryMigrationID,
		"result", result, "duration_ms", time.Since(started).Milliseconds())
	return err
}

func (s *PostgresStore) applyEarningsSummaryMigration(ctx context.Context) (bool, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return false, err
	}
	defer tx.Rollback(ctx)
	// Unique insert serializes concurrent attempts. Marker and backfills commit
	// together; failure/cancellation rolls back all three.
	tag, err := tx.Exec(ctx, `INSERT INTO schema_migrations (id) VALUES ($1) ON CONFLICT (id) DO NOTHING`, earningsSummaryMigrationID)
	if err != nil {
		return false, fmt.Errorf("store: claim earnings summary migration: %w", err)
	}
	if tag.RowsAffected() == 0 {
		return false, nil
	}
	for _, statement := range []string{
		`INSERT INTO earnings_summary (key, key_type, total_count, total_micro_usd, total_prompt_tokens, total_completion_tokens, updated_at)
		 SELECT account_id, 'account', COUNT(*), COALESCE(SUM(amount_micro_usd), 0),
		        COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0), NOW()
		 FROM provider_earnings
		 WHERE account_id != ''
		 GROUP BY account_id
		 ON CONFLICT (key, key_type) DO NOTHING`,
		`INSERT INTO earnings_summary (key, key_type, total_count, total_micro_usd, total_prompt_tokens, total_completion_tokens, updated_at)
		 SELECT provider_key, 'provider', COUNT(*), COALESCE(SUM(amount_micro_usd), 0),
		        COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0), NOW()
		 FROM provider_earnings
		 WHERE provider_key != ''
		 GROUP BY provider_key
		 ON CONFLICT (key, key_type) DO NOTHING`,
	} {
		if _, err := tx.Exec(ctx, statement); err != nil {
			return false, fmt.Errorf("store: backfill earnings summary: %w", err)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return false, err
	}
	return true, nil
}
