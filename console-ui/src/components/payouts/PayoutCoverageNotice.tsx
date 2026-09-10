import { Globe } from "lucide-react";

// Payout-coverage caveat shown next to the Stripe country picker. Wording
// agreed with legal — running a provider does not by itself guarantee a
// payout path in every region (Stripe Connect coverage varies by country).
export function PayoutCoverageNotice() {
  return (
    <div className="flex items-start gap-2.5 rounded-lg border border-accent-amber/20 bg-accent-amber-dim px-4 py-3 mb-4">
      <Globe size={16} className="mt-0.5 shrink-0 text-accent-amber" aria-hidden />
      <p className="text-sm leading-relaxed text-text-secondary">
        Please note that operating as a provider on the platform does not, by
        itself, establish payout availability in certain regions. We are
        actively evaluating additional jurisdictions for payment integration;
        however, we cannot confirm whether or when coverage will extend to
        your country.
      </p>
    </div>
  );
}
