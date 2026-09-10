import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { StripePayoutsCard } from "@/components/payouts/StripePayoutsCard";
import { PayoutCoverageNotice } from "@/components/payouts/PayoutCoverageNotice";
import type { StripeStatus } from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = (await importOriginal()) as Record<string, unknown>;
  return { ...actual };
});

const NOTICE_SNIPPET = /does not, by itself, establish payout availability/;

const baseProps = {
  withdrawals: [],
  balanceMicroUsd: 5_000_000,
  onboardLoading: false,
  selectedCountry: "",
  onCountryChange: () => {},
  onOnboard: () => {},
  onOpenWithdraw: () => {},
  title: "Withdraw Earnings",
  icon: <span />,
  noun: "earnings",
  className: "",
};

const noAccountStatus: StripeStatus = {
  configured: true,
  has_account: false,
  status: "",
  min_withdraw_micro_usd: 1_000_000,
};

const restrictedStatus: StripeStatus = {
  ...noAccountStatus,
  has_account: true,
  status: "restricted",
};

const readyStatus: StripeStatus = {
  ...noAccountStatus,
  has_account: true,
  status: "ready",
};

describe("PayoutCoverageNotice", () => {
  it("renders the coverage caveat", () => {
    render(<PayoutCoverageNotice />);
    expect(screen.getByText(NOTICE_SNIPPET)).toBeTruthy();
  });
});

describe("StripePayoutsCard countryNotice slot", () => {
  it("shows the notice under the country picker before onboarding", () => {
    render(
      <StripePayoutsCard
        {...baseProps}
        status={noAccountStatus}
        countryNotice={<PayoutCoverageNotice />}
      />,
    );
    expect(screen.getByText(NOTICE_SNIPPET)).toBeTruthy();
  });

  it("shows the notice for accounts stuck in a restricted state", () => {
    render(
      <StripePayoutsCard
        {...baseProps}
        status={restrictedStatus}
        countryNotice={<PayoutCoverageNotice />}
      />,
    );
    expect(screen.getByText(NOTICE_SNIPPET)).toBeTruthy();
  });

  it("does not show the notice once payouts are ready (no country picker)", () => {
    render(
      <StripePayoutsCard
        {...baseProps}
        status={readyStatus}
        countryNotice={<PayoutCoverageNotice />}
      />,
    );
    expect(screen.queryByText(NOTICE_SNIPPET)).toBeNull();
  });

  it("renders nothing extra when the slot is not passed (billing unchanged)", () => {
    render(<StripePayoutsCard {...baseProps} status={noAccountStatus} />);
    expect(screen.queryByText(NOTICE_SNIPPET)).toBeNull();
  });
});
