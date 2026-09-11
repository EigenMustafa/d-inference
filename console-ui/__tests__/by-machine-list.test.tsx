import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  ByMachineList,
  machineLabel,
  displayLabels,
  type MachineEarnings,
} from "@/app/providers/earnings/ByMachineList";

const machine = (overrides: Partial<MachineEarnings>): MachineEarnings => ({
  provider_key: "k".repeat(44),
  total_micro_usd: 1_500_000,
  total_usd: "1.500000",
  job_count: 42,
  prompt_tokens: 1000,
  completion_tokens: 500,
  last_earned_at: "2026-09-10T12:00:00Z",
  ...overrides,
});

describe("machineLabel", () => {
  it("uses chip and memory when known", () => {
    expect(machineLabel(machine({ chip_name: "Apple M4 Max", memory_gb: 64 }))).toBe(
      "Apple M4 Max · 64 GB",
    );
  });

  it("falls back to a key suffix without hardware info", () => {
    expect(machineLabel(machine({ provider_key: "abcdef123456" }))).toBe("Machine …123456");
  });

  it("labels the empty key as Other", () => {
    expect(machineLabel(machine({ provider_key: "" }))).toBe("Other");
  });
});

describe("displayLabels", () => {
  it("disambiguates identical hardware with key suffixes", () => {
    const labels = displayLabels([
      machine({ provider_key: "key-one-aaaaaa", chip_name: "Apple M4 Max", memory_gb: 64 }),
      machine({ provider_key: "key-two-bbbbbb", chip_name: "Apple M4 Max", memory_gb: 64 }),
      machine({ provider_key: "key-three-cccccc", chip_name: "Apple M3 Pro", memory_gb: 36 }),
    ]);
    expect(labels).toEqual([
      "Apple M4 Max · 64 GB (…aaaaaa)",
      "Apple M4 Max · 64 GB (…bbbbbb)",
      "Apple M3 Pro · 36 GB",
    ]);
  });
});

describe("ByMachineList", () => {
  it("renders one row per machine with earnings and jobs", () => {
    render(
      <ByMachineList
        machines={[
          machine({ chip_name: "Apple M4 Max", memory_gb: 64 }),
          machine({ provider_key: "zzzz11223344", total_usd: "0.250000" }),
        ]}
      />,
    );
    expect(screen.getByText("Apple M4 Max · 64 GB")).toBeTruthy();
    expect(screen.getByText("Machine …223344")).toBeTruthy();
    expect(screen.getByText("$1.500000")).toBeTruthy();
    expect(screen.getByText("$0.250000")).toBeTruthy();
    // "Last earned" was cut deliberately — it read as "last active" but only
    // reflected earnings rows, going stale for online-but-idle machines.
    expect(screen.queryByText("Last active")).toBeNull();
  });

  it("shows the unattributed bucket as Other with an explainer", () => {
    render(<ByMachineList machines={[machine({ provider_key: "" })]} />);
    expect(screen.getByText("Other")).toBeTruthy();
    expect(screen.getByText(/not attributed to a machine/)).toBeTruthy();
  });

  it("renders nothing when there are no machines", () => {
    const { container } = render(<ByMachineList machines={[]} />);
    expect(container.innerHTML).toBe("");
  });
});
