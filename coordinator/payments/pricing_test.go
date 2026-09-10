package payments

import (
	"testing"

	"github.com/eigeninference/d-inference/coordinator/store"
)

func int64Ptr(v int64) *int64 { return &v }

func TestDefaultRates(t *testing.T) {
	// Without DB-configured prices, every model gets the fallback defaults and a
	// cache-read rate derived from the default input price.
	got := DefaultRates()
	want := Rates{
		Input:     DefaultInputPricePerMillion,
		Output:    DefaultOutputPricePerMillion,
		CacheRead: DefaultCacheReadPrice(DefaultInputPricePerMillion),
	}
	if got != want {
		t.Fatalf("DefaultRates() = %+v, want %+v", got, want)
	}
	if got.CacheRead != 25_000 {
		t.Fatalf("default cache read = %d, want 25000 (50%% of $0.05)", got.CacheRead)
	}
	if RatesFor(store.ModelPrice{InputPrice: 1, OutputPrice: 2}, false) != want {
		t.Fatal("an unconfigured lookup must ignore the zero row and use the defaults")
	}
}

func TestRatesForDerivesCacheReadFromInputWhenUnset(t *testing.T) {
	got := RatesFor(store.ModelPrice{InputPrice: 300_000, OutputPrice: 1_200_000}, true)
	want := Rates{Input: 300_000, Output: 1_200_000, CacheRead: 150_000}
	if got != want {
		t.Fatalf("derived rates = %+v, want %+v", got, want)
	}
}

func TestRatesForHonorsExplicitCacheRead(t *testing.T) {
	for _, explicit := range []int64{0, 1, 30_000, 300_000} {
		got := RatesFor(store.ModelPrice{InputPrice: 300_000, OutputPrice: 1_200_000, CacheReadPrice: int64Ptr(explicit)}, true)
		if got.CacheRead != explicit {
			t.Errorf("explicit cache_read_price %d resolved to %d", explicit, got.CacheRead)
		}
	}
}

func TestDefaultCacheReadPriceFloorsToWholeMicroUSD(t *testing.T) {
	if got := DefaultCacheReadPrice(1); got != 0 {
		t.Errorf("DefaultCacheReadPrice(1) = %d, want 0 (integer arithmetic floors)", got)
	}
	if got := DefaultCacheReadPrice(50_001); got != 25_000 {
		t.Errorf("DefaultCacheReadPrice(50001) = %d, want 25000", got)
	}
}

func TestCostWithMinimum(t *testing.T) {
	// All cases use the fallback rates ($0.05 input, $0.025 cache read, $0.20
	// output per 1M tokens).
	rates := DefaultRates()
	tests := []struct {
		name  string
		usage Usage
		want  int64
	}{
		{"1M output tokens", Usage{CompletionTokens: 1_000_000}, 200_000},
		{"1M input + 1M output", Usage{PromptTokens: 1_000_000, CompletionTokens: 1_000_000}, 250_000},
		{"only input tokens", Usage{PromptTokens: 1_000_000}, 50_000},
		{"half the prompt cached bills half at the cache-read rate", Usage{PromptTokens: 1_000_000, CachedTokens: 500_000}, 25_000 + 12_500},
		{"fully cached prompt", Usage{PromptTokens: 1_000_000, CachedTokens: 1_000_000}, 25_000},
		{"small request hits minimum", Usage{PromptTokens: 10, CompletionTokens: 10}, 100},
		{"zero tokens hits minimum", Usage{}, 100},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := rates.CostWithMinimum(tc.usage); got != tc.want {
				t.Errorf("CostWithMinimum(%+v) = %d, want %d", tc.usage, got, tc.want)
			}
		})
	}
}

func TestCostWithCustomRates(t *testing.T) {
	rates := Rates{Input: 15_000, Output: 70_000, CacheRead: 3_000}
	tests := []struct {
		name  string
		usage Usage
		want  int64
	}{
		{"custom rates, no cache", Usage{PromptTokens: 1_000_000, CompletionTokens: 1_000_000}, 85_000},
		{"custom rates, cached prefix", Usage{PromptTokens: 1_000_000, CachedTokens: 800_000, CompletionTokens: 1_000_000}, 200_000*15_000/1_000_000 + 800_000*3_000/1_000_000 + 70_000},
		{"tiny request floors to the minimum", Usage{PromptTokens: 10, CompletionTokens: 10}, 100},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := rates.CostWithMinimum(tc.usage); got != tc.want {
				t.Errorf("CostWithMinimum(%+v) = %d, want %d", tc.usage, got, tc.want)
			}
		})
	}
}

// A cache hit can only ever lower the bill relative to the same prompt served
// cold, and never below the rate the cache-read price implies.
func TestCachedTokensNeverIncreaseCost(t *testing.T) {
	rates := Rates{Input: 500_000, Output: 2_000_000, CacheRead: 100_000}
	cold := rates.Cost(Usage{PromptTokens: 40_000, CompletionTokens: 1_000})
	for cached := 0; cached <= 40_000; cached += 5_000 {
		warm := rates.Cost(Usage{PromptTokens: 40_000, CachedTokens: cached, CompletionTokens: 1_000})
		if warm > cold {
			t.Fatalf("cached=%d cost %d exceeds cold cost %d", cached, warm, cold)
		}
		want := int64(40_000-cached)*500_000/1_000_000 + int64(cached)*100_000/1_000_000 + 1_000*2_000_000/1_000_000
		if warm != want {
			t.Fatalf("cached=%d cost %d, want %d", cached, warm, want)
		}
	}
}

// Malformed usage must not produce a negative charge or bill more tokens than
// were in the prompt: cached tokens are clamped to the prompt and negative
// counts bill as zero.
func TestCostClampsMalformedUsage(t *testing.T) {
	rates := Rates{Input: 1_000_000, Output: 1_000_000, CacheRead: 0}
	// More cached than prompt tokens: the whole prompt is cached, nothing more.
	if got := rates.Cost(Usage{PromptTokens: 100, CachedTokens: 1_000, CompletionTokens: 100}); got != 100 {
		t.Errorf("over-reported cache cost = %d, want 100 (completion only)", got)
	}
	// Negative counts never turn into credits.
	if got := rates.Cost(Usage{PromptTokens: -100, CachedTokens: -5, CompletionTokens: -100}); got != 0 {
		t.Errorf("negative usage cost = %d, want 0", got)
	}
	if got := rates.CostWithMinimum(Usage{PromptTokens: -100}); got != MinimumCharge() {
		t.Errorf("negative usage with minimum = %d, want %d", got, MinimumCharge())
	}
}

func TestPlatformFee(t *testing.T) {
	tests := []struct {
		totalCost int64
		wantFee   int64
	}{
		// Default platform fee is 0% during the public alpha.
		{100_000, 0},
		{1_000_000, 0},
		{500_000, 0},
		{1_000, 0},
		{0, 0},
	}

	for _, tc := range tests {
		got := PlatformFee(tc.totalCost)
		if got != tc.wantFee {
			t.Errorf("PlatformFee(%d) = %d, want %d", tc.totalCost, got, tc.wantFee)
		}
	}
}

func TestProviderPayout(t *testing.T) {
	tests := []struct {
		totalCost  int64
		wantPayout int64
	}{
		// Providers keep 100% during the public alpha (0% default fee).
		{100_000, 100_000},
		{1_000_000, 1_000_000},
		{1_000, 1_000},
		{0, 0},
	}

	for _, tc := range tests {
		got := ProviderPayout(tc.totalCost)
		if got != tc.wantPayout {
			t.Errorf("ProviderPayout(%d) = %d, want %d", tc.totalCost, got, tc.wantPayout)
		}
	}
}

func TestPlatformFeeAndProviderPayoutSumToTotal(t *testing.T) {
	totals := []int64{1_000, 10_000, 100_000, 500_000, 1_000_000, 10_000_000}
	for _, total := range totals {
		fee := PlatformFee(total)
		payout := ProviderPayout(total)
		if fee+payout != total {
			t.Errorf("PlatformFee(%d) + ProviderPayout(%d) = %d + %d = %d, want %d",
				total, total, fee, payout, fee+payout, total)
		}
	}
}

func TestFormatPerMillionUSD(t *testing.T) {
	cases := map[int64]string{
		0:          "$0.0000",
		25_000:     "$0.0250",
		50_000:     "$0.0500",
		200_000:    "$0.2000",
		1_234_567:  "$1.2346",
		10_000_000: "$10.0000",
	}
	for in, want := range cases {
		if got := FormatPerMillionUSD(in); got != want {
			t.Errorf("FormatPerMillionUSD(%d) = %q, want %q", in, got, want)
		}
	}
}
