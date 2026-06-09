"use client";

import { useEffect, useState } from "react";
import {
  getSignalPerformance,
  getSignalPerformanceByRegime,
  backfillTradeRegimes,
} from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { HelpSection } from "@/components/HelpSection";
import { RegimeBadge } from "@/components/dashboard/RegimeBadge";
import {
  Loader2,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  CheckCircle2,
  AlertTriangle,
  Minus,
  Brain,
} from "lucide-react";
import { toast } from "sonner";

type SignalRow = {
  signal_type: string;
  weight_key: string;
  current_weight: number;
  n: number;
  wins: number;
  losses: number;
  win_rate: number;
  wilson_lower_80: number;
  avg_return_5d_pct: number;
  suggested_weight: number;
  delta: number;
  verdict: "TUNE_UP" | "TUNE_DOWN" | "KEEP" | "INSUFFICIENT_DATA";
};

type Performance = {
  lookback_days: number;
  total_closed_trades: number;
  min_sample_size: number;
  signals: SignalRow[];
};

const helpItems = [
  {
    question: "What is this page?",
    answer:
      "This is the diagnostic feedback loop for the Recommendation Engine. Every paper trade and shadow trade stores which signals fired. After each trade closes (5-day P&L), we credit/blame each signal that was present to track their real-world performance under different market regimes.",
  },
  {
    question: "How is 'win rate' calculated?",
    answer:
      "For each closed trade, we look at every signal that fired:\n  • Bullish signals 'win' if the stock went UP after 5 days\n  • Bearish signals 'win' if the stock went DOWN\n  • One trade with multiple signals contributes 1 observation to each signal type.\n\nThis multi-attribution metric helps identify which signals correlate with profitable outcomes.",
  },
  {
    question: "What's the Wilson lower bound?",
    answer:
      "A win rate of 4/5 (80%) is unreliable on a tiny sample. The Wilson score lower bound at 80% confidence gives a conservative estimate that scales down raw rates when observations are low, preventing statistical noise from distorting performance analysis.",
  },
  {
    question: "How are these statistics used by the Recommendation Engine?",
    answer:
      "Instead of using static weights or manual tuning, the engine is powered by an L1-regularized logistic regression probabilistic model. The model automatically trains on all historical trades, learns the regime-specific win probabilities, and outputs a calibrated success probability to drive position sizing. These stats are displayed here as a transparency tool for you to monitor signal behavior.",
  },
];

type RegimeStats = {
  n: number;
  wins: number;
  win_rate: number | null;
  avg_return_5d_pct: number | null;
};
type RegimeSignal = {
  signal_type: string;
  weight_key: string;
  current_weight: number;
  total_n: number;
  by_regime: Record<string, RegimeStats>;
  regime_spread: number | null;
  is_regime_dependent: boolean;
};
type RegimePerf = {
  lookback_days: number;
  regimes: string[];
  by_signal: RegimeSignal[];
  total_tagged_trades: number;
};

const REGIME_COLORS: Record<string, string> = {
  BULL: "text-green-700",
  BEAR: "text-red-700",
  SIDEWAYS: "text-amber-700",
  HIGH_VOL: "text-purple-700",
};

type RegimeSuggestion = {
  current: number;
  suggested: number;
  delta: number;
  n: number;
  wins: number;
  win_rate: number;
  wilson_lower_80: number;
  avg_return_5d_pct: number | null;
  verdict: string;
};
type RegimeSuggestions = {
  lookback_days: number;
  min_sample_per_regime: number;
  by_regime: Record<string, Record<string, RegimeSuggestion>>;
  summary: Record<string, { override_count: number }>;
};

export default function SignalsPage() {
  const [data, setData] = useState<Performance | null>(null);
  const [regimeData, setRegimeData] = useState<RegimePerf | null>(null);
  const [loading, setLoading] = useState(true);
  const [windowDays, setWindowDays] = useState(90);

  const load = async () => {
    setLoading(true);
    try {
      const [perf, regime]: any[] = await Promise.all([
        getSignalPerformance(windowDays),
        getSignalPerformanceByRegime(Math.max(windowDays, 180)),
      ]);
      setData(perf);
      setRegimeData(regime);
    } catch (e: any) {
      toast.error(e.message || "Failed to load signal performance");
    }
    setLoading(false);
  };

  const backfillRegimes = async () => {
    try {
      const r: any = await backfillTradeRegimes();
      toast.success(`Tagged ${r.trades_updated} trade(s) with regime`);
      await load();
    } catch (e: any) {
      toast.error(e.message || "Backfill failed");
    }
  };

  useEffect(() => {
    load();
  }, [windowDays]);

  return (
    <div className="p-6 space-y-5 max-w-6xl">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Brain className="h-6 w-6 text-purple-600" />
            Signal Performance
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Real win rate of each recommender signal — auto-tune the engine from your trade outcomes.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <select
            className="border rounded-md px-2 py-1.5 text-sm"
            value={windowDays}
            onChange={(e) => setWindowDays(parseInt(e.target.value, 10))}
          >
            <option value={30}>Last 30 days</option>
            <option value={60}>Last 60 days</option>
            <option value={90}>Last 90 days</option>
            <option value={180}>Last 180 days</option>
            <option value={365}>Last 365 days</option>
          </select>
          <Button variant="outline" size="sm" onClick={load} disabled={loading}>
            {loading ? <Loader2 className="h-3 w-3 animate-spin mr-1" /> : <RefreshCw className="h-3 w-3 mr-1" />}
            Refresh
          </Button>
        </div>
      </div>

      {/* L1 Model Info Callout */}
      <Card className="border-purple-200 bg-purple-50/10 backdrop-blur-sm shadow-sm">
        <CardContent className="p-5">
          <div className="flex items-start gap-4">
            <div className="p-2.5 rounded-xl bg-purple-100/80 text-purple-700 flex-shrink-0 shadow-inner">
              <Brain className="h-6 w-6 text-purple-700 animate-pulse" />
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="font-bold text-base text-purple-900 mb-1">Automated Probabilistic Modeling Active</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                The legacy manual weight-tuning system has been retired. The recommendation engine is now powered by a 
                trained <strong className="text-purple-800 font-semibold">L1-regularized logistic regression probabilistic model</strong>.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-4">
                <div className="bg-white/40 p-3 rounded-lg border border-purple-100/50">
                  <div className="text-xs font-semibold text-purple-900 uppercase tracking-wider mb-1">How it works</div>
                  <p className="text-xs text-muted-foreground leading-normal">
                    The engine extracts a 28-dimensional binary feature vector (17 signals, 4 regimes, 6 interaction terms, and intercept) and fits weights dynamically.
                  </p>
                </div>
                <div className="bg-white/40 p-3 rounded-lg border border-purple-100/50">
                  <div className="text-xs font-semibold text-purple-900 uppercase tracking-wider mb-1">Model calibration</div>
                  <p className="text-xs text-muted-foreground leading-normal">
                    Fitted coefficients are updated via background retraining. Predictions are promoted only if the validation metrics are verified (AUC &gt; 0.55, Brier &lt; 0.20).
                  </p>
                </div>
                <div className="bg-white/40 p-3 rounded-lg border border-purple-100/50">
                  <div className="text-xs font-semibold text-purple-900 uppercase tracking-wider mb-1">This View</div>
                  <p className="text-xs text-muted-foreground leading-normal">
                    The statistics below show real-time performance diagnostics. They are informational to help you understand which signals work best in different regimes.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Summary Diagnostics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <Card className="md:col-span-1">
          <CardContent className="p-5 space-y-4">
            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wider font-semibold">Closed Trades (Window)</p>
              <p className="text-3xl font-bold mt-1 text-purple-950">{data?.total_closed_trades ?? "—"}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wider font-semibold">Signals Tracked</p>
              <p className="text-3xl font-bold mt-1 text-purple-950">{data?.signals.length ?? 0}</p>
            </div>
          </CardContent>
        </Card>
        <div className="md:col-span-2">
          <RegimeBadge />
        </div>
      </div>

      {/* Per-signal table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Per-Signal Breakdown</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted/50 border-y">
                <tr className="text-left">
                  <th className="px-4 py-2 font-medium">Signal</th>
                  <th className="px-2 py-2 font-medium text-right">Observations (N)</th>
                  <th className="px-2 py-2 font-medium text-right">Raw Win Rate</th>
                  <th className="px-2 py-2 font-medium text-right" title="Wilson lower bound at 80% confidence">Honest WR (80% CI)</th>
                  <th className="px-2 py-2 font-medium text-right">Avg 5d Return</th>
                  <th className="px-4 py-2 font-medium text-right">Base Weight</th>
                </tr>
              </thead>
              <tbody>
                {loading && !data && (
                  <tr><td colSpan={6} className="text-center py-8 text-muted-foreground">
                    <Loader2 className="h-5 w-5 animate-spin inline" />
                  </td></tr>
                )}
                {data?.signals.map((s) => {
                  return (
                    <tr key={s.weight_key} className="border-b hover:bg-muted/30">
                      <td className="px-4 py-2 font-medium">
                        {s.signal_type}
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums text-muted-foreground">{s.n}</td>
                      <td className="px-2 py-2 text-right tabular-nums">
                        {s.n > 0 ? `${(s.win_rate * 100).toFixed(0)}%` : "—"}
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums text-muted-foreground">
                        {s.n > 0 ? `${(s.wilson_lower_80 * 100).toFixed(0)}%` : "—"}
                      </td>
                      <td className={`px-2 py-2 text-right tabular-nums ${s.avg_return_5d_pct > 0 ? "text-green-700" : s.avg_return_5d_pct < 0 ? "text-red-700" : ""}`}>
                        {s.n > 0 ? `${s.avg_return_5d_pct > 0 ? "+" : ""}${s.avg_return_5d_pct.toFixed(2)}%` : "—"}
                      </td>
                      <td className="px-4 py-2 text-right tabular-nums font-mono text-muted-foreground">
                        {s.current_weight > 0 ? "+" : ""}{s.current_weight.toFixed(1)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Regime-conditional how-to callout */}
      <Card className="border-amber-200 bg-amber-50/30">
        <CardContent className="p-4">
          <div className="flex items-start gap-3">
            <div className="p-1.5 rounded-lg bg-amber-100 flex-shrink-0">
              <span className="text-amber-700 text-lg leading-none">⚡</span>
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold text-sm mb-2">How to use the regime breakdown</h3>
              <ol className="text-sm text-muted-foreground space-y-1.5 list-decimal list-inside">
                <li><span className="text-foreground font-medium">Watch today's regime:</span> The badge on the Dashboard shows the current regime. Most signals behave differently in HIGH_VOL vs BULL.</li>
                <li><span className="text-foreground font-medium">Spot regime-dependent signals:</span> Any row flagged <span className="text-amber-700 font-medium">⚡ regime-dependent</span> works in some regimes but fails in others. The blanket weight in the main table is a misleading average.</li>
                <li><span className="text-foreground font-medium">Read the spread:</span> A spread of 50% means the signal's win rate ranges by 50 percentage points across regimes — huge signal. Spread &lt;20% means it works consistently everywhere.</li>
                <li><span className="text-foreground font-medium">Filter manually for now:</span> If "Volume Spike Bullish" wins 75% in BULL but 25% in BEAR, only act on it during BULL regimes (check the Dashboard badge before trading).</li>
                <li><span className="text-foreground font-medium">Need data:</span> Spread is only computed when n≥5 in at least 2 regimes. Most users will start in one regime — accumulate trades, wait for the market to shift, then come back.</li>
                <li><span className="text-foreground font-medium">Future:</span> When enough data exists, the recommender will auto-apply regime-conditional weights (skips ⚡ signals when the wrong regime is active).</li>
              </ol>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Regime-conditional signal performance */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="text-base">Regime-Conditional Win Rates</CardTitle>
            <p className="text-xs text-muted-foreground mt-1">
              Same signals, split by what the market regime was when the trade opened.
              Regime-dependent signals are flagged ⚡ — these need different weights per regime.
            </p>
          </div>
          <Button onClick={backfillRegimes} size="sm" variant="outline">
            Backfill Regimes
          </Button>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted/50 border-y">
                <tr className="text-left">
                  <th className="px-4 py-2 font-medium">Signal</th>
                  {(regimeData?.regimes || ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOL"]).map((r) => (
                    <th key={r} className={`px-2 py-2 font-medium text-right ${REGIME_COLORS[r]}`}>
                      {r}
                    </th>
                  ))}
                  <th className="px-2 py-2 font-medium text-right" title="Max - min win rate across regimes with n>=5">
                    Spread
                  </th>
                  <th className="px-4 py-2 font-medium text-center">Note</th>
                </tr>
              </thead>
              <tbody>
                {(!regimeData || regimeData.by_signal.length === 0) && (
                  <tr>
                    <td colSpan={7} className="text-center py-8 text-muted-foreground text-xs">
                      {regimeData?.total_tagged_trades === 0
                        ? "No trades have regime tags yet — click 'Backfill Regimes' above."
                        : "No closed trades in window."}
                    </td>
                  </tr>
                )}
                {regimeData?.by_signal.map((s) => (
                  <tr key={s.weight_key} className="border-b hover:bg-muted/30">
                    <td className="px-4 py-2 font-medium">
                      {s.signal_type}
                      {s.is_regime_dependent && (
                        <Badge variant="outline" className="ml-2 text-xs bg-amber-50 text-amber-700 border-amber-200">
                          ⚡ regime-dependent
                        </Badge>
                      )}
                    </td>
                    {regimeData.regimes.map((regime) => {
                      const stats = s.by_regime[regime];
                      if (!stats || stats.n === 0) {
                        return (
                          <td key={regime} className="px-2 py-2 text-right text-muted-foreground text-xs">
                            —
                          </td>
                        );
                      }
                      const wr = stats.win_rate ?? 0;
                      return (
                        <td key={regime} className="px-2 py-2 text-right tabular-nums">
                          <span className={`${wr >= 0.55 ? "text-green-700 font-semibold" : wr <= 0.40 ? "text-red-700 font-semibold" : ""}`}>
                            {(wr * 100).toFixed(0)}%
                          </span>
                          <span className="text-xs text-muted-foreground ml-1">
                            (n={stats.n})
                          </span>
                        </td>
                      );
                    })}
                    <td className="px-2 py-2 text-right tabular-nums">
                      {s.regime_spread != null ? (
                        <span className={s.is_regime_dependent ? "text-amber-700 font-semibold" : "text-muted-foreground"}>
                          {(s.regime_spread * 100).toFixed(0)}%
                        </span>
                      ) : (
                        <span className="text-muted-foreground text-xs">—</span>
                      )}
                    </td>
                    <td className="px-4 py-2 text-xs text-muted-foreground">
                      {s.is_regime_dependent
                        ? "Use only in best-performing regime"
                        : s.regime_spread != null
                        ? "Works across regimes"
                        : "Need n≥5 in 2+ regimes"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {regimeData && regimeData.total_tagged_trades > 0 && (
              <div className="px-4 py-2 text-xs text-muted-foreground border-t">
                {regimeData.total_tagged_trades} tagged trades in last {regimeData.lookback_days} days.
                Need n≥5 in at least 2 regimes for spread calculation.
              </div>
            )}
          </div>
        </CardContent>
      </Card>



      <HelpSection items={helpItems} />
    </div>
  );
}
