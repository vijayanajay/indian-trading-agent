"use client";

import { useEffect, useState } from "react";
import { getRecommendations, getWatchlist, openPaperTrade } from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Loader2, TrendingUp, TrendingDown, Sparkles, RefreshCw, ArrowRight, Bell, Star, FlaskConical } from "lucide-react";
import Link from "next/link";
import { toast } from "sonner";

function HonestAssessmentBadge({ assessment }: { assessment: any }) {
  if (!assessment) return null;
  const { tier, display_message } = assessment;
  let bg = "bg-gray-50 border-gray-200 text-gray-700";
  if (tier === "EXPLORATORY") {
    bg = "bg-amber-50 border-amber-200 text-amber-700";
  } else if (tier === "EMERGING") {
    bg = "bg-blue-50 border-blue-200 text-blue-700";
  } else if (tier === "EMPIRICAL") {
    bg = "bg-indigo-50 border-indigo-200 text-indigo-700";
  } else if (tier === "CALIBRATED") {
    bg = "bg-green-100 border-green-300 text-green-800 font-semibold";
  }

  return (
    <Badge variant="outline" className={`border ${bg} text-[10px] sm:text-xs py-0.5 px-2 flex items-center gap-1`}>
      {tier === "EXPLORATORY" && "⚠️"}
      {tier === "EMERGING" && "📊"}
      {tier === "EMPIRICAL" && "📈"}
      {tier === "CALIBRATED" && "🎯"}
      {display_message}
    </Badge>
  );
}

export function TodayPicks({ universe = "nifty100" }: { universe?: string }) {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<any>(null);
  const [lastRun, setLastRun] = useState<Date | null>(null);
  const [watchlistTickers, setWatchlistTickers] = useState<string[]>([]);
  const [confirmations, setConfirmations] = useState<Record<string, boolean>>({});

  const load = async () => {
    setLoading(true);
    try {
      const [result, watchlist]: any[] = await Promise.all([
        getRecommendations(universe, 2),
        getWatchlist().catch(() => []),
      ]);
      setData(result);
      setLastRun(new Date());
      setConfirmations({});

      const wlTickers: string[] = (watchlist || []).map((w: any) => w.ticker);
      setWatchlistTickers(wlTickers);

      // Alert for watchlist matches
      const allPicks = [...(result.strong_buys || []), ...(result.buys || [])];
      const matches = allPicks.filter((p: any) => wlTickers.includes(p.ticker));
      if (matches.length > 0) {
        toast.success(`${matches.length} watchlist stock${matches.length > 1 ? "s" : ""} in Top Picks: ${matches.map((m: any) => m.ticker).join(", ")}`, {
          duration: 6000,
          icon: <Bell className="h-4 w-4" />,
        });
      }
    } catch {}
    setLoading(false);
  };

  useEffect(() => {
    load();
  }, [universe]);

  const topPicks = data
    ? [...(data.strong_buys || []), ...(data.buys || [])].slice(0, 5)
    : [];
  const topSells = data
    ? [...(data.strong_sells || []), ...(data.sells || [])].slice(0, 3)
    : [];

  return (
    <Card className="border-yellow-200">
      <CardContent className="p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-yellow-100">
              <Sparkles className="h-5 w-5 text-yellow-600" />
            </div>
            <div>
              <h2 className="font-semibold text-lg">Today's Top Picks</h2>
              <p className="text-xs text-muted-foreground">
                {lastRun ? `Updated ${lastRun.toLocaleTimeString()}` : "Loading..."} · {universe.toUpperCase()}
                {data?.active_regime && (
                  <>
                    {" · "}
                    <span title={`${data.active_regime} regime active. Visit Signal Performance to view diagnostics.`}>
                      <span className="font-medium">{data.active_regime}</span>
                    </span>
                  </>
                )}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={load} disabled={loading}>
              {loading ? <Loader2 className="h-3 w-3 animate-spin mr-1" /> : <RefreshCw className="h-3 w-3 mr-1" />}
              Refresh
            </Button>
            <Link href="/recommendations">
              <Button size="sm" variant="ghost">
                View all <ArrowRight className="h-3 w-3 ml-1" />
              </Button>
            </Link>
          </div>
        </div>

        {loading && !data && (
          <div className="py-8 text-center">
            <Loader2 className="h-6 w-6 animate-spin mx-auto text-muted-foreground" />
            <p className="text-xs text-muted-foreground mt-2">Scanning {universe.toUpperCase()} stocks for opportunities...</p>
          </div>
        )}

        {data && topPicks.length === 0 && topSells.length === 0 && (
          <div className="py-6 text-center">
            <p className="text-sm text-muted-foreground">No strong signals in the market right now.</p>
            <p className="text-xs text-muted-foreground mt-1">Try again later or check the Market Scan.</p>
          </div>
        )}

        {data && topPicks.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-medium text-green-700 mb-2">BUY OPPORTUNITIES</p>
            {topPicks.map((pick: any, i: number) => {
              const isStrongBuy = pick.direction === "STRONG BUY";
              const isConfirmed = !isStrongBuy || !!confirmations[pick.ticker];
              return (
                <div
                  key={pick.ticker}
                  className="flex flex-col p-3 rounded-lg bg-green-50/50 border border-green-100 hover:bg-green-50 transition-colors space-y-2"
                >
                  <div className="flex items-center justify-between w-full">
                    <div className="flex items-center gap-3 min-w-0">
                      <span className="text-sm font-semibold text-muted-foreground w-5">{i + 1}.</span>
                      <TrendingUp className="h-4 w-4 text-green-600 flex-shrink-0" />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="font-semibold">{pick.ticker}</span>
                          {watchlistTickers.includes(pick.ticker) && (
                            <Badge variant="outline" className="bg-yellow-50 text-yellow-700 border-yellow-300 text-xs">
                              <Star className="h-2.5 w-2.5 mr-0.5 fill-yellow-600" />
                              Watchlist
                            </Badge>
                          )}
                          <span className="text-sm text-muted-foreground">Rs.{pick.price}</span>
                          <Badge variant="outline" className="bg-green-100 text-green-800 border-green-300 text-xs">
                            {pick.direction}
                          </Badge>
                          {pick.correlation_breach && (
                            <Badge variant="outline" className="bg-yellow-50 text-yellow-700 border-yellow-300 text-xs cursor-help" title={pick.correlation_warning}>
                              🔗 Cluster Risk
                            </Badge>
                          )}
                          <HonestAssessmentBadge assessment={pick.honest_assessment} />
                        </div>
                        <p className="text-xs text-muted-foreground truncate">
                          {pick.bullish_signal_count} signals · RSI {pick.rsi || "—"}
                        </p>
                      </div>
                    </div>
                    <div className="flex gap-1">
                      <Button
                        size="sm"
                        variant="ghost"
                        title={isStrongBuy && !isConfirmed ? "Please confirm stop-loss placement first" : "Open paper trade (simulated)"}
                        disabled={isStrongBuy && !isConfirmed}
                        onClick={async () => {
                          try {
                            await openPaperTrade({
                              ticker: pick.ticker,
                              source: "recommendation",
                              strategy: "Recommendation Engine (combined signals)",
                              signal: pick.direction,
                              score: pick.score,
                              confidence: pick.confidence,
                              success_probability: pick.honest_assessment?.probability,
                              triggered_signals: pick.signals,
                              position_size_pct: pick.suggested_position_size_pct,
                              stop_loss_price: pick.suggested_stop_loss,
                              risk_reward_ratio: pick.risk_reward_ratio,
                            } as any);
                            toast.success(`${pick.ticker} tracked at Rs.${pick.price}`, {
                              description: "Paper trade opened. Check P&L at 1/3/5/10 days on the Simulation page.",
                              duration: 6000,
                              action: {
                                label: "View Simulation",
                                onClick: () => { window.location.href = "/simulation"; },
                              },
                            });
                          } catch (e: any) {
                            toast.error(e.message || "Failed to track");
                          }
                        }}
                      >
                        <FlaskConical className="h-3 w-3 mr-1" /> Track
                      </Button>
                      <Link href={`/analysis?ticker=${pick.ticker}`}>
                        <Button size="sm" variant="outline">
                          Analyze <ArrowRight className="h-3 w-3 ml-1" />
                        </Button>
                      </Link>
                    </div>
                  </div>

                  {isStrongBuy && (
                    <div className="p-3 bg-white border border-green-200 rounded-lg text-xs space-y-2 w-full shadow-sm">
                      <div className="font-semibold text-green-800 flex items-center gap-1 text-[11px] sm:text-xs">
                        📋 Trade Plan (Risk-Defined Entry)
                      </div>
                      <div className="grid grid-cols-4 gap-2 text-center bg-gray-50/50 p-2 rounded border border-gray-100">
                        <div>
                          <div className="text-[10px] text-muted-foreground uppercase font-medium">Entry</div>
                          <div className="font-mono font-semibold text-gray-900">Rs.{pick.price}</div>
                        </div>
                        <div>
                          <div className="text-[10px] text-muted-foreground uppercase font-medium">Stop Loss</div>
                          <div className="font-mono font-semibold text-red-600">Rs.{pick.suggested_stop_loss || "—"}</div>
                        </div>
                        <div>
                          <div className="text-[10px] text-muted-foreground uppercase font-medium">Target</div>
                          <div className="font-mono font-semibold text-green-600">Rs.{pick.target_price || "—"}</div>
                        </div>
                        <div>
                          <div className="text-[10px] text-muted-foreground uppercase font-medium">R:R Ratio</div>
                          <div className="font-mono font-semibold text-purple-600">{pick.risk_reward_ratio ? `${pick.risk_reward_ratio}:1` : "—"}</div>
                        </div>
                      </div>
                      {pick.invalidation_reason && (
                        <p className="text-[10px] text-muted-foreground italic leading-normal">
                          Reason: {pick.invalidation_reason}
                        </p>
                      )}
                      <div className="flex items-center gap-2 mt-1.5 pt-1.5 border-t border-gray-100">
                        <input
                          type="checkbox"
                          id={`confirm-${pick.ticker}`}
                          checked={!!confirmations[pick.ticker]}
                          onChange={(e) => setConfirmations({ ...confirmations, [pick.ticker]: e.target.checked })}
                          className="h-3.5 w-3.5 rounded border-gray-300 text-green-600 focus:ring-green-500 cursor-pointer"
                        />
                        <label
                          htmlFor={`confirm-${pick.ticker}`}
                          className="text-[10px] font-medium text-gray-700 cursor-pointer select-none"
                        >
                          I confirm stop-loss is placed before entry.
                        </label>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {data && topSells.length > 0 && (
          <div className="space-y-2 mt-4">
            <p className="text-xs font-medium text-red-700 mb-2">AVOID / EXIT</p>
            {topSells.map((pick: any) => (
              <div
                key={pick.ticker}
                className="flex items-center justify-between p-3 rounded-lg bg-red-50/50 border border-red-100"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <TrendingDown className="h-4 w-4 text-red-600 flex-shrink-0" />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-semibold">{pick.ticker}</span>
                      <span className="text-sm text-muted-foreground">Rs.{pick.price}</span>
                      <Badge variant="outline" className="bg-red-100 text-red-800 border-red-300 text-xs">
                        {pick.direction}
                      </Badge>
                    </div>
                  </div>
                </div>
                <Link href={`/analysis?ticker=${pick.ticker}`}>
                  <Button size="sm" variant="outline">
                    View
                  </Button>
                </Link>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
