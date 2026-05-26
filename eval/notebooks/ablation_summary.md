# Ablation summary

Ten retrieval configurations evaluated on the 42-question gold set.
Cells show **pass-rate / mean Recall@k / mean MRR**. Bold marks the
strongest configuration (dense k=30 with aggressive xref expansion).

## Full results matrix

| # | Config | Factual | Named ent. | Narrative | Thematic | Interp. | Overall |
|---|---|---|---|---|---|---|---|
| 1 | Dense k=10 (baseline) | 5/10 / 0.350 / 0.278 | 5/9 / 0.317 / 0.456 | 0/8 / 0.076 / 0.304 | 0/10 / 0.071 / 0.210 | 4/5 / 0.354 / 0.607 | 14/42 / 0.225 / 0.344 |
| 2 | Dense k=20 | 7/10 / 0.408 / 0.291 | 5/9 / 0.394 / 0.456 | 1/8 / 0.132 / 0.304 | 2/10 / 0.154 / 0.228 | 4/5 / 0.394 / 0.607 | 19/42 / 0.291 / 0.351 |
| 3 | Dense k=20, no MHC | 7/10 / 0.408 / 0.314 | 5/9 / 0.394 / 0.458 | 1/8 / 0.132 / 0.310 | 2/10 / 0.154 / 0.282 | 0/5 / 0.204 / 0.467 | 15/42 / 0.268 / 0.355 |
| 4 | Dense k=20, no xref | 7/10 / 0.408 / 0.291 | 5/9 / 0.394 / 0.456 | 1/8 / 0.132 / 0.304 | 2/10 / 0.154 / 0.228 | 4/5 / 0.394 / 0.607 | 19/42 / 0.291 / 0.351 |
| 5 | Dense k=20, xref 3/10 | 7/10 / 0.458 / 0.291 | 5/9 / 0.394 / 0.456 | 1/8 / 0.132 / 0.304 | 2/10 / 0.174 / 0.228 | 4/5 / 0.444 / 0.607 | 19/42 / 0.313 / 0.351 |
| 6 | **Dense k=30, xref 10/15** | **8/10 / 0.542 / 0.294** | **6/9 / 0.506 / 0.458** | **1/8 / 0.150 / 0.307** | **2/10 / 0.225 / 0.228** | **4/5 / 0.517 / 0.607** | **21/42 / 0.381 / 0.353** |
| 7 | Hybrid k=10, α=.5, mult=3 | 4/10 / 0.283 / 0.128 | 5/9 / 0.294 / 0.294 | 0/8 / 0.090 / 0.099 | 1/10 / 0.069 / 0.244 | 4/5 / 0.376 / 0.767 | 14/42 / 0.209 / 0.262 |
| 8 | Hybrid k=10, α=.5, mult=10 | 3/10 / 0.183 / 0.084 | 4/9 / 0.202 / 0.260 | 0/8 / 0.069 / 0.049 | 1/10 / 0.069 / 0.224 | 4/5 / 0.347 / 0.767 | 12/42 / 0.158 / 0.230 |
| 9 | Hybrid k=10, α=.75, mult=3 | 4/10 / 0.317 / 0.150 | 4/9 / 0.239 / 0.269 | 0/8 / 0.090 / 0.109 | 1/10 / 0.099 / 0.228 | 4/5 / 0.347 / 0.767 | 13/42 / 0.209 / 0.260 |
| 10 | Hybrid k=20, α=.5, mult=3 | 5/10 / 0.350 / 0.121 | 6/9 / 0.478 / 0.285 | 0/8 / 0.111 / 0.067 | 1/10 / 0.099 / 0.250 | 4/5 / 0.466 / 0.767 | 16/42 / 0.286 / 0.253 |

## What each row tested

| # | Tested |
|---|---|
| 1 | First runnable baseline. Top_k=10, xrefs with conservative caps, MHC on, dense only. |
| 2 | Effect of wider retrieval (top_k=20). |
| 3 | Effect of disabling commentary. Tests whether MHC is competing with verse gold. |
| 4 | Effect of disabling cross-reference expansion entirely (MHC on) |
| 5 | Effect of aggressive xref at k=20 (max_per_seed=3, max_extra=10). |
| 6 | Effect of going further on both k and xref. **The production config.** |
| 7 | Hybrid retrieval at k=10, equal-weight RRF, default fetch. |
| 8 | Hybrid k=10, same weights, wider fetch window (mult=10). |
| 9 | Hybrid k=10, downweighted BM25 (dense weight 0.75). |
| 10 | Hybrid at k=20 to match dense at matched retrieval depth. |

## Key comparisons

**Dense k=10 → k=20 (rows 1, 2):** +5 questions, +29% recall, +2% MRR.
Wider retrieval helps everywhere, most in multi-gold categories.

**Dense MHC on/off at k=20 (rows 2, 3):** Interpretive drops 4/5 → 0/5,
as expected. Narrative/thematic recall identical to three decimal
places. MHC doesn't compete with verse gold.

**Xref expansion at k=20 (rows 2, 4, 5):** Pass-rate and MRR
unchanged across no/conservative/aggressive xref. Recall lifts
+0.022 at aggressive setting, mostly in factual. This looked like
a dead-end finding.

**Xref expansion at k=30 (rows 6):** Going to k=30 with very
aggressive xref (10/15) produces +2 pass-rate, +31% recall.
Recall lifts substantially in *every* category, even the ones
that don't flip pass-rate thresholds. The interaction between
retrieval depth and xref aggressiveness is the most important
finding in the project.

**Dense vs hybrid at k=10 (rows 1, 7):** Pass-rate matches (14/42)
but MRR drops 0.344 → 0.262. BM25 contribution dragged down
ranking quality without helping pass-rate.

**Dense vs hybrid at k=20 (rows 2, 10):** Pass-rate 19/42 → 16/42.
MRR 0.351 → 0.253. The negative result holds at higher k. One
positive: named entity rises from 5/9 to 6/9 under hybrid.

**Hybrid α=.5 vs α=.75 at k=10 (rows 7, 9):** Reducing BM25's vote
recovers slightly but doesn't reach dense baseline. Monotonic
pattern: more BM25 influence, worse overall.

## Sources

Results files in `eval/results/`:
- Row 1: `run_20260526_170315.json`
- Row 2: `run_20260526_170431.json`
- Row 3: `run_20260526_170522.json`
- Row 4: `run_20260526_170727.json`
- Row 5: `run_20260526_170842.json`
- Row 6: `run_20260526_171032.json` (production config)
- Row 7: `run_20260526_173019.json`
- Row 8: `run_20260526_173047.json`
- Row 9: `run_20260526_173108.json`
- Row 10: `rrun_20260526_173130.json`


Each results file contains the full config dict and per-question
retrieval traces, so any row can be reproduced from the runner.
