# Tennis Analytics — Point Win %, Clutch, and Match Outcomes

DSAI Project Seminar, Universität des Saarlandes.

This project replicates the core result of the reference paper — that a player's
career **point win percentage (pw)** almost fully explains their **match win
percentage (mwp)** — and then extends it in four different directions :
non-linear functional forms, a non-linear treatment of the "clutch"
break-point metric, a parsimony test, and an original test of whether clutch
performance persists across a career.

> **Reference paper:** Revisiting Non-i.i.d. Effects in Tennis by the
Reference of Clutch-Performance by Pascal Bauer*, Luis K. H. Holzhauer, and Jan Bauer

## Key questions

1. **Replication.** Does the linear pw → mwp relationship (Table 4, models R1–R4)
   reproduce on the full ATP (1990–) and WTA (2003–) samples?
2. **Functional form.** Does allowing the curve to bend (sigmoid / cubic / GAM)
   beat the straight line, and where does the linear model systematically fail?
3. **Clutch proxy.** Once non-linearity is allowed, does adding the break-point
   ratio `bpw` meaningfully improve fit?
4. **Parsimony.** Does splitting `pw` into serve/return (`spw`, `rpw`) earn its
   extra parameter, or does the simpler `pw + bpw` model win?
5. **Persistence (original).** Does a player's early-career clutch predict their
   late-career clutch, or is "clutch" mostly noise?

## Repository contents

```
.
├── tennis_analysis.py     # full pipeline: Phases 1–6
├── clutch_style_check.py  # does clutch just track playing strength? (robustness)
├── atp_panel.csv          # derived match-level panel (used by Phase 5)
├── wta_panel.csv          # derived match-level panel (used by Phase 5)
├── tennis_atp/            # raw ATP match CSVs (Jeff Sackmann)
├── tennis_wta/            # raw WTA match CSVs (Jeff Sackmann)
├── outputs/               # all generated figures (.png) and tables (.csv)
├── requirements.txt
├── LICENSE
└── README.md
```



## Method in brief

One script, six labelled phases:

| Phase | What it does |
|------:|--------------|
| 1 | Load, filter (Table 3 rules), build per-player career stats, replicate Table 4 (R1–R4) and Figure 1 |
| 2 | Non-linear extension of R1 (pw only): linear vs sigmoid vs cubic vs GAM, plus a chronological hold-out |
| 3 | Clutch proxy: non-linear R3/R4 across all families, additive vs `spw × rpw` interaction |
| 4 | Parsimony check: `pw + bpw` vs `spw + rpw + bpw` by AIC |
| 5 | Clutch persistence: split each career in half, Monte-Carlo simulate an i.i.d. baseline, regress late-career clutch on early-career clutch |
| 6 | Player-level 5-fold cross-validation for Phases 2/3/4 (guards against overfitting, especially GAM) |

`bpw` = break points won on return ÷ break points lost on serve — the paper's
clutch metric. "Clutch score" in Phase 5 = observed mwp − simulated mwp.

## Headline results

**Replication (Table 4, R²)** — reproduces the paper closely; small gaps come
from the larger, more recent sample.

| Model | ATP (this repo) | ATP (paper) | WTA (this repo) | WTA (paper) |
|-------|:---------------:|:-----------:|:---------------:|:-----------:|
| R1 `pw`            | 93.6% | 94.0% | 95.6% | 95.8% |
| R2 `pw+bpw`        | 94.2% | 94.5% | 95.6% | 95.7% |
| R3 `spw+rpw`       | 92.4% | 93.0% | 95.4% | 95.5% |
| R4 `spw+rpw+bpw`   | 93.3% | 94.1% | 95.4% | 95.6% |

**Functional form (Phase 2)** — non-linear fits beat the line only slightly, so
the relationship is nearly linear over the observed range. GAM is best both times.

| | Linear | Sigmoid | Poly-3 | GAM |
|-|:------:|:-------:|:------:|:---:|
| ATP R² | 93.59% | 93.83% | 93.83% | **93.99%** |
| WTA R² | 95.55% | 95.85% | 95.84% | **95.99%** |

**Parsimony (Phase 4)** — in **all 16** comparisons, the simpler `pw + bpw` beats
`spw + rpw + bpw` on AIC. Splitting `pw` into serve/return does not earn its extra
parameter — a non-linear strengthening of the paper's own R2-beats-R4 finding.

**Persistence (Phase 5, original)** — early-career clutch carries some signal,
more so for men:

| Tour | n | β (early → late) | p | R² |
|------|--:|:----------------:|:-:|:--:|
| ATP  | 517 | 0.394 | <0.001 | 0.160 |
| WTA  | 270 | 0.221 | <0.001 | 0.046 |

Significant but shallow slope → clutch is a **real but small and largely
non-repeatable** effect.

**Robustness — does "clutch" just re-measure playing strength?** The clutch score
correlates with career point win percentage (*r* ≈ 0.38 on both tours), so early and
late clutch share a stable skill component. Controlling for career serve/return rates
(`clutch_style_check.py`):

| Tour | β raw | β controlling for strength | verdict |
|------|:-----:|:--------------------------:|---------|
| ATP  | 0.394 | **0.315** (*p* < 0.001)    | persists — 20% of the raw slope was strength |
| WTA  | 0.221 | **0.075** (*p* = 0.23)     | **not significant** — 66% was strength |

So the men's persistence survives, the women's does not. Reproduce in seconds with
`python3 clutch_style_check.py` (reads the committed Phase 5 output — no Monte-Carlo
rerun needed). Written to `outputs/clutch_style_control.csv`.

## Figures (in `outputs/`)

| File | Content |
|------|---------|
| `tennis_figure1.png` | Replication of Figure 1: pw → mwp with top-10 players labelled |
| `tennis_figure_main.png` | Linear baseline vs. sigmoid fit, with tail divergence zones |
| `tennis_figure_all_models.png` | All four families side by side |
| `tennis_figure_residuals.png` | Where the linear model systematically errs |
| `tennis_figure_r2_comparison.png` | R² bar chart across families |
| `tennis_figure_clutch_extension.png` | Clutch-proxy R² across families (Phase 3) |
| `persistence_test.png` | Early- vs late-career clutch (Phase 5) |

## Data

Raw match data is from Jeff Sackmann's
[tennis_atp](https://github.com/JeffSackmann/tennis_atp) and
[tennis_wta](https://github.com/JeffSackmann/tennis_wta) repositories
included here in `tennis_atp/` and `tennis_wta/`.


## Author

Prabhjot Singh Rawal  and Akshat Tinjani — DSAI Project Seminar, Universität des Saarlandes.
