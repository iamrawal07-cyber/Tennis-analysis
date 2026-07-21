# ============================================================
#  Does the clutch score just track how good the player is?
#
#  The clutch score (observed mwp - i.i.d.-simulated mwp, Phase 5)
#  should be orthogonal to overall skill: the simulator already
#  conditions on each match's own serve/return rates. It is not --
#  it correlates with career point-win percentage. Because early- and
#  late-career clutch therefore share a stable skill component, the
#  raw early->late persistence slope partly measures "being good
#  persists" rather than "clutch persists".
#
#  This script quantifies that and re-estimates the slope controlling
#  for career serve/return rates (and best-of-5 share, where it varies
#  -- the WTA plays best-of-3 only, so that control is dropped there).
#
#  Backs the limitation sentence in the report. Runs in seconds:
#  it reads the Phase 5 output, and does NOT re-run the Monte-Carlo.
#
#  Usage:  python3 clutch_style_check.py     (after tennis_analysis.py)
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

TOURS = [
    ("ATP", SCRIPT_DIR / "atp_panel.csv", OUTPUT_DIR / "persistence_atp.csv"),
    ("WTA", SCRIPT_DIR / "wta_panel.csv", OUTPUT_DIR / "persistence_wta.csv"),
]


def career_style(panel_path):
    """Point-weighted career serve/return rates + best-of-5 share per player."""
    p = pd.read_csv(panel_path)
    g = p.groupby("player").agg(
        svpt=("svpt", "sum"), svpt_won=("svpt_won", "sum"),
        retpt=("retpt", "sum"), retpt_won=("retpt_won", "sum"),
        bo5=("best_of", lambda x: (x == 5).mean()),
    )
    g["spw"] = g.svpt_won / g.svpt * 100
    g["rpw"] = g.retpt_won / g.retpt * 100
    g["pw"] = (g.svpt_won + g.retpt_won) / (g.svpt + g.retpt) * 100
    return g.reset_index()[["player", "spw", "rpw", "pw", "bo5"]]


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    s2 = (resid ** 2).sum() / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta, se, r2, n - k


def main():
    rows = []
    for tour, panel_path, pers_path in TOURS:
        d = pd.read_csv(pers_path).merge(career_style(panel_path), on="player", how="left")
        assert d.spw.notna().all(), "style join failed for some players"

        r_pw, p_pw = stats.pearsonr(d.pw, d.clutch_early)
        y = d.clutch_late.values
        ones = np.ones(len(d))

        b_raw, se_raw, r2_raw, df_raw = ols(np.column_stack([ones, d.clutch_early]), y)
        # WTA is best-of-3 only -> bo5 is constant -> would make X singular
        ctrl = ["spw", "rpw", "bo5"] if d.bo5.nunique() > 1 else ["spw", "rpw"]
        Xc = np.column_stack([ones, d.clutch_early] + [d[c].values for c in ctrl])
        b_ctl, se_ctl, r2_ctl, df_ctl = ols(Xc, y)

        p_ctl = 2 * (1 - stats.t.cdf(abs(b_ctl[1] / se_ctl[1]), df_ctl))
        tcrit = stats.t.ppf(0.975, df_ctl)
        lo, hi = b_ctl[1] - tcrit * se_ctl[1], b_ctl[1] + tcrit * se_ctl[1]

        print(f"\n{'='*62}\n  {tour}  (n = {len(d)})   controls: {', '.join(ctrl)}\n{'='*62}")
        print(f"  corr(clutch_early, career pw) = {r_pw:+.3f}   p = {p_pw:.2e}")
        print(f"  beta raw                      = {b_raw[1]:.4f}")
        print(f"  beta controlling for style    = {b_ctl[1]:.4f}  "
              f"CI[{lo:.4f}, {hi:.4f}]  p = {p_ctl:.3g}")
        print(f"  -> {(1 - b_ctl[1]/b_raw[1])*100:.0f}% of the raw slope is attributable to style"
              f"   ({'still significant' if p_ctl < 0.05 else 'NOT significant'})")

        rows.append(dict(tour=tour, n=len(d), corr_clutch_pw=r_pw, p_corr=p_pw,
                         beta_raw=b_raw[1], r2_raw=r2_raw,
                         beta_style_controlled=b_ctl[1], ci_low=lo, ci_high=hi,
                         p_style_controlled=p_ctl, r2_controlled=r2_ctl,
                         pct_slope_from_style=(1 - b_ctl[1] / b_raw[1]) * 100,
                         controls="+".join(ctrl)))

    dest = OUTPUT_DIR / "clutch_style_control.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"\n  written -> {dest}\n")


if __name__ == "__main__":
    main()
