# ============================================================
#  Tennis analytics -- DSAI Project Seminar, Universitaet des Saarlandes
#
#  Structure:
#    Phase 1  Replicate Table 4 (R1-R4) and Figure 1
#    Phase 2  Non-linear extension of R1 (pw only): sigmoid / poly / GAM
#    Phase 3  Clutch proxy extension: non-linear R3/R4 + interaction terms
#    Phase 4  Parsimony check: pw+bpw vs spw+rpw+bpw
#    Phase 5  Clutch persistence (split-half career reliability)
#    Phase 6  Player-level 5-fold CV for Phases 2/3/4
# ============================================================

import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression as _LinReg
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from scipy.optimize import curve_fit
from pygam import LinearGAM, s, te

SCRIPT_DIR  = Path(__file__).resolve().parent
ATP_RAW_DIR = SCRIPT_DIR / "tennis_atp"
WTA_RAW_DIR = SCRIPT_DIR / "tennis_wta"
DATA_DIR    = SCRIPT_DIR                       # atp_panel.csv / wta_panel.csv live here
OUTPUT_DIR  = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  STEP 1 - Loading match data")
print("=" * 60)


def load_all_matches(folder, prefix, start_year):
    # Read every atp_matches_YYYY.csv (or wta_) from start_year onward
    # and concatenate into a single frame.
    pattern = os.path.join(folder, f"{prefix}_matches_[0-9][0-9][0-9][0-9].csv")
    all_files = sorted(glob.glob(pattern))

    dfs = []
    for filepath in all_files:
        year = int(filepath.split("_")[-1].replace(".csv", ""))
        if year >= start_year:
            df = pd.read_csv(filepath, low_memory=False)
            df["year"] = year
            dfs.append(df)

    if not dfs:
        print(f"  No files found at: {folder}")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


# Sample windows follow the paper's Table 3: ATP from 1990, WTA from 2003.
atp_raw = load_all_matches(ATP_RAW_DIR, "atp", start_year=1990)
wta_raw = load_all_matches(WTA_RAW_DIR, "wta", start_year=2003)

print(f"  ATP raw rows loaded : {len(atp_raw):>8,}")
print(f"  WTA raw rows loaded : {len(wta_raw):>8,}")


# ============================================================
print("\n" + "=" * 60)
print("  STEP 2 - Filtering (Table 3)")
print("=" * 60)
# ============================================================

def filter_matches(df):
    # Two filters from Table 3:
    #   (1) keep rows that actually have point-level stats (w_svpt present)
    #   (2) drop retirements and walkovers (RET / W/O in the score string)
    n_start = len(df)

    df = df[df["w_svpt"].notna()].copy()
    print(f"  After keeping rows with point stats : {len(df):,}  (removed {n_start - len(df):,})")

    if "score" in df.columns:
        mask_bad = df["score"].astype(str).str.contains("RET|W/O", na=False)
        df = df[~mask_bad].copy()
        print(f"  After removing retirements/walkovers: {len(df):,}")

    return df


atp = filter_matches(atp_raw)
print()
wta = filter_matches(wta_raw)


# ============================================================
print("\n" + "=" * 60)
print("  STEP 3 - Per-player career statistics")
print("=" * 60)
# ============================================================
# Each match row holds a winner and a loser. We expand every match into
# two player-rows (winner view + loser view), then aggregate per player:
#   serve points played/won      -> spw
#   return points played/won     -> rpw
#   break points faced/saved on own serve
#   break points on opponent serve (conversions)
#   matches won / played         -> mwp

def build_player_stats(df):
    # Winner's perspective
    winner_rows = pd.DataFrame({
        "player"           : df["winner_name"],
        "won_match"        : 1,
        "year"             : df["year"],
        "svpt"             : df["w_svpt"],
        "svpt_won"         : df["w_1stWon"].fillna(0) + df["w_2ndWon"].fillna(0),
        "retpt"            : df["l_svpt"],                                   # opponent's serve points = our return points
        "retpt_won"        : df["l_svpt"] - df["l_1stWon"].fillna(0) - df["l_2ndWon"].fillna(0),
        "bp_faced"         : df["w_bpFaced"].fillna(0),
        "bp_saved"         : df["w_bpSaved"].fillna(0),
        "bp_opp_faced"     : df["l_bpFaced"].fillna(0),
        "bp_opp_saved"     : df["l_bpSaved"].fillna(0),
    })

    # Loser's perspective
    loser_rows = pd.DataFrame({
        "player"           : df["loser_name"],
        "won_match"        : 0,
        "year"             : df["year"],
        "svpt"             : df["l_svpt"],
        "svpt_won"         : df["l_1stWon"].fillna(0) + df["l_2ndWon"].fillna(0),
        "retpt"            : df["w_svpt"],
        "retpt_won"        : df["w_svpt"] - df["w_1stWon"].fillna(0) - df["w_2ndWon"].fillna(0),
        "bp_faced"         : df["l_bpFaced"].fillna(0),
        "bp_saved"         : df["l_bpSaved"].fillna(0),
        "bp_opp_faced"     : df["w_bpFaced"].fillna(0),
        "bp_opp_saved"     : df["w_bpSaved"].fillna(0),
    })

    all_rows = pd.concat([winner_rows, loser_rows], ignore_index=True)

    # Rows with no serve/return points recorded are useless here.
    all_rows = all_rows[all_rows["svpt"] > 0]
    all_rows = all_rows[all_rows["retpt"] > 0]

    grouped = all_rows.groupby("player").agg(
        matches_total  = ("won_match",    "count"),
        matches_won    = ("won_match",    "sum"),
        svpt_total     = ("svpt",         "sum"),
        svpt_won_total = ("svpt_won",     "sum"),
        retpt_total    = ("retpt",        "sum"),
        retpt_won_total= ("retpt_won",    "sum"),
        bp_faced_total = ("bp_faced",     "sum"),
        bp_saved_total = ("bp_saved",     "sum"),
        bp_opp_faced   = ("bp_opp_faced", "sum"),
        bp_opp_saved   = ("bp_opp_saved", "sum"),
        # modal_year = the single year the player played most in. It's only a
        # rough per-player timestamp (one value per player, not per match),
        # which is why the Step 8 time-split carries the caveat noted there.
        modal_year     = ("year",      lambda x: x.mode()[0]),
    ).reset_index()
    grouped = grouped.rename(columns={"modal_year": "year"})

    grouped["spw"] = grouped["svpt_won_total"] / grouped["svpt_total"] * 100
    grouped["rpw"] = grouped["retpt_won_total"] / grouped["retpt_total"] * 100
    grouped["pw"]  = ((grouped["svpt_won_total"] + grouped["retpt_won_total"]) /
                      (grouped["svpt_total"]     + grouped["retpt_total"])) * 100
    grouped["mwp"] = grouped["matches_won"] / grouped["matches_total"] * 100

    # bpw = clutch ratio = break points won on return / break points lost on serve.
    bp_lost_on_serve = grouped["bp_faced_total"] - grouped["bp_saved_total"]
    bp_won_on_return = grouped["bp_opp_faced"] - grouped["bp_opp_saved"]
    # Players who never lost a break point on serve give a zero denominator ->
    # NaN. These get dropped by any regression that uses bpw (see run_regression).
    grouped["bpw"] = np.where(
        bp_lost_on_serve > 0,
        bp_won_on_return / bp_lost_on_serve,
        np.nan
    )

    grouped = grouped[grouped["matches_total"] >= 100].copy()   # career-length cutoff
    return grouped


print("  Building ATP player statistics...")
atp_players = build_player_stats(atp)
print(f"  ATP players with >=100 matches: {len(atp_players)}")
print(f"  Total ATP matches used        : {atp_players['matches_total'].sum():,}")

print()
print("  Building WTA player statistics...")
wta_players = build_player_stats(wta)
print(f"  WTA players with >=100 matches: {len(wta_players)}")
print(f"  Total WTA matches used        : {wta_players['matches_total'].sum():,}")

print("\n  Top 5 ATP players by mwp:")
top5 = atp_players.nlargest(5, "mwp")[["player","matches_total","mwp","pw","spw","rpw"]]
print(top5.to_string(index=False))


# ============================================================
print("\n" + "=" * 60)
print("  STEP 4 - Four regression models (Table 4)")
print("=" * 60)
# ============================================================
#   R1: mwp ~ pw
#   R2: mwp ~ pw + bpw
#   R3: mwp ~ spw + rpw
#   R4: mwp ~ spw + rpw + bpw

def run_regression(players_df, features, label):
    # dropna runs per call, so R1 (pw only) may be fit on a slightly larger
    # sample than R2/R3/R4 (which drop the NaN-bpw players). The effect is
    # tiny in practice but the four R2 values aren't from an identical sample.
    cols = features + ["mwp"]
    data = players_df[cols].dropna()

    X = data[features].values
    y = data["mwp"].values

    model = LinearRegression()
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))

    print(f"  {label:30s}  R2 = {r2:.3f}  ({r2*100:.1f}%)")
    return model, r2, data


print("\n  -- ATP (men) --")
m_r1, r2_m_r1, data_m_r1 = run_regression(atp_players, ["pw"],              "R1: pw only")
m_r2, r2_m_r2, data_m_r2 = run_regression(atp_players, ["pw","bpw"],        "R2: pw + bpw")
m_r3, r2_m_r3, data_m_r3 = run_regression(atp_players, ["spw","rpw"],       "R3: spw + rpw")
m_r4, r2_m_r4, data_m_r4 = run_regression(atp_players, ["spw","rpw","bpw"], "R4: spw + rpw + bpw")

print()
print("  -- WTA (women) --")
f_r1, r2_f_r1, data_f_r1 = run_regression(wta_players, ["pw"],              "R1: pw only")
f_r2, r2_f_r2, data_f_r2 = run_regression(wta_players, ["pw","bpw"],        "R2: pw + bpw")
f_r3, r2_f_r3, data_f_r3 = run_regression(wta_players, ["spw","rpw"],       "R3: spw + rpw")
f_r4, r2_f_r4, data_f_r4 = run_regression(wta_players, ["spw","rpw","bpw"], "R4: spw + rpw + bpw")

print()
print("  -- Paper's reported values (Table 4) --")
print("  ATP R1=94.0%  R2=94.5%  R3=93.0%  R4=94.1%")
print("  WTA R1=95.8%  R2=95.7%  R3=95.5%  R4=95.6%")


# ============================================================
print("\n" + "=" * 60)
print("  STEP 5 - Figure 1 (scatter + regression line)")
print("=" * 60)
# ============================================================

# Top-10 lists come from the paper's Table 1; used only for labelling.
TOP10_ATP = ["Novak Djokovic","Rafael Nadal","Roger Federer","Pete Sampras",
             "Carlos Alcaraz","Andre Agassi","Andy Roddick","Stefan Edberg",
             "Jannik Sinner","Boris Becker"]
TOP10_WTA = ["Justine Henin","Serena Williams","Iga Swiatek","Lindsay Davenport",
             "Maria Sharapova","Kim Clijsters","Ashleigh Barty","Victoria Azarenka",
             "Simona Halep","Elena Rybakina"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# No figure number in the title: this plot is Figure 1 in the report, and an
# embedded number that disagrees with the LaTeX caption looks like an error.
fig.suptitle("Match win % predicted by point win %",
             fontsize=13, fontweight="bold")

for ax, players_df, top10, title, r2_val, model_r1, data_r1 in [
    (axes[0], atp_players, TOP10_ATP, "ATP (men)",   r2_m_r1, m_r1, data_m_r1),
    (axes[1], wta_players, TOP10_WTA, "WTA (women)", r2_f_r1, f_r1, data_f_r1),
]:
    is_top = players_df["player"].isin(top10)
    others = players_df[~is_top]
    top    = players_df[is_top]

    ax.scatter(others["pw"], others["mwp"], s=15, alpha=0.4,
               color="#2563EB", label="Other players")
    ax.scatter(top["pw"], top["mwp"], s=60, alpha=0.9,
               color="#F97316", zorder=5, label="Top 10 players")

    for _, row in top.iterrows():
        ax.annotate(row["player"].split()[-1],      # last name only, keeps labels short
                    xy=(row["pw"], row["mwp"]),
                    xytext=(4, 2), textcoords="offset points",
                    fontsize=7, color="#1e3a5f")

    x_line = np.linspace(players_df["pw"].min(), players_df["pw"].max(), 200)
    y_line = model_r1.predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line, color="black", linewidth=1.5, label=f"Regression (R2={r2_val:.3f})")

    ax.set_xlabel("Point win percentage (pw %)", fontsize=11)
    ax.set_ylabel("Match win percentage (mwp %)", fontsize=11)
    ax.set_title(f"{title}  -  R2 = {r2_val*100:.1f}%", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.2)
    # Axis limits are fixed to the observed range of this ATP/WTA sample. On a
    # different subset (fewer years, one tour, stricter cutoff) some points may
    # fall outside these bounds and get clipped.
    ax.set_xlim(45, 57)
    ax.set_ylim(20, 105)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tennis_figure1.png", dpi=150, bbox_inches="tight")
print(f"  Figure saved to {OUTPUT_DIR / 'tennis_figure1.png'}")
plt.close('all')


# ============================================================
print("\n" + "=" * 60)
print("  STEP 6 - Descriptive findings")
print("=" * 60)
# ============================================================

print("\n  -- Match/point ratio (mpr) --")
# mpr = mwp / pw: how much winning matches outpaces winning points.
atp_players["mpr"] = atp_players["mwp"] / atp_players["pw"]
wta_players["mpr"] = wta_players["mwp"] / wta_players["pw"]

print("\n  ATP top 5 by mpr:")
print(atp_players.nlargest(5, "mpr")[["player","matches_total","mwp","pw","mpr"]].round(2).to_string(index=False))

print("\n  WTA top 5 by mpr:")
print(wta_players.nlargest(5, "mpr")[["player","matches_total","mwp","pw","mpr"]].round(2).to_string(index=False))

print("\n  -- Top-10 vs rest (average stats) --")
for label, players_df, top10 in [("ATP", atp_players, TOP10_ATP), ("WTA", wta_players, TOP10_WTA)]:
    top = players_df[players_df["player"].isin(top10)]
    rest = players_df[~players_df["player"].isin(top10)]
    print(f"\n  {label}:")
    print(f"    Top-10  - mwp: {top['mwp'].mean():.1f}%  pw: {top['pw'].mean():.1f}%  spw: {top['spw'].mean():.1f}%  rpw: {top['rpw'].mean():.1f}%")
    print(f"    Others  - mwp: {rest['mwp'].mean():.1f}%  pw: {rest['pw'].mean():.1f}%  spw: {rest['spw'].mean():.1f}%  rpw: {rest['rpw'].mean():.1f}%")

print("\n  -- R1 coefficients --")
b0_m, b1_m = m_r1.intercept_, m_r1.coef_[0]
b0_f, b1_f = f_r1.intercept_, f_r1.coef_[0]
print(f"  ATP: mwp = {b1_m:.2f} * pw + ({b0_m:.1f})  -> +1% pw adds ~{b1_m:.1f}% mwp")
print(f"  WTA: mwp = {b1_f:.2f} * pw + ({b0_f:.1f})  -> +1% pw adds ~{b1_f:.1f}% mwp")


# ============================================================
#   PHASE 2 - NON-LINEAR MODELING
#   Same pw -> mwp relationship as Phase 1, but allowing sigmoid / cubic /
#   GAM shapes instead of forcing a straight line.
# ============================================================

MEN   = atp_players
WOMEN = wta_players


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def sigmoid(x, L, k, x0, b):
    # Four-parameter logistic:
    #   L  amplitude (range covered), k steepness, x0 inflection, b floor.
    # Flat at low x, steep through the middle, flat again at high x -- the
    # shape the pw/mwp scatter actually shows.
    return L / (1 + np.exp(-k * (x - x0))) + b


def fit_linear_model(pw, mwp):
    m, c = np.polyfit(pw, mwp, deg=1)
    preds = m * pw + c
    return m, c, preds, r_squared(mwp, preds)


def fit_sigmoid_model(pw, mwp, pw_mean):
    initial_guess = [65, 0.55, pw_mean, 28]     # L, k, x0, b
    try:
        params, _ = curve_fit(
            sigmoid, pw, mwp,
            p0     = initial_guess,
            # scipy's default maxfev (~800) is too tight here: the four sigmoid
            # params trade off against each other and need more iterations.
            maxfev = 20000,
            bounds = ([20, 0.01, 45, 0],
                      [100, 5.0, 58, 50])
        )
        preds = sigmoid(pw, *params)
        return params, preds, r_squared(mwp, preds)
    except RuntimeError as e:
        print(f"    Sigmoid fit failed: {e}")
        return None, pw * 0, 0.0


def fit_polynomial_model(pw, mwp, degree=3):
    # Centre pw before fitting for numerical stability.
    pw_centred = pw - pw.mean()
    coeffs = np.polyfit(pw_centred, mwp, deg=degree)
    preds  = np.polyval(coeffs, pw_centred)
    return coeffs, preds, r_squared(mwp, preds)


def fit_gam_model(pw, mwp):
    # Single smooth spline on pw; the shape is learned, not assumed.
    pw_2d = pw.values.reshape(-1, 1)
    gam   = LinearGAM(s(0)).fit(pw_2d, mwp)
    preds = gam.predict(pw_2d)
    return gam, preds, r_squared(mwp, preds)


def time_split_r2(df, model_type, split_year=2015):
    # Chronological hold-out: train pre-2015, test 2015+.
    #
    # Caveat: 'year' is each player's modal year (see build_player_stats), so
    # this splits PLAYERS by one coarse timestamp, not actual matches by date.
    # Phase 6A adds a cleaner player-level 5-fold CV; both are kept because this
    # split's numbers are already on the poster.
    if 'year' not in df.columns:
        return None

    train = df[df['year'] <  split_year]
    test  = df[df['year'] >= split_year]

    if len(train) < 30 or len(test) < 10:
        return None

    pw_train, mwp_train = train['pw'].values, train['mwp'].values
    pw_test,  mwp_test  = test['pw'].values,  test['mwp'].values

    if model_type == 'linear':
        m, c = np.polyfit(pw_train, mwp_train, 1)
        preds = m * pw_test + c

    elif model_type == 'sigmoid':
        try:
            params, _ = curve_fit(sigmoid, pw_train, mwp_train,
                                  p0=[65, 0.55, pw_train.mean(), 28],
                                  maxfev=20000)
            preds = sigmoid(pw_test, *params)
        except:
            return None

    elif model_type == 'poly3':
        centre = pw_train.mean()
        coeffs = np.polyfit(pw_train - centre, mwp_train, 3)
        preds  = np.polyval(coeffs, pw_test - centre)

    elif model_type == 'gam':
        gam = LinearGAM(s(0)).fit(pw_train.reshape(-1,1), mwp_train)
        preds = gam.predict(pw_test.reshape(-1,1))

    return r_squared(mwp_test, preds)


# ---- STEP 7: fit all four families on ATP and WTA ----

print()
print("=" * 60)
print("  PHASE 2 - NON-LINEAR MODELING")
print("=" * 60)

results = {}

for tour_name, df in [("ATP (men)", MEN), ("WTA (women)", WOMEN)]:

    print(f"\n  -- {tour_name} --")
    print(f"  Players used : {len(df)}\n")

    pw  = df['pw'].values
    mwp = df['mwp'].values

    m, c, pred_lin, r2_lin = fit_linear_model(pw, mwp)
    print(f"  Model 1 | Linear")
    print(f"           R2 (in-sample) = {r2_lin:.4f}  ({r2_lin*100:.1f}%)")
    print(f"           mwp = {m:.3f} * pw + ({c:.1f})\n")

    sig_params, pred_sig, r2_sig = fit_sigmoid_model(pw, mwp, pw.mean())
    if sig_params is not None:
        L, k, x0, b = sig_params
        print(f"  Model 2 | Sigmoid (logistic)")
        print(f"           R2 (in-sample) = {r2_sig:.4f}  ({r2_sig*100:.1f}%)")
        print(f"           L={L:.2f}  k={k:.3f}  x0={x0:.2f}  b={b:.2f}")
        print(f"           dR2 vs Linear  = {r2_sig - r2_lin:+.4f}  ({(r2_sig-r2_lin)*100:+.2f}%)")
    print()

    poly_coeffs, pred_poly, r2_poly = fit_polynomial_model(pw, mwp, degree=3)
    print(f"  Model 3 | Polynomial (deg 3)")
    print(f"           R2 (in-sample) = {r2_poly:.4f}  ({r2_poly*100:.1f}%)")
    print(f"           dR2 vs Linear  = {r2_poly - r2_lin:+.4f}  ({(r2_poly-r2_lin)*100:+.2f}%)\n")

    gam_model, pred_gam, r2_gam = fit_gam_model(df[['pw']], mwp)
    print(f"  Model 4 | GAM (spline)")
    print(f"           R2 (in-sample) = {r2_gam:.4f}  ({r2_gam*100:.1f}%)")
    print(f"           dR2 vs Linear  = {r2_gam - r2_lin:+.4f}  ({(r2_gam-r2_lin)*100:+.2f}%)\n")

    print(f"  -- R2 summary ({tour_name}) --")
    print(f"  {'Model':<30} {'R2 in-sample':>14}  {'Best?':>6}")
    print(f"  {'-'*54}")
    model_r2s = {
        'Linear (baseline)':    r2_lin,
        'Sigmoid':              r2_sig,
        'Polynomial (deg 3)':   r2_poly,
        'GAM (spline)':         r2_gam,
    }
    best_r2 = max(model_r2s.values())
    for name, r2 in model_r2s.items():
        marker = "  <- best" if abs(r2 - best_r2) < 0.0001 else ""
        print(f"  {name:<30} {r2*100:>13.2f}%{marker}")
    print()

    results[tour_name] = {
        'pw': pw, 'mwp': mwp,
        'pred_lin': pred_lin, 'r2_lin': r2_lin,
        'pred_sig': pred_sig, 'r2_sig': r2_sig,
        'pred_poly': pred_poly, 'r2_poly': r2_poly,
        'pred_gam': pred_gam, 'r2_gam': r2_gam,
        'sig_params': sig_params, 'linear_m': m, 'linear_c': c,
        'gam_model': gam_model,
    }


# ---- STEP 8: out-of-sample evaluation (time split) ----

print()
print("=" * 60)
print("  STEP 8 - OUT-OF-SAMPLE (train<2015, test>=2015)")
print("=" * 60)

for tour_name, df in [("ATP (men)", MEN), ("WTA (women)", WOMEN)]:
    print(f"\n  -- {tour_name} --")
    for mtype in ['linear', 'sigmoid', 'poly3', 'gam']:
        oos = time_split_r2(df, mtype)
        label = {'linear': 'Linear', 'sigmoid': 'Sigmoid',
                 'poly3': 'Polynomial (deg 3)', 'gam': 'GAM'}[mtype]
        if oos is not None:
            print(f"  {label:<22}  out-of-sample R2 = {oos*100:.2f}%")
        else:
            print(f"  {label:<22}  out-of-sample R2 = n/a")


# ---- STEP 9: figures ----

print()
print("=" * 60)
print("  STEP 9 - Generating figures")
print("=" * 60)

C_LINEAR   = '#1a1a1a'
C_SIGMOID  = '#C0392B'
C_POLY     = '#2980B9'
C_GAM      = '#27AE60'
C_SCATTER  = '#5B8DB8'
C_TOP      = '#E8652A'


# Figure 1 (Phase 2): linear vs sigmoid, publication version
fig1, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='white', sharey=False)
fig1.suptitle(
    'Figure 1 - Career Point Win % predicts Match Win %\n'
    'Linear Baseline vs. Non-linear Sigmoid Fit',
    fontsize=14, fontweight='bold', y=0.98
)

for ax, (tour_name, res) in zip(axes, results.items()):
    pw, mwp = res['pw'], res['mwp']

    ax.scatter(pw, mwp, s=25, alpha=0.25, color=C_SCATTER,
               linewidths=0, zorder=2, label='Players')

    xgrid = np.linspace(pw.min() - 0.3, pw.max() + 0.3, 500)

    m, c = res['linear_m'], res['linear_c']
    ax.plot(xgrid, m * xgrid + c,
            color=C_LINEAR, lw=2.0, ls='--', zorder=5,
            label=f"Linear  R2={res['r2_lin']:.3f}")

    if res['sig_params'] is not None:
        y_sig = sigmoid(xgrid, *res['sig_params'])
        ax.plot(xgrid, y_sig,
                color=C_SIGMOID, lw=2.8, zorder=6,
                label=f"Sigmoid  R2={res['r2_sig']:.3f}")

        # Shade the tails, where linear and sigmoid disagree most.
        y_lin_grid = m * xgrid + c
        ax.fill_between(xgrid, y_lin_grid, y_sig,
                        where=((xgrid < pw.mean() - 1.5) |
                               (xgrid > pw.mean() + 1.5)),
                        color=C_SIGMOID, alpha=0.10, zorder=1,
                        label='Divergence zone (tails)')

        delta = res['r2_sig'] - res['r2_lin']
        ax.text(0.97, 0.05, f"dR2 = {delta:+.4f}",
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=10.5, color=C_SIGMOID, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.35',
                          facecolor='white', edgecolor=C_SIGMOID, lw=1.3))

    ax.set_xlabel('Point win percentage (pw %)', fontsize=12, labelpad=7)
    ax.set_ylabel('Match win percentage (mwp %)', fontsize=12, labelpad=7)
    ax.set_title(tour_name, fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.22, lw=0.7)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9.5, loc='upper left', framealpha=0.9, edgecolor='#ccc')

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig1.savefig(OUTPUT_DIR / 'tennis_figure_main.png', dpi=180, bbox_inches='tight', facecolor='white')
print(f"  Figure 1 (main) saved -> {OUTPUT_DIR / 'tennis_figure_main.png'}")


# Figure 2 (Phase 2): all four families side by side
fig2, axes2 = plt.subplots(2, 4, figsize=(20, 11), facecolor='white')
fig2.suptitle('Figure 2 - All Non-linear Models vs. Linear Baseline',
              fontsize=14, fontweight='bold', y=0.99)

model_configs = [
    ('Linear\n(Baseline)',  'pred_lin',  'r2_lin',  C_LINEAR,  '--', 2.0),
    ('Sigmoid\n(Logistic)', 'pred_sig',  'r2_sig',  C_SIGMOID, '-',  2.8),
    ('Polynomial\n(deg 3)', 'pred_poly', 'r2_poly', C_POLY,    '-',  2.4),
    ('GAM\n(Spline)',       'pred_gam',  'r2_gam',  C_GAM,     '-',  2.4),
]

for row_idx, (tour_name, res) in enumerate(results.items()):
    pw, mwp = res['pw'], res['mwp']
    xgrid = np.linspace(pw.min() - 0.2, pw.max() + 0.2, 400)

    for col_idx, (model_label, pred_key, r2_key, colour, ls, lw) in enumerate(model_configs):
        ax = axes2[row_idx, col_idx]
        ax.scatter(pw, mwp, s=15, alpha=0.20, color=C_SCATTER, linewidths=0, zorder=2)

        if pred_key == 'pred_sig' and res['sig_params'] is not None:
            ax.plot(xgrid, sigmoid(xgrid, *res['sig_params']), color=colour, lw=lw, ls=ls, zorder=5)
        elif pred_key == 'pred_lin':
            m2, c2 = res['linear_m'], res['linear_c']
            ax.plot(xgrid, m2*xgrid+c2, color=colour, lw=lw, ls=ls, zorder=5)
        elif pred_key == 'pred_gam':
            ax.plot(xgrid, res['gam_model'].predict(xgrid.reshape(-1,1)), color=colour, lw=lw, ls=ls, zorder=5)
        elif pred_key == 'pred_poly':
            centre = pw.mean()
            poly_y = np.polyval(np.polyfit(pw - centre, mwp, 3), xgrid - centre)
            ax.plot(xgrid, poly_y, color=colour, lw=lw, ls=ls, zorder=5)

        r2_val = res[r2_key]
        ax.set_title(f"{model_label}\nR2 = {r2_val:.4f}", fontsize=10, fontweight='bold', color=colour)
        ax.set_xlabel('pw %', fontsize=9)
        ax.set_ylabel('mwp %', fontsize=9) if col_idx == 0 else None
        ax.grid(True, alpha=0.20, lw=0.6)
        ax.tick_params(labelsize=8)
        if col_idx == 0:
            ax.set_ylabel(f'{tour_name}\nmwp %', fontsize=9, fontweight='bold')

plt.tight_layout()
fig2.savefig(OUTPUT_DIR / 'tennis_figure_all_models.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f"  Figure 2 (all models) saved -> {OUTPUT_DIR / 'tennis_figure_all_models.png'}")


# Figure 3 (Phase 2): residuals of the linear model
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
# This plot is Figure 2 in the report, so no embedded figure number here --
# it previously read "Figure 3", contradicting the caption.
fig3.suptitle(
    'Residuals of the Linear Model\n'
    '(Positive = linear underestimates mwp, Negative = overestimates)',
    fontsize=13, fontweight='bold'
)

for ax, (tour_name, res) in zip(axes3, results.items()):
    pw, mwp = res['pw'], res['mwp']
    residuals = mwp - res['pred_lin']
    colours = [C_SIGMOID if r > 0 else C_POLY for r in residuals]

    ax.scatter(pw, residuals, s=20, alpha=0.35, c=colours, linewidths=0, zorder=3)
    ax.axhline(0, color='black', lw=1.5, ls='--', zorder=5, label='Zero residual')

    gam_res = LinearGAM(s(0)).fit(pw.reshape(-1,1), residuals)
    xgrid = np.linspace(pw.min(), pw.max(), 300)
    ax.plot(xgrid, gam_res.predict(xgrid.reshape(-1,1)),
            color='black', lw=2.2, zorder=6, label='Residual trend (GAM)')

    res_trend = gam_res.predict(xgrid.reshape(-1,1))
    ax.fill_between(xgrid, 0, res_trend, where=(xgrid < pw.mean() - 1),
                    alpha=0.18, color=C_POLY, label='Underfit (left tail)')
    ax.fill_between(xgrid, 0, res_trend, where=(xgrid > pw.mean() + 1),
                    alpha=0.18, color=C_SIGMOID, label='Overfit (right tail)')

    ax.set_xlabel('Point win percentage (pw %)', fontsize=11, labelpad=7)
    ax.set_ylabel('Residual (actual - predicted mwp %)', fontsize=11, labelpad=7)
    ax.set_title(tour_name, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.20, lw=0.6)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=8.5, loc='upper left', framealpha=0.9)

plt.tight_layout()
fig3.savefig(OUTPUT_DIR / 'tennis_figure_residuals.png', dpi=160, bbox_inches='tight', facecolor='white')
print(f"  Figure 3 (residuals) saved -> {OUTPUT_DIR / 'tennis_figure_residuals.png'}")


# Figure 4 (Phase 2): R2 bar chart
fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
fig4.suptitle('Figure 4 - R2 Comparison Across All Models', fontsize=13, fontweight='bold')

model_names = ['Linear\n(baseline)', 'Sigmoid', 'Polynomial\n(deg 3)', 'GAM\n(spline)']
bar_colours = [C_LINEAR, C_SIGMOID, C_POLY, C_GAM]

for ax, (tour_name, res) in zip(axes4, results.items()):
    r2_values = [res['r2_lin'], res['r2_sig'], res['r2_poly'], res['r2_gam']]
    bars = ax.bar(model_names, [v * 100 for v in r2_values],
                  color=bar_colours, alpha=0.85, edgecolor='white', linewidth=1.2, zorder=3)

    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val*100:.2f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax.axhline(res['r2_lin'] * 100, color=C_LINEAR, lw=1.2, ls=':', alpha=0.6, zorder=4)
    ax.set_ylim(min(r2_values)*100 - 0.5, max(r2_values)*100 + 0.8)
    ax.set_title(tour_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('R2 (%)', fontsize=11)
    ax.grid(True, axis='y', alpha=0.25, lw=0.7)
    ax.tick_params(labelsize=9)

plt.tight_layout()
fig4.savefig(OUTPUT_DIR / 'tennis_figure_r2_comparison.png', dpi=160, bbox_inches='tight', facecolor='white')
print(f"  Figure 4 (R2 comparison) saved -> {OUTPUT_DIR / 'tennis_figure_r2_comparison.png'}")


# ---- Phase 2 summary ----

print()
print("=" * 60)
print("  PHASE 2 - SUMMARY")
print("=" * 60)
print()

for tour_name, res in results.items():
    print(f"  {tour_name}")
    print(f"  {'-'*40}")
    models = [
        ('Linear (paper baseline)', res['r2_lin']),
        ('Sigmoid / Logistic',      res['r2_sig']),
        ('Polynomial (degree 3)',   res['r2_poly']),
        ('GAM (spline)',            res['r2_gam']),
    ]
    best = max(models, key=lambda x: x[1])
    for name, r2 in models:
        delta = r2 - res['r2_lin']
        delta_str = f"  (d = {delta:+.4f})" if name != 'Linear (paper baseline)' else ""
        best_str  = "  <- best" if name == best[0] and name != 'Linear (paper baseline)' else ""
        print(f"    {name:<28}  R2 = {r2:.4f}  ({r2*100:.2f}%){delta_str}{best_str}")
    print()

print(f"  Figures saved to: {OUTPUT_DIR}")
print()
print("  Reading dR2:")
print("    > 0.005      meaningful gain over linear")
print("    0.001-0.005  modest but real")
print("    < 0.001      negligible (linear already near-perfect)")
print()
print("  Figure 3 is the key diagnostic: a U-shape in the linear residuals")
print("  is what motivates the sigmoid -- systematic tail error the")
print("  non-linear fits correct.")
plt.close('all')


# ============================================================
#   PHASE 3 - CLUTCH PROXY EXTENSION
#   Adds bpw to the non-linear models, in both an additive and an explicit
#   spw x rpw interaction form.
#
#   Table 4 goes R1 (pw) -> R2 (pw+bpw) and R3 (spw+rpw) -> R4 (+bpw).
#   Phase 2 only built the non-linear version of R1. This phase builds the
#   missing non-linear analogues of R3 and R4:
#       R3-style = spw + rpw           (no clutch term)
#       R4-style = spw + rpw + bpw     (clutch proxy added)
#   for every family (Linear, Poly, GAM, Sigmoid) and two combine modes:
#       additive    = spw and rpw enter separately (matches the paper)
#       interaction = an explicit spw x rpw term on top (the paper never
#                     tests this joint effect)
#
#   Central question: once the pw->mwp curve is allowed to bend, does adding
#   bpw still buy much (higher R2, lower AIC)? That's the non-linear analogue
#   of the paper's R2-vs-R1 / R4-vs-R3 comparison.
#
#   Expects MEN / WOMEN to carry columns spw, rpw, bpw, mwp -- all already
#   computed in build_player_stats(), so nothing is recomputed here.
# ============================================================

# r_squared() and sigmoid() are reused from Phase 2, not redefined.

def aic_score(y_true, y_pred, n_params):
    # AIC = n*ln(RSS/n) + 2k, same criterion the paper uses (lower = better).
    # n_params must include the intercept. For GAMs we pass effective dof as k,
    # since a spline has no fixed integer parameter count.
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    if rss <= 0:
        rss = 1e-10
    return n * np.log(rss / n) + 2 * n_params


# --- Linear (multivariate OLS) = what R3/R4 actually are, plus interaction ---
def fit_linear_features(X, y):
    model = _LinReg().fit(X, y)
    preds = model.predict(X)
    k = X.shape[1] + 1
    return preds, r_squared(y, preds), aic_score(y, preds, k)


# --- Polynomial: additive vs interaction ---
# Additive: each predictor gets its own x, x^2, x^3 but no cross-products.
# Interaction: one explicit spw_c * rpw_c column added on top (a single
# interpretable joint term, not a full multivariate expansion).
def fit_poly_additive(feature_arrays, y, degree=3):
    cols, names = [], []
    for name, arr in feature_arrays.items():
        centred = arr - arr.mean()
        for d in range(1, degree + 1):
            cols.append(centred ** d)
            names.append(f"{name}^{d}")
    X = np.column_stack(cols)
    preds, r2, aic = fit_linear_features(X, y)
    return preds, r2, aic, names


def fit_poly_interaction(feature_arrays, y, degree=3, interaction_pairs=None):
    cols, names = [], []
    centred = {name: arr - arr.mean() for name, arr in feature_arrays.items()}
    for name, arr in centred.items():
        for d in range(1, degree + 1):
            cols.append(arr ** d)
            names.append(f"{name}^{d}")
    if interaction_pairs:
        for a, b in interaction_pairs:
            cols.append(centred[a] * centred[b])
            names.append(f"{a}x{b}")
    X = np.column_stack(cols)
    preds, r2, aic = fit_linear_features(X, y)
    return preds, r2, aic, names


# --- GAM: additive vs tensor interaction ---
# Additive: one spline per predictor, s(0)+s(1)+s(2).
# Interaction: swaps the two spw/rpw splines for a 2D tensor smooth te(i,j),
# a fully flexible joint surface over (spw, rpw). bpw stays additive.
def fit_gam_additive(feature_arrays, y):
    names = list(feature_arrays.keys())
    X = np.column_stack([feature_arrays[n] for n in names])
    term = s(0)
    for i in range(1, len(names)):
        term = term + s(i)
    gam = LinearGAM(term).fit(X, y)
    preds = gam.predict(X)
    edof = gam.statistics_['edof']
    aic = aic_score(y, preds, edof + 1)
    return preds, r_squared(y, preds), aic, edof


def fit_gam_interaction(feature_arrays, y, interaction_pair_idx=(0, 1)):
    # interaction_pair_idx=(0,1) assumes the two interacting features were
    # inserted into feature_arrays FIRST (relies on dict insertion order).
    # Every caller builds FS3/FS4 with spw, rpw first for this reason -- reorder
    # and this silently fits the wrong pair, no error raised.
    names = list(feature_arrays.keys())
    X = np.column_stack([feature_arrays[n] for n in names])
    i, j = interaction_pair_idx
    term = te(i, j)
    for idx in range(len(names)):
        if idx not in (i, j):
            term = term + s(idx)
    gam = LinearGAM(term).fit(X, y)
    preds = gam.predict(X)
    edof = gam.statistics_['edof']
    aic = aic_score(y, preds, edof + 1)
    return preds, r_squared(y, preds), aic, edof


# --- Sigmoid: multivariate single-index model ---
# Phase 2's sigmoid is a 4-param S-curve of one input. To keep that shape with
# several predictors we build one linear index z and push it through the same
# logistic:
#     z   = w1*spw + w2*rpw (+ w3*bpw) (+ w_int*spw*rpw)
#     mwp = L / (1 + exp(-k*(z - z0))) + b
# Predictors are standardised first, purely for numerical stability -- spw/rpw
# sit near 50-70, bpw near 0.3-3, an interaction on yet another scale; an
# unstandardised index is hard for curve_fit. Standardising changes the
# conditioning, not what's being tested.
def _standardize(arr):
    mu, sigma = arr.mean(), arr.std()
    sigma = sigma if sigma > 1e-9 else 1.0
    return (arr - mu) / sigma


def _sigmoid_index(X_flat, *params):
    # params is flat: one weight per feature, then the 4 shape params
    # (L, k, z0, b) at the end.
    n_features = X_flat.shape[0]
    weights = np.array(params[:n_features])
    L, k, z0, b = params[n_features:]
    z = weights @ X_flat
    return L / (1 + np.exp(-k * (z - z0))) + b


def fit_sigmoid_multivar(feature_arrays, y, interaction_pairs=None):
    names = list(feature_arrays.keys())
    std_cols = [_standardize(feature_arrays[n]) for n in names]
    if interaction_pairs:
        for a, b in interaction_pairs:
            std_cols.append(_standardize(feature_arrays[a] * feature_arrays[b]))
            names.append(f"{a}x{b}")
    X_flat = np.vstack(std_cols)

    n_features = X_flat.shape[0]
    w0 = [1.0 / n_features] * n_features
    p0 = w0 + [65, 1.0, 0.0, 28]           # L, k, z0, b -- Phase 2 ballpark
    lower = [-20] * n_features + [20, 0.001, -10, 0]
    upper = [20] * n_features + [100, 10.0, 10, 50]

    try:
        params, _ = curve_fit(_sigmoid_index, X_flat, y, p0=p0,
                               maxfev=50000, bounds=(lower, upper))
        preds = _sigmoid_index(X_flat, *params)
        return preds, r_squared(y, preds), aic_score(y, preds, len(params))
    except RuntimeError as e:
        print(f"    Multivariate sigmoid fit failed: {e}")
        return y * 0.0, 0.0, np.inf


# ---- Run: 2 tours x 2 feature-sets x 4 families x 2 combos ----

MODEL_FAMILIES = ["Linear", "Polynomial (deg 3)", "GAM", "Sigmoid"]
COMBOS = ["additive", "interaction"]

all_rows = []
figure_data = {}

print()
print("=" * 90)
print("  PHASE 3 - CLUTCH PROXY EXTENSION (spw+rpw -> spw+rpw+bpw, non-linear)")
print("=" * 90)

for tour_name, df in [("ATP (men)", MEN), ("WTA (women)", WOMEN)]:
    spw, rpw, bpw, mwp = df['spw'].values, df['rpw'].values, df['bpw'].values, df['mwp'].values
    n_players = len(df)

    FS3 = {'spw': spw, 'rpw': rpw}
    FS4 = {'spw': spw, 'rpw': rpw, 'bpw': bpw}

    fs_results = {}

    for fs_label, fs in [("R3-style (spw+rpw)", FS3), ("R4-style (spw+rpw+bpw)", FS4)]:

        X_add = np.column_stack(list(fs.values()))
        # Interaction is always spw x rpw specifically -- even for R4-style --
        # because the question is whether serve and return jointly interact.
        X_int = np.column_stack(list(fs.values()) + [fs['spw'] * fs['rpw']])

        _, r2, aic = fit_linear_features(X_add, mwp)
        fs_results[(fs_label, "Linear", "additive")] = (r2, aic, None)
        _, r2, aic = fit_linear_features(X_int, mwp)
        fs_results[(fs_label, "Linear", "interaction")] = (r2, aic, None)

        _, r2, aic, _ = fit_poly_additive(fs, mwp, degree=3)
        fs_results[(fs_label, "Polynomial (deg 3)", "additive")] = (r2, aic, None)
        _, r2, aic, _ = fit_poly_interaction(fs, mwp, degree=3, interaction_pairs=[('spw', 'rpw')])
        fs_results[(fs_label, "Polynomial (deg 3)", "interaction")] = (r2, aic, None)

        _, r2, aic, edof = fit_gam_additive(fs, mwp)
        fs_results[(fs_label, "GAM", "additive")] = (r2, aic, edof)
        _, r2, aic, edof = fit_gam_interaction(fs, mwp, interaction_pair_idx=(0, 1))
        fs_results[(fs_label, "GAM", "interaction")] = (r2, aic, edof)

        _, r2, aic = fit_sigmoid_multivar(fs, mwp)
        fs_results[(fs_label, "Sigmoid", "additive")] = (r2, aic, None)
        _, r2, aic = fit_sigmoid_multivar(fs, mwp, interaction_pairs=[('spw', 'rpw')])
        fs_results[(fs_label, "Sigmoid", "interaction")] = (r2, aic, None)

    print(f"\n  -- {tour_name}  (n players = {n_players}) --\n")

    for family in MODEL_FAMILIES:
        for combo in COMBOS:
            r2_3, aic_3, _ = fs_results[("R3-style (spw+rpw)", family, combo)]
            r2_4, aic_4, edof_4 = fs_results[("R4-style (spw+rpw+bpw)", family, combo)]
            delta_r2 = r2_4 - r2_3
            delta_aic = aic_4 - aic_3

            edof_note = ""
            if edof_4 is not None:
                edof_note = f"   (GAM edof={edof_4:.1f} on n={n_players})"
                if edof_4 > n_players / 10:
                    edof_note += "  <- edof high vs n, read R2 with caution (possible overfit)"

            print(f"  {family:<20} {combo:<12}  "
                  f"R3 R2={r2_3:.4f}   R4 R2={r2_4:.4f}   "
                  f"dR2(bpw)={delta_r2:+.4f}   dAIC(bpw)={delta_aic:+.1f}{edof_note}")

            all_rows.append(dict(
                tour=tour_name, family=family, combo=combo,
                r2_r3=r2_3, r2_r4=r2_4, delta_r2_bpw=delta_r2,
                aic_r3=aic_3, aic_r4=aic_4, delta_aic_bpw=delta_aic,
            ))
        print()

    figure_data[tour_name] = fs_results


results_table = pd.DataFrame(all_rows)
print("=" * 90)
print("  PHASE 3 - FULL RESULTS TABLE")
print("=" * 90)
print(results_table.round(4).to_string(index=False))

results_table.to_csv(OUTPUT_DIR / "clutch_proxy_extension_results.csv", index=False)
print(f"\n  Saved -> {OUTPUT_DIR / 'clutch_proxy_extension_results.csv'}")


# Figure 5 (Phase 3): R2 by family, R3-style vs R4-style, additive vs interaction
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
fig.suptitle('Figure 5 - Clutch proxy (bpw) extension: R2 across non-linear families',
             fontsize=13, fontweight='bold')

bar_width = 0.18
x = np.arange(len(MODEL_FAMILIES))
combo_colors = {"additive": "#5B8DB8", "interaction": "#E8652A"}

for row_idx, tour_name in enumerate(["ATP (men)", "WTA (women)"]):
    for col_idx, fs_label in enumerate(["R3-style (spw+rpw)", "R4-style (spw+rpw+bpw)"]):
        ax = axes[row_idx, col_idx]
        for combo_idx, combo in enumerate(COMBOS):
            vals = [figure_data[tour_name][(fs_label, fam, combo)][0] * 100 for fam in MODEL_FAMILIES]
            offset = (combo_idx - 0.5) * bar_width
            ax.bar(x + offset, vals, width=bar_width, label=combo,
                   color=combo_colors[combo], alpha=0.85, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_FAMILIES, rotation=20, fontsize=8)
        ax.set_title(f"{tour_name} - {fs_label}", fontsize=10, fontweight='bold')
        ax.set_ylabel("R2 (%)", fontsize=9)
        ax.grid(True, axis='y', alpha=0.25)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_DIR / 'tennis_figure_clutch_extension.png', dpi=160, bbox_inches='tight', facecolor='white')
print(f"\n  Figure 5 saved -> {OUTPUT_DIR / 'tennis_figure_clutch_extension.png'}")
plt.close('all')


print()
print("=" * 90)
print("  PHASE 3 - NOTES")
print("=" * 90)
print("""
  dR2(bpw) is the non-linear analogue of the paper's R2-vs-R1 and R4-vs-R3:
  how much the clutch proxy improves fit once non-linearity is allowed.

  If dR2(bpw) stays small (~0.001-0.005, matching the paper's 0.1-0.5pp bump)
  across every family, that strengthens the paper's conclusion -- clutch adds
  little regardless of functional form.

  The interaction columns test something the paper does not: whether spw and
  rpw have a joint effect. If interaction barely beats additive, the two skills
  operate roughly independently, consistent with the paper's i.i.d. framing.

  GAM is the most flexible family here and so the most prone to fitting noise,
  especially on the smaller WTA sample. A GAM-only R2 win that poly doesn't
  echo is weaker evidence than one that shows up everywhere.
""")


# ============================================================
#   PHASE 4 - PARSIMONY CHECK: pw+bpw vs spw+rpw+bpw
#
#   In the paper's Table 4, R2 (pw+bpw) beats R4 (spw+rpw+bpw) on both R2 and
#   AIC, for both tours:
#       ATP:  R2 = 94.5% (AIC -2291)   R4 = 94.1% (AIC -2256)
#       WTA:  R2 = 95.7% (AIC -1168)   R4 = 95.6% (AIC -1165)
#   Splitting pw into spw/rpw costs a parameter without adding much new info
#   (spw and rpw are correlated), and AIC's penalty isn't covered by the fit
#   gain. This phase checks whether "simpler wins" survives once every family
#   is allowed to be non-linear.
#
#   pw is arithmetically derived from spw and rpw, so pw is NEVER placed in the
#   same model as spw/rpw (near-perfect collinearity). The two feature sets stay
#   in separate models; only their FITS are compared, as the paper compares R2
#   against R4.
# ============================================================

step11_rows = []

print()
print("=" * 95)
print("  PHASE 4 - R2-style (pw+bpw) vs R4-style (spw+rpw+bpw), non-linear")
print("=" * 95)

for tour_name, df in [("ATP (men)", MEN), ("WTA (women)", WOMEN)]:
    pw, spw, rpw, bpw, mwp = df['pw'].values, df['spw'].values, df['rpw'].values, df['bpw'].values, df['mwp'].values
    n_players = len(df)

    FS2 = {'pw': pw, 'bpw': bpw}
    FS4 = {'spw': spw, 'rpw': rpw, 'bpw': bpw}

    fs_results = {}
    for fs_label, fs, inter_pair in [
        ("R2-style (pw+bpw)",       FS2, ('pw', 'bpw')),
        ("R4-style (spw+rpw+bpw)",  FS4, ('spw', 'rpw')),
    ]:
        X_add = np.column_stack(list(fs.values()))
        X_int = np.column_stack(list(fs.values()) + [fs[inter_pair[0]] * fs[inter_pair[1]]])

        _, r2, aic = fit_linear_features(X_add, mwp)
        fs_results[(fs_label, "Linear", "additive")] = (r2, aic)
        _, r2, aic = fit_linear_features(X_int, mwp)
        fs_results[(fs_label, "Linear", "interaction")] = (r2, aic)

        _, r2, aic, _ = fit_poly_additive(fs, mwp, degree=3)
        fs_results[(fs_label, "Polynomial (deg 3)", "additive")] = (r2, aic)
        _, r2, aic, _ = fit_poly_interaction(fs, mwp, degree=3, interaction_pairs=[inter_pair])
        fs_results[(fs_label, "Polynomial (deg 3)", "interaction")] = (r2, aic)

        _, r2, aic, edof = fit_gam_additive(fs, mwp)
        fs_results[(fs_label, "GAM", "additive")] = (r2, aic)
        _, r2, aic, edof = fit_gam_interaction(fs, mwp, interaction_pair_idx=(0, 1))
        fs_results[(fs_label, "GAM", "interaction")] = (r2, aic)

        _, r2, aic = fit_sigmoid_multivar(fs, mwp)
        fs_results[(fs_label, "Sigmoid", "additive")] = (r2, aic)
        _, r2, aic = fit_sigmoid_multivar(fs, mwp, interaction_pairs=[inter_pair])
        fs_results[(fs_label, "Sigmoid", "interaction")] = (r2, aic)

    print(f"\n  -- {tour_name}  (n players = {n_players}) --\n")

    for family in MODEL_FAMILIES:
        for combo in COMBOS:
            r2_2, aic_2 = fs_results[("R2-style (pw+bpw)", family, combo)]
            r2_4, aic_4 = fs_results[("R4-style (spw+rpw+bpw)", family, combo)]

            # AIC is the fair criterion (fit vs parameter count), same as the
            # paper's R2-vs-R4 comparison. R2 alone almost always favours the
            # model with more predictors.
            winner = "R2-style (pw+bpw)" if aic_2 < aic_4 else "R4-style (spw+rpw+bpw)"

            print(f"  {family:<20} {combo:<12}  "
                  f"R2-style: R2={r2_2:.4f} AIC={aic_2:.1f}   "
                  f"R4-style: R2={r2_4:.4f} AIC={aic_4:.1f}   "
                  f"-> better (AIC): {winner}")

            step11_rows.append(dict(
                tour=tour_name, family=family, combo=combo,
                r2_pw_bpw=r2_2, r2_spw_rpw_bpw=r2_4,
                aic_pw_bpw=aic_2, aic_spw_rpw_bpw=aic_4,
                better_fit_by_aic=winner,
            ))
        print()

step11_table = pd.DataFrame(step11_rows)
print("=" * 95)
print("  PHASE 4 - FULL RESULTS TABLE")
print("=" * 95)
print(step11_table.round(4).to_string(index=False))

step11_table.to_csv(OUTPUT_DIR / "parsimony_check_pw_vs_spw_rpw.csv", index=False)
print(f"\n  Saved -> {OUTPUT_DIR / 'parsimony_check_pw_vs_spw_rpw.csv'}")

n_favor_r2 = (step11_table["better_fit_by_aic"] == "R2-style (pw+bpw)").sum()
n_favor_r4 = (step11_table["better_fit_by_aic"] == "R4-style (spw+rpw+bpw)").sum()

print()
print("=" * 95)
print("  PHASE 4 - NOTES")
print("=" * 95)
print(f"""
  Of {len(step11_table)} model/combo comparisons:
    {n_favor_r2:>2} favour the simpler pw+bpw model (by AIC)
    {n_favor_r4:>2} favour the granular spw+rpw+bpw model (by AIC)

  pw+bpw winning most comparisons replicates the paper's R2-beats-R4 result and
  extends it: the serve/return split doesn't earn its extra parameter even with
  a non-linear functional form.

  If spw+rpw+bpw starts winning more often here than in the paper's linear-only
  comparison, check which families drive it. If it's mainly GAM, apply the same
  overfitting caution as Phase 3 -- the most flexible family is the most likely
  to reward extra parameters that don't reflect real structure.
""")


# ============================================================
#  PHASE 5 - CLUTCH PERSISTENCE TEST
#  Does first-half-career clutch predict second-half clutch?
# ============================================================

print("=" * 60)
print("  PHASE 5 - CLUTCH PERSISTENCE TEST")
print("=" * 60)

from scipy import stats

# --- Load panels ---
print("\nLoading panels...")
atp_panel = pd.read_csv(DATA_DIR / "atp_panel.csv")
wta_panel = pd.read_csv(DATA_DIR / "wta_panel.csv")
print(f"  ATP panel: {len(atp_panel):,} rows")
print(f"  WTA panel: {len(wta_panel):,} rows")

for panel in [atp_panel, wta_panel]:
    panel["date"] = pd.to_datetime(panel["date"], format="%Y%m%d", errors="coerce")


# --- Match simulator (Sim B: i.i.d. Bernoulli points + tennis scoring) ---
def simulate_match(spw, rpw, best_of_5, rng):
    # spw, rpw in [0,1]; returns 1 if the player wins the match.
    sets_to_win = 3 if best_of_5 else 2
    player_sets = 0
    opp_sets = 0

    while player_sets < sets_to_win and opp_sets < sets_to_win:
        player_games = 0
        opp_games = 0
        server_is_player = True   # alternates; set outcome barely depends on it

        while True:
            # One game: first to 4 points, win-by-2.
            p_win_point = spw if server_is_player else rpw
            player_pts = 0
            opp_pts = 0
            while True:
                if rng.random() < p_win_point:
                    player_pts += 1
                else:
                    opp_pts += 1
                if player_pts >= 4 and player_pts - opp_pts >= 2:
                    player_games += 1
                    break
                if opp_pts >= 4 and opp_pts - player_pts >= 2:
                    opp_games += 1
                    break
            server_is_player = not server_is_player

            if player_games >= 6 and player_games - opp_games >= 2:
                player_sets += 1
                break
            if opp_games >= 6 and opp_games - player_games >= 2:
                opp_sets += 1
                break
            if player_games == 6 and opp_games == 6:
                # Tiebreak: first to 7, win-by-2, using the average of spw/rpw.
                p_tb = (spw + rpw) / 2
                tb_player = 0
                tb_opp = 0
                while True:
                    if rng.random() < p_tb:
                        tb_player += 1
                    else:
                        tb_opp += 1
                    if tb_player >= 7 and tb_player - tb_opp >= 2:
                        player_sets += 1
                        break
                    if tb_opp >= 7 and tb_opp - tb_player >= 2:
                        opp_sets += 1
                        break
                break

    return 1 if player_sets == sets_to_win else 0


# --- Sim B over one half of a player's career ---
def sim_b_half(matches_df, n_sims=200, seed=42):
    rng = np.random.default_rng(seed)
    spw_arr = matches_df["spw_match"].values / 100.0
    rpw_arr = matches_df["rpw_match"].values / 100.0
    bo5_arr = (matches_df["best_of"].values == 5)

    n_matches = len(matches_df)
    sim_mwps = np.zeros(n_sims)

    for sim_idx in range(n_sims):
        wins = 0
        for i in range(n_matches):
            wins += simulate_match(spw_arr[i], rpw_arr[i], bo5_arr[i], rng)
        sim_mwps[sim_idx] = wins / n_matches * 100

    return sim_mwps.mean()


# --- Persistence pipeline per tour ---
def persistence_pipeline(panel, tour_name, min_per_half=50, n_sims=200):
    print(f"\n-- {tour_name} --")

    panel = panel.dropna(subset=["spw_match", "rpw_match", "date", "best_of"])
    panel = panel[(panel["spw_match"] >= 0) & (panel["spw_match"] <= 100)]
    panel = panel[(panel["rpw_match"] >= 0) & (panel["rpw_match"] <= 100)]

    results = []
    players = panel["player"].unique()
    print(f"  Total players in panel: {len(players)}")

    for i, player in enumerate(players):
        if (i + 1) % 50 == 0:
            print(f"    Processing player {i+1}/{len(players)}...")

        player_df = panel[panel["player"] == player].sort_values("date").reset_index(drop=True)
        n_matches = len(player_df)

        if n_matches < 2 * min_per_half:
            continue

        mid = n_matches // 2
        early = player_df.iloc[:mid]
        late = player_df.iloc[mid:]

        obs_mwp_early = early["won_match"].mean() * 100
        obs_mwp_late = late["won_match"].mean() * 100

        # Different seeds per half so the two simulations are independent.
        sim_mwp_early = sim_b_half(early, n_sims=n_sims, seed=42)
        sim_mwp_late = sim_b_half(late, n_sims=n_sims, seed=137)

        results.append({
            "player": player,
            "n_early": len(early),
            "n_late": len(late),
            "obs_mwp_early": obs_mwp_early,
            "sim_mwp_early": sim_mwp_early,
            "clutch_early": obs_mwp_early - sim_mwp_early,
            "obs_mwp_late": obs_mwp_late,
            "sim_mwp_late": sim_mwp_late,
            "clutch_late": obs_mwp_late - sim_mwp_late,
        })

    return pd.DataFrame(results)


print("\n" + "=" * 60)
print("  Running Sim B per half (takes a few minutes)...")
print("=" * 60)

atp_results = persistence_pipeline(atp_panel, "ATP", min_per_half=50, n_sims=200)
wta_results = persistence_pipeline(wta_panel, "WTA", min_per_half=50, n_sims=200)

print(f"\nATP qualifying players: {len(atp_results)}")
print(f"WTA qualifying players: {len(wta_results)}")

atp_results.to_csv(OUTPUT_DIR / "persistence_atp.csv", index=False)
wta_results.to_csv(OUTPUT_DIR / "persistence_wta.csv", index=False)


# --- Persistence regression: does early clutch predict late clutch? ---
print("\n" + "=" * 60)
print("  PERSISTENCE REGRESSION")
print("=" * 60)

def run_persistence(df, label):
    x = df["clutch_early"].values
    y = df["clutch_late"].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r2 = r_value ** 2
    n = len(df)
    t_crit = stats.t.ppf(0.975, n - 2)
    ci_low = slope - t_crit * std_err
    ci_high = slope + t_crit * std_err

    print(f"\n  {label}:")
    print(f"    n = {n}")
    print(f"    beta (clutch_early -> clutch_late) = {slope:.4f}")
    print(f"    SE = {std_err:.4f}")
    print(f"    95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"    p-value = {p_value:.4f}")
    print(f"    R2 = {r2:.4f}")

    if p_value < 0.05 and slope > 0.1:
        verdict = "PERSISTENT - significant skill persistence"
    elif p_value < 0.05:
        verdict = "WEAKLY PERSISTENT - significant but small"
    else:
        verdict = "NOT PERSISTENT - consistent with noise"
    print(f"    Verdict: {verdict}")

    return slope, intercept, p_value, r2, n

atp_slope, atp_int, atp_p, atp_r2, atp_n = run_persistence(atp_results, "ATP")
wta_slope, wta_int, wta_p, wta_r2, wta_n = run_persistence(wta_results, "WTA")


# --- Persistence plot ---
print("\nGenerating plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")
fig.suptitle("Clutch Persistence - Does early-career clutch predict late-career clutch?",
             fontsize=13, fontweight="bold")

for ax, (df, label, slope, intercept, p, r2, n, colour) in zip(
    axes,
    [(atp_results, "ATP (men)", atp_slope, atp_int, atp_p, atp_r2, atp_n, "#2563EB"),
     (wta_results, "WTA (women)", wta_slope, wta_int, wta_p, wta_r2, wta_n, "#F97316")]):

    x = df["clutch_early"].values
    y = df["clutch_late"].values

    ax.scatter(x, y, s=30, alpha=0.5, color=colour, linewidths=0)

    x_line = np.linspace(x.min(), x.max(), 100)
    # "p = 0.000" is never true; report a bound instead.
    p_txt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.plot(x_line, slope * x_line + intercept, color="black", lw=2,
            label=f"beta = {slope:.3f}, {p_txt}")

    lims = [min(x.min(), y.min()) - 1, max(x.max(), y.max()) + 1]
    ax.plot(lims, lims, "k--", alpha=0.3, lw=1, label="y = x")

    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.5)

    ax.set_xlabel("Early-career clutch score (pp)", fontsize=11)
    ax.set_ylabel("Late-career clutch score (pp)", fontsize=11)
    ax.set_title(f"{label}   n={n}   R2 = {r2:.3f}", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "persistence_test.png", dpi=160, bbox_inches="tight")
print(f"  Plot saved -> {OUTPUT_DIR / 'persistence_test.png'}")
plt.close('all')


# ============================================================
#   PHASE 6 - PLAYER-LEVEL CROSS-VALIDATION
#
#   Runs alongside Phase 2's Step 8 as a stronger cross-check . Reuses
#   results_table and step11_table so CV numbers sit next to the in-sample ones.
#
#   Why this matters for the headline finding: Phase 3 found GAM's dR2(bpw) for
#   ATP was +0.026, ~3x the linear +0.009 -- but Phase 3 flagged GAM's edof as
#   high vs n, so that gain could be partly overfitting. In-sample R2 can't tell
#   the difference; only a held-out test can. This is that test.
# ============================================================

# --- Part A: single predictor (pw), for the four Phase 2 baselines ---
# Reuses the original univariate sigmoid(), not the multivariate single-index
# version -- attaching a free weight to a single feature makes k and the weight
# unidentifiable (many (k, weight) pairs give the same fit) and destabilises
# curve_fit for no benefit.
def _fit_predict_linear_uni(pw_tr, mwp_tr, pw_te):
    m, c = np.polyfit(pw_tr, mwp_tr, 1)
    return m * pw_te + c


def _fit_predict_sigmoid_uni(pw_tr, mwp_tr, pw_te):
    try:
        params, _ = curve_fit(sigmoid, pw_tr, mwp_tr,
                               p0=[65, 0.55, pw_tr.mean(), 28],
                               maxfev=20000,
                               bounds=([20, 0.01, 45, 0], [100, 5.0, 58, 50]))
        return sigmoid(pw_te, *params)
    except RuntimeError:
        return None


def _fit_predict_poly3_uni(pw_tr, mwp_tr, pw_te):
    centre = pw_tr.mean()
    coeffs = np.polyfit(pw_tr - centre, mwp_tr, 3)
    return np.polyval(coeffs, pw_te - centre)


def _fit_predict_gam_uni(pw_tr, mwp_tr, pw_te):
    gam = LinearGAM(s(0)).fit(pw_tr.reshape(-1, 1), mwp_tr)
    return gam.predict(pw_te.reshape(-1, 1))


def player_cv_r2_uni(df, model_type, n_splits=5, seed=42):
    # 5-fold CV across players. Each player gets one out-of-fold prediction;
    # R2 is pooled over all of them (not averaged per fold), which avoids
    # weighting issues when fold sizes differ.
    pw = df['pw'].values
    mwp = df['mwp'].values
    n = len(df)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.full(n, np.nan)

    fit_fns = {
        'linear': _fit_predict_linear_uni,
        'sigmoid': _fit_predict_sigmoid_uni,
        'poly3': _fit_predict_poly3_uni,
        'gam': _fit_predict_gam_uni,
    }
    fn = fit_fns[model_type]

    for train_idx, test_idx in kf.split(pw):
        preds = fn(pw[train_idx], mwp[train_idx], pw[test_idx])
        if preds is None:
            return None
        oof_preds[test_idx] = preds

    if np.isnan(oof_preds).any():
        return None
    return r_squared(mwp, oof_preds)


# --- Part B: multi-predictor CV, for Phase 3/4 feature sets ---
# Every transform that could leak test info (poly centering means, sigmoid
# standardisation stats) is fit on the TRAIN fold only and reused on test.
# GAM needs no freezing -- pyGAM's .predict() uses the already-fitted basis.
def _cv_fit_linear(train_dict, y_train, interaction_pair=None):
    cols = list(train_dict.values())
    if interaction_pair:
        a, b = interaction_pair
        cols = cols + [train_dict[a] * train_dict[b]]
    X = np.column_stack(cols)
    return _LinReg().fit(X, y_train)


def _cv_predict_linear(model, test_dict, interaction_pair=None):
    cols = list(test_dict.values())
    if interaction_pair:
        a, b = interaction_pair
        cols = cols + [test_dict[a] * test_dict[b]]
    X = np.column_stack(cols)
    return model.predict(X)


def _cv_fit_poly(train_dict, y_train, degree=3, interaction_pair=None):
    means = {k: v.mean() for k, v in train_dict.items()}   # frozen on TRAIN
    cols = []
    for name, arr in train_dict.items():
        c = arr - means[name]
        for d in range(1, degree + 1):
            cols.append(c ** d)
    if interaction_pair:
        a, b = interaction_pair
        cols.append((train_dict[a] - means[a]) * (train_dict[b] - means[b]))
    X = np.column_stack(cols)
    model = _LinReg().fit(X, y_train)
    return (model, means)


def _cv_predict_poly(state, test_dict, degree=3, interaction_pair=None):
    model, means = state
    cols = []
    for name, arr in test_dict.items():
        c = arr - means[name]           # reuse TRAIN means, not test means
        for d in range(1, degree + 1):
            cols.append(c ** d)
    if interaction_pair:
        a, b = interaction_pair
        cols.append((test_dict[a] - means[a]) * (test_dict[b] - means[b]))
    X = np.column_stack(cols)
    return model.predict(X)


def _cv_fit_gam(train_dict, y_train, interaction_pair_idx=None):
    # Same dict-insertion-order dependency as fit_gam_interaction (Phase 3).
    names = list(train_dict.keys())
    X = np.column_stack([train_dict[n] for n in names])
    if interaction_pair_idx:
        i, j = interaction_pair_idx
        term = te(i, j)
        for idx in range(len(names)):
            if idx not in (i, j):
                term = term + s(idx)
    else:
        term = s(0)
        for idx in range(1, len(names)):
            term = term + s(idx)
    gam = LinearGAM(term).fit(X, y_train)
    return (gam, names)


def _cv_predict_gam(state, test_dict):
    gam, names = state
    X = np.column_stack([test_dict[n] for n in names])
    return gam.predict(X)


def _cv_fit_sigmoid(train_dict, y_train, interaction_pair=None):
    names = list(train_dict.keys())
    stats = {}
    std_cols = []
    for n in names:
        mu, sigma = train_dict[n].mean(), train_dict[n].std()
        sigma = sigma if sigma > 1e-9 else 1.0
        stats[n] = (mu, sigma)                       # frozen on TRAIN
        std_cols.append((train_dict[n] - mu) / sigma)
    if interaction_pair:
        a, b = interaction_pair
        raw = train_dict[a] * train_dict[b]
        mu, sigma = raw.mean(), raw.std()
        sigma = sigma if sigma > 1e-9 else 1.0
        stats['__interaction__'] = (mu, sigma)
        std_cols.append((raw - mu) / sigma)
    X_flat = np.vstack(std_cols)
    n_features = X_flat.shape[0]
    w0 = [1.0 / n_features] * n_features
    p0 = w0 + [65, 1.0, 0.0, 28]
    lower = [-20] * n_features + [20, 0.001, -10, 0]
    upper = [20] * n_features + [100, 10.0, 10, 50]
    try:
        params, _ = curve_fit(_sigmoid_index, X_flat, y_train, p0=p0,
                               maxfev=50000, bounds=(lower, upper))
        return (params, stats, names, interaction_pair)
    except RuntimeError:
        return None


def _cv_predict_sigmoid(state, test_dict):
    if state is None:
        return None
    params, stats, names, interaction_pair = state
    cols = []
    for n in names:
        mu, sigma = stats[n]
        cols.append((test_dict[n] - mu) / sigma)      # reuse TRAIN stats
    if interaction_pair:
        a, b = interaction_pair
        mu, sigma = stats['__interaction__']
        raw = test_dict[a] * test_dict[b]
        cols.append((raw - mu) / sigma)
    X_flat = np.vstack(cols)
    return _sigmoid_index(X_flat, *params)


def player_cv_r2_multivar(feature_dict, y, family, n_splits=5, seed=42,
                           degree=3, interaction_pair=None, gam_interaction=False):
    # Multi-predictor version of player_cv_r2_uni. gam_interaction=True uses a
    # tensor smooth on indices (0,1) -- same convention as Phase 3: the
    # interacting pair must be the first two keys inserted.
    names = list(feature_dict.keys())
    n = len(next(iter(feature_dict.values())))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.full(n, np.nan)
    idx_all = np.arange(n)

    for train_idx, test_idx in kf.split(idx_all):
        train_dict = {k: v[train_idx] for k, v in feature_dict.items()}
        test_dict = {k: v[test_idx] for k, v in feature_dict.items()}
        y_train = y[train_idx]

        if family == 'linear':
            model = _cv_fit_linear(train_dict, y_train, interaction_pair)
            preds = _cv_predict_linear(model, test_dict, interaction_pair)
        elif family == 'poly':
            state = _cv_fit_poly(train_dict, y_train, degree, interaction_pair)
            preds = _cv_predict_poly(state, test_dict, degree, interaction_pair)
        elif family == 'gam':
            pair_idx = (0, 1) if gam_interaction else None
            state = _cv_fit_gam(train_dict, y_train, pair_idx)
            preds = _cv_predict_gam(state, test_dict)
        elif family == 'sigmoid':
            state = _cv_fit_sigmoid(train_dict, y_train, interaction_pair)
            preds = _cv_predict_sigmoid(state, test_dict)
        else:
            raise ValueError(family)

        if preds is None:
            return None
        oof_preds[test_idx] = preds

    if np.isnan(oof_preds).any():
        return None
    return r_squared(y, oof_preds)


FAMILY_LABEL_TO_KEY = {
    "Linear": "linear",
    "Polynomial (deg 3)": "poly",
    "GAM": "gam",
    "Sigmoid": "sigmoid",
}


# --- Part A run: complements Step 8 ---
print()
print("=" * 90)
print("  PHASE 6A - PLAYER-LEVEL 5-FOLD CV (complements Phase 2 Step 8)")
print("=" * 90)

for tour_name, df in [("ATP (men)", MEN), ("WTA (women)", WOMEN)]:
    print(f"\n  -- {tour_name} --")
    for mtype, label in [('linear', 'Linear'), ('sigmoid', 'Sigmoid'),
                          ('poly3', 'Polynomial (deg 3)'), ('gam', 'GAM')]:
        cv_r2 = player_cv_r2_uni(df, mtype, n_splits=5, seed=42)
        in_sample_r2 = results[tour_name][f"r2_{'lin' if mtype=='linear' else ('sig' if mtype=='sigmoid' else ('poly' if mtype=='poly3' else 'gam'))}"]
        if cv_r2 is not None:
            gap = in_sample_r2 - cv_r2
            print(f"  {label:<22}  in-sample R2={in_sample_r2:.4f}   CV R2={cv_r2:.4f}   "
                  f"gap={gap:+.4f}{'  <- overfit risk' if gap > 0.01 else ''}")
        else:
            print(f"  {label:<22}  CV fit failed on a fold")


# --- Part B run: validates Phase 3's clutch proxy finding ---
print()
print("=" * 90)
print("  PHASE 6B - 5-FOLD CV FOR PHASE 3 (R3-style vs R4-style)")
print("=" * 90)

cv10_rows = []
for tour_name, df in [("ATP (men)", MEN), ("WTA (women)", WOMEN)]:
    spw, rpw, bpw, mwp = df['spw'].values, df['rpw'].values, df['bpw'].values, df['mwp'].values
    FS3 = {'spw': spw, 'rpw': rpw}
    FS4 = {'spw': spw, 'rpw': rpw, 'bpw': bpw}

    print(f"\n  -- {tour_name} --")
    for family_label, family_key in FAMILY_LABEL_TO_KEY.items():
        for combo in ["additive", "interaction"]:
            inter = ('spw', 'rpw') if combo == "interaction" else None
            gam_inter = (combo == "interaction")

            cv_r2_3 = player_cv_r2_multivar(FS3, mwp, family_key, interaction_pair=inter, gam_interaction=gam_inter)
            cv_r2_4 = player_cv_r2_multivar(FS4, mwp, family_key, interaction_pair=inter, gam_interaction=gam_inter)

            row_match = results_table[(results_table.tour == tour_name) &
                                       (results_table.family == family_label) &
                                       (results_table.combo == combo)]
            in_r2_3 = row_match['r2_r3'].values[0]
            in_r2_4 = row_match['r2_r4'].values[0]

            if cv_r2_3 is not None and cv_r2_4 is not None:
                cv_delta_bpw = cv_r2_4 - cv_r2_3
                gap4 = in_r2_4 - cv_r2_4
                print(f"  {family_label:<20} {combo:<12}  "
                      f"in-sample dR2(bpw)={in_r2_4-in_r2_3:+.4f}   CV dR2(bpw)={cv_delta_bpw:+.4f}   "
                      f"R4 gap={gap4:+.4f}{'  <- shrank OOS' if gap4 > 0.01 else ''}")
            else:
                cv_delta_bpw, gap4 = None, None
                print(f"  {family_label:<20} {combo:<12}  CV fit failed on a fold")

            cv10_rows.append(dict(
                tour=tour_name, family=family_label, combo=combo,
                in_sample_r2_r3=in_r2_3, in_sample_r2_r4=in_r2_4,
                cv_r2_r3=cv_r2_3, cv_r2_r4=cv_r2_4, cv_delta_r2_bpw=cv_delta_bpw,
                overfit_gap_r4=gap4,
            ))

cv10_table = pd.DataFrame(cv10_rows)
cv10_table.to_csv(OUTPUT_DIR / "step12b_cv_clutch_proxy.csv", index=False)
print(f"\n  Saved -> {OUTPUT_DIR / 'step12b_cv_clutch_proxy.csv'}")


# --- Part C run: validates Phase 4's parsimony verdict ---
print()
print("=" * 90)
print("  PHASE 6C - 5-FOLD CV FOR PHASE 4 (pw+bpw vs spw+rpw+bpw)")
print("=" * 90)

cv11_rows = []
for tour_name, df in [("ATP (men)", MEN), ("WTA (women)", WOMEN)]:
    pw, spw, rpw, bpw, mwp = df['pw'].values, df['spw'].values, df['rpw'].values, df['bpw'].values, df['mwp'].values
    FS2 = {'pw': pw, 'bpw': bpw}
    FS4 = {'spw': spw, 'rpw': rpw, 'bpw': bpw}

    print(f"\n  -- {tour_name} --")
    for family_label, family_key in FAMILY_LABEL_TO_KEY.items():
        for combo in ["additive", "interaction"]:
            inter2 = ('pw', 'bpw') if combo == "interaction" else None
            inter4 = ('spw', 'rpw') if combo == "interaction" else None
            gam_inter = (combo == "interaction")

            cv_r2_2 = player_cv_r2_multivar(FS2, mwp, family_key, interaction_pair=inter2, gam_interaction=gam_inter)
            cv_r2_4 = player_cv_r2_multivar(FS4, mwp, family_key, interaction_pair=inter4, gam_interaction=gam_inter)

            if cv_r2_2 is not None and cv_r2_4 is not None:
                cv_winner = "R2-style (pw+bpw)" if cv_r2_2 > cv_r2_4 else "R4-style (spw+rpw+bpw)"
                print(f"  {family_label:<20} {combo:<12}  "
                      f"CV R2: pw+bpw={cv_r2_2:.4f}   spw+rpw+bpw={cv_r2_4:.4f}   "
                      f"-> better OOS: {cv_winner}")
            else:
                cv_winner = None
                print(f"  {family_label:<20} {combo:<12}  CV fit failed on a fold")

            cv11_rows.append(dict(
                tour=tour_name, family=family_label, combo=combo,
                cv_r2_pw_bpw=cv_r2_2, cv_r2_spw_rpw_bpw=cv_r2_4,
                better_fit_by_cv=cv_winner,
            ))

cv11_table = pd.DataFrame(cv11_rows)
cv11_table.to_csv(OUTPUT_DIR / "step12c_cv_parsimony.csv", index=False)
print(f"\n  Saved -> {OUTPUT_DIR / 'step12c_cv_parsimony.csv'}")

n_cv_favor_2 = (cv11_table["better_fit_by_cv"] == "R2-style (pw+bpw)").sum()
n_cv_favor_4 = (cv11_table["better_fit_by_cv"] == "R4-style (spw+rpw+bpw)").sum()

print()
print("=" * 90)
print("  PHASE 6 - NOTES")
print("=" * 90)
print(f"""
  Out-of-sample parsimony verdict: {n_cv_favor_2}/{len(cv11_table)} comparisons
  favour pw+bpw on held-out players (vs the in-sample AIC verdict from Phase 4).
  If they roughly agree, the parsimony conclusion is trustworthy, not an AIC
  penalty-term artefact.

  For the clutch proxy (6B): a small or negative overfit gap means the bpw
  improvement generalises to new players. A large positive gap (in-sample R2
  well above CV R2) means that family was fitting training-player noise -- treat
  its in-sample dR2(bpw) with real scepticism, especially GAM, the family
  already flagged in Phase 3.
""")
print("=" * 90)