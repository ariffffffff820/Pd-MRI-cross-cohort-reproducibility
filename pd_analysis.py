#!/usr/bin/env python3
"""
Parkinson's MRI project: cross-cohort analysis on FastSurfer regional volumes.

What it runs (all on CPU, no GPU needed):
  1. Baselines (sex only, age only, age + sex) and logistic-regression models on
     subcortical, cortical and all-structure volumes.
     Internal: repeated stratified 5-fold CV within each cohort.
     External: train on one cohort, test on the other, both directions.
     The test cohort is never used for tuning, scaling or thresholds.
  2. Cohort-identity check (can the volumes tell Neurocon from Taowu?).
  3. ComBat harmonisation arm (fitted WITHOUT diagnosis; age and sex preserved).
  4. Replication of PD-vs-control effect sizes between the two cohorts.

Usage in a Kaggle notebook (Internet ON is only needed if neuroCombat is not installed):
    !python -u /kaggle/input/<your-code-dataset>/pd_analysis.py
    !python -u /kaggle/input/<your-code-dataset>/pd_analysis.py --exclude sub-patient032037 sub-patient032078
Quick test:  add  --reps 1 --nboot 50 --nperm 50
Outputs are written to --out (default: current directory, /kaggle/working on Kaggle).
"""
import argparse, os, sys, subprocess, warnings, platform, datetime
import numpy as np, pandas as pd
from scipy import stats
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--vol", default="/kaggle/input/datasets/arifkhan820/results-new-pd/pd_volumes/volumes.csv")
ap.add_argument("--demo", default="/kaggle/input/datasets/arifkhan820/sameeeeee/subject_table_with_demographics.csv")
ap.add_argument("--out", default=".")
ap.add_argument("--exclude", nargs="*", default=[], help="subject IDs to drop (sensitivity run)")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--nboot", type=int, default=1000)
ap.add_argument("--nperm", type=int, default=2000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--skip-combat", action="store_true")
args, _ = ap.parse_known_args()      # ignores extra arguments injected by Jupyter
REPS, NBOOT, NPERM, SEED = args.reps, args.nboot, args.nperm, args.seed
TAG = "_excluded" if args.exclude else ""
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------- data
v = pd.read_csv(args.vol)
d = pd.read_csv(args.demo)[["cohort", "subject", "age", "sex"]].rename(columns={"subject": "sid"})
df = v.merge(d, on=["cohort", "sid"], validate="one_to_one")
df = df[~df.sid.isin(args.exclude)].reset_index(drop=True)
y, coh = df.label.values, df.cohort.values
age, male = df.age.values.astype(float), (df.sex == "M").values.astype(float)
print("subjects:", len(df), df.groupby(["cohort", "label"]).size().to_dict(), "| excluded:", args.exclude)
assert df[["age", "sex"]].notna().all().all(), "missing age/sex after merge"

META = {"subject", "cohort", "sid", "label", "age", "sex"}
GLOBAL = {"Mask", "BrainSeg", "BrainSegNotVent", "SupraTentorial", "SupraTentorialNotVent", "SubCortGray",
          "rhCerebralWhiteMatter", "lhCerebralWhiteMatter", "CerebralWhiteMatter", "CSF"}
cols = [c for c in df.columns if c not in META]
SUB = [f"{s}-{n}" for n in ["Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala",
       "Accumbens-area", "VentralDC", "Lateral-Ventricle"] for s in ("Left", "Right")] + ["Brain-Stem", "3rd-Ventricle"]
CTX = [c for c in cols if c.startswith("ctx-")]
# empty regions (e.g. corpus callosum when that module is skipped) are removed
ALL = [c for c in cols if c not in GLOBAL and all(df[df.cohort == k][c].std() > 0 for k in ("neurocon", "taowu"))]
vol = lambda fs: (df[fs].div(df["Mask"], axis=0) * 1000).values      # head-size normalised (mask proxy)
G_LOW, G_HIGH = [0.01, 0.1, 1], [1e-4, 1e-3, 1e-2, 1e-1]
SETS = {
    "sex only": (male[:, None], None),
    "age only": (age[:, None], None),
    "age + sex": (np.column_stack([age, male]), None),
    "subcortical volumes (20)": (vol(SUB), G_LOW),
    "cortical volumes (62)": (vol(CTX), G_HIGH),
    f"all structures ({len(ALL)})": (vol(ALL), G_HIGH),
}

# ----------------------------------------------------------------------------- helpers
def model(grid):
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(class_weight="balanced", max_iter=5000))])
    if grid is None:
        return pipe
    return GridSearchCV(pipe, {"lr__C": grid}, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc")

def internal(X, grid, c):
    idx = np.where(coh == c)[0]; Xi, yi = X[idx], y[idx]; out = []
    for r in range(REPS):
        p = np.zeros(len(idx))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(Xi, yi):
            p[te] = model(grid).fit(Xi[tr], yi[tr]).predict_proba(Xi[te])[:, 1]
        out.append(roc_auc_score(yi, p))
    return np.mean(out), np.std(out)

def external(X, grid, a, b):
    tr, te = np.where(coh == a)[0], np.where(coh == b)[0]          # test cohort never used for tuning
    p = model(grid).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]; yt = y[te]
    auc, ba = roc_auc_score(yt, p), balanced_accuracy_score(yt, p > 0.5)
    bs = []
    for _ in range(NBOOT):
        i = rng.integers(0, len(te), len(te))
        if len(np.unique(yt[i])) == 2:
            bs.append(roc_auc_score(yt[i], p[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    perm = np.array([roc_auc_score(rng.permutation(yt), p) for _ in range(NPERM)])
    return auc, lo, hi, ba, (np.sum(perm >= auc) + 1) / (NPERM + 1)

def cohort_identity(X):
    yc = (coh == "taowu").astype(int); a = []
    for r in range(REPS):
        p = cross_val_predict(Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.01, max_iter=5000))]),
                              X, yc, cv=StratifiedKFold(5, shuffle=True, random_state=SEED + r), method="predict_proba")[:, 1]
        a.append(roc_auc_score(yc, p))
    return np.mean(a)

# ----------------------------------------------------------------------------- 1. main table
print("\n=== 1. Baselines and volume models ===", flush=True)
rows = []
for name, (X, grid) in SETS.items():
    for c in ["neurocon", "taowu"]:
        m, s = internal(X, grid, c); rows.append((name, f"internal {c}", f"{m:.2f} (sd {s:.2f})", "", ""))
    for a, b in [("neurocon", "taowu"), ("taowu", "neurocon")]:
        auc, lo, hi, ba, pv = external(X, grid, a, b)
        rows.append((name, f"external {a} -> {b}", f"{auc:.2f} [{lo:.2f}, {hi:.2f}]", f"{ba:.2f}", f"{pv:.3f}"))
res = pd.DataFrame(rows, columns=["features", "test", "AUC", "bal.acc", "perm p"])
print(res.to_string(index=False)); res.to_csv(os.path.join(args.out, f"results_table{TAG}.csv"), index=False)

# ----------------------------------------------------------------------------- 2. cohort identity
print("\n=== 2. Cohort identity (AUC for telling Neurocon from Taowu) ===", flush=True)
for name in list(SETS)[3:]:
    print(f"  {name}: {cohort_identity(SETS[name][0]):.2f}")

# ----------------------------------------------------------------------------- 3. ComBat
if not args.skip_combat:
    print("\n=== 3. ComBat harmonisation (fitted without diagnosis; age and sex preserved) ===", flush=True)
    try:
        try:
            from neuroCombat import neuroCombat
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "neuroCombat"])
            from neuroCombat import neuroCombat
        covars = pd.DataFrame({"batch": (coh == "taowu").astype(int), "age": age, "sex": male.astype(int)})
        harmonise = lambda X: neuroCombat(dat=X.T, covars=covars, batch_col="batch",
                                          continuous_cols=["age"], categorical_cols=["sex"])["data"].T
        rows = []
        for name in list(SETS)[3:]:
            if name.startswith("cortical"):
                continue                    # primary and all-structure sets only
            Xraw, grid = SETS[name]; Xh = harmonise(Xraw)
            for tag, X in [("raw", Xraw), ("ComBat", Xh)]:
                r = dict(features=name, data=tag, cohort_identity_AUC=round(cohort_identity(X), 2),
                         internal_neurocon=round(internal(X, grid, "neurocon")[0], 2),
                         internal_taowu=round(internal(X, grid, "taowu")[0], 2))
                for a, b in [("neurocon", "taowu"), ("taowu", "neurocon")]:
                    auc, lo, hi, _, pv = external(X, grid, a, b)
                    r[f"{a[:3]}->{b[:3]}"] = f"{auc:.2f} [{lo:.2f}, {hi:.2f}] p={pv:.2f}"
                rows.append(r)
        cb = pd.DataFrame(rows); print(cb.to_string(index=False))
        cb.to_csv(os.path.join(args.out, f"combat_results{TAG}.csv"), index=False)
    except Exception as exc:
        print("ComBat arm skipped:", repr(exc)[:200])

# ----------------------------------------------------------------------------- 4. effect-size replication
print("\n=== 4. Do PD-vs-control effect sizes replicate between cohorts? ===", flush=True)
def cohen_d(fs, c, labels=None):
    x = df[df.cohort == c]; yy = x.label.values if labels is None else labels
    X = x[fs].div(x["Mask"], axis=0).values * 1000; a, b = X[yy == 1], X[yy == 0]
    sp = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean(0) - b.mean(0)) / sp
lines = []
for name, fs in [("subcortical (20)", SUB), (f"all usable structures ({len(ALL)})", ALL)]:
    dn, dt = cohen_d(fs, "neurocon"), cohen_d(fs, "taowu"); r = stats.spearmanr(dn, dt)[0]
    nulls = np.array([stats.spearmanr(cohen_d(fs, "neurocon", rng.permutation(df[df.cohort == "neurocon"].label.values)),
                                      cohen_d(fs, "taowu", rng.permutation(df[df.cohort == "taowu"].label.values)))[0]
                      for _ in range(NPERM)])
    p = (np.sum(np.abs(nulls) >= abs(r)) + 1) / (len(nulls) + 1)
    lines.append(f"{name}: Spearman r = {r:.2f} | permutation p = {p:.3f} | same sign in {np.mean(np.sign(dn) == np.sign(dt)) * 100:.0f}% of regions")
print("\n".join(lines))
open(os.path.join(args.out, f"replication{TAG}.txt"), "w").write("\n".join(lines) + "\n")

# ----------------------------------------------------------------------------- provenance
with open(os.path.join(args.out, f"run_info{TAG}.txt"), "w") as fh:
    fh.write(f"date: {datetime.datetime.now().isoformat()}\nargs: {vars(args)}\npython: {platform.python_version()}\n"
             f"numpy {np.__version__} | pandas {pd.__version__} | scipy {__import__('scipy').__version__} | sklearn {sklearn.__version__}\n")
print("\nDone. Files written to", os.path.abspath(args.out))
