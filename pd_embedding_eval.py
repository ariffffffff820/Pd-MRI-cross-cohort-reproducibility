#!/usr/bin/env python3
"""
Parkinson's MRI project: evaluate frozen pretrained embeddings under the pre-declared protocol.

Declared models (nothing added after seeing results):
  M1  BrainIAC frozen embeddings -> logistic regression              (PRIMARY)
  M2  MedicalNet ResNet-18 frozen embeddings -> logistic regression
  M3  subcortical volumes + BrainIAC embeddings, PCA inside the folds -> logistic regression
  M4  volume-only logistic regression (reference)
Baselines in every table: sex only, age only, age + sex.

Rules: no fine-tuning; scaling, PCA and C are chosen by 5-fold CV inside the training cohort only;
the test cohort is used once per model and direction; the Bonferroni threshold is 0.05 / 4 models.

Usage:
  python pd_embedding_eval.py --emb brainiac=/path/embeddings_brainiac.csv
  python pd_embedding_eval.py --emb brainiac=/path/embeddings_brainiac.csv --emb medicalnet=/path/embeddings_medicalnet.csv
Each embedding CSV needs the columns: cohort, sid, and numeric feature columns named Feature_0, Feature_1, ...
Quick test: add --reps 1 --nboot 30 --nperm 30
"""
import argparse, os, warnings, datetime
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--vol", default="/kaggle/input/datasets/arifkhan820/results-new-pd/pd_volumes/volumes.csv")
ap.add_argument("--demo", default="/kaggle/input/datasets/arifkhan820/sameeeeee/subject_table_with_demographics.csv")
ap.add_argument("--emb", action="append", required=True, help="name=path/to/embeddings.csv (repeat for several)")
ap.add_argument("--out", default=".")
ap.add_argument("--exclude", nargs="*", default=[])
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--nboot", type=int, default=1000)
ap.add_argument("--nperm", type=int, default=2000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--n-models", type=int, default=4, help="number of declared models, for the Bonferroni threshold")
args, _ = ap.parse_known_args()
REPS, NBOOT, NPERM, SEED = args.reps, args.nboot, args.nperm, args.seed
ALPHA = 0.05 / args.n_models
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------- data
v = pd.read_csv(args.vol)
d = pd.read_csv(args.demo)[["cohort", "subject", "age", "sex"]].rename(columns={"subject": "sid"})
df = v.merge(d, on=["cohort", "sid"], validate="one_to_one")
embs = {}
for item in args.emb:
    name, path = item.split("=", 1)
    e = pd.read_csv(path)
    fcols = [c for c in e.columns if c.startswith("Feature_")]
    assert fcols, f"{path}: no Feature_* columns"
    m = df[["cohort", "sid"]].merge(e[["cohort", "sid"] + fcols], on=["cohort", "sid"], how="left", validate="one_to_one")
    assert m[fcols].notna().all().all(), f"{name}: missing embeddings for some subjects"
    embs[name] = m[fcols].values
    print(f"embedding '{name}': {len(fcols)} dimensions")
keep = ~df.sid.isin(args.exclude).values
df = df[keep].reset_index(drop=True); embs = {k: x[keep] for k, x in embs.items()}
y, coh = df.label.values, df.cohort.values
age, male = df.age.values.astype(float), (df.sex == "M").values.astype(float)
print("subjects:", len(df), df.groupby(["cohort", "label"]).size().to_dict(), "| excluded:", args.exclude)

SUB = [f"{s}-{n}" for n in ["Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala",
       "Accumbens-area", "VentralDC", "Lateral-Ventricle"] for s in ("Left", "Right")] + ["Brain-Stem", "3rd-Ventricle"]
vol_sub = (df[SUB].div(df["Mask"], axis=0) * 1000).values

# ----------------------------------------------------------------------------- models
def lr_pipe(pca=None):
    steps = [("sc", StandardScaler())]
    if pca: steps.append(("pca", PCA(n_components=pca, random_state=SEED)))
    steps.append(("lr", LogisticRegression(class_weight="balanced", max_iter=5000)))
    return Pipeline(steps)
def make(grid, pca=None):
    if grid is None: return lr_pipe(pca)
    return GridSearchCV(lr_pipe(pca), {"lr__C": grid}, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc")

G_EMB, G_VOL = [1e-4, 1e-3, 1e-2, 1e-1], [0.01, 0.1, 1]
SETS = {"sex only": (male[:, None], None, None), "age only": (age[:, None], None, None),
        "age + sex": (np.column_stack([age, male]), None, None),
        "M4 volumes (20), reference": (vol_sub, G_VOL, None)}
first = True
for name, E in embs.items():
    tag = "M1" if first else "M2"
    SETS[f"{tag} {name} embeddings"] = (E, G_EMB, None)
    first = False
b1 = next(iter(embs))
SETS[f"M3 volumes + {b1} embeddings, PCA(20)"] = (np.column_stack([vol_sub, embs[b1]]), G_VOL, 20)

def internal(X, grid, pca, c):
    idx = np.where(coh == c)[0]; Xi, yi = X[idx], y[idx]; out = []
    for r in range(REPS):
        p = np.zeros(len(idx))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(Xi, yi):
            p[te] = make(grid, pca).fit(Xi[tr], yi[tr]).predict_proba(Xi[te])[:, 1]
        out.append(roc_auc_score(yi, p))
    return np.mean(out), np.std(out)
def external(X, grid, pca, a, b):
    tr, te = np.where(coh == a)[0], np.where(coh == b)[0]            # test cohort used once, never for tuning
    p = make(grid, pca).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]; yt = y[te]
    auc, ba = roc_auc_score(yt, p), balanced_accuracy_score(yt, p > 0.5)
    bs = []
    for _ in range(NBOOT):
        i = rng.integers(0, len(te), len(te))
        if len(np.unique(yt[i])) == 2: bs.append(roc_auc_score(yt[i], p[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    perm = np.array([roc_auc_score(rng.permutation(yt), p) for _ in range(NPERM)])
    return auc, lo, hi, ba, (np.sum(perm >= auc) + 1) / (NPERM + 1)

# ----------------------------------------------------------------------------- pre-checks (reported whatever they show)
print("\n=== Pre-checks on the embeddings ===", flush=True)
yc = (coh == "taowu").astype(int)
for name, E in embs.items():
    ci, sx, ag = [], [], []
    for r in range(REPS):
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.01, max_iter=5000))])
        ci.append(roc_auc_score(yc, cross_val_predict(pipe, E, yc, cv=StratifiedKFold(5, shuffle=True, random_state=SEED + r), method="predict_proba")[:, 1]))
        sx.append(roc_auc_score(male, cross_val_predict(pipe, E, male.astype(int), cv=StratifiedKFold(5, shuffle=True, random_state=SEED + r), method="predict_proba")[:, 1]))
        pa = cross_val_predict(Pipeline([("sc", StandardScaler()), ("rg", Ridge(alpha=1000.0))]), E, age, cv=KFold(5, shuffle=True, random_state=SEED + r))
        ag.append(np.corrcoef(pa, age)[0, 1])
    print(f"  {name}: cohort-identity AUC {np.mean(ci):.2f} | sex AUC {np.mean(sx):.2f} (0.5 = none) | age r {np.mean(ag):.2f}")

# ----------------------------------------------------------------------------- evaluation
print(f"\n=== Evaluation (Bonferroni threshold for a claim: p < {ALPHA:.4f}) ===", flush=True)
rows, ext = [], {}
for name, (X, grid, pca) in SETS.items():
    for c in ["neurocon", "taowu"]:
        m, s = internal(X, grid, pca, c); rows.append((name, f"internal {c}", f"{m:.2f} (sd {s:.2f})", "", ""))
    for a, b in [("neurocon", "taowu"), ("taowu", "neurocon")]:
        auc, lo, hi, ba, pv = external(X, grid, pca, a, b); ext[(name, a)] = (auc, pv)
        rows.append((name, f"external {a} -> {b}", f"{auc:.2f} [{lo:.2f}, {hi:.2f}]", f"{ba:.2f}", f"{pv:.3f}"))
res = pd.DataFrame(rows, columns=["model", "test", "AUC", "bal.acc", "perm p"])
print(res.to_string(index=False))
res.to_csv(os.path.join(args.out, "embedding_results" + ("_excluded" if args.exclude else "") + ".csv"), index=False)

print("\n=== Decision rule: a signal is claimed only if BOTH directions have p < threshold AND beat the age + sex baseline ===")
for name in [n for n in SETS if n.startswith("M")]:
    ok = all(ext[(name, a)][1] < ALPHA and ext[(name, a)][0] > ext[("age + sex", a)][0] for a in ("neurocon", "taowu"))
    print(f"  {name}: {'CLAIM' if ok else 'no claim'}")
open(os.path.join(args.out, "embedding_run_info.txt"), "w").write(f"{datetime.datetime.now().isoformat()}\nargs: {vars(args)}\n")
print("\nDone. Files written to", os.path.abspath(args.out))
