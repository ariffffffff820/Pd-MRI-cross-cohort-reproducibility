#!/usr/bin/env python3
"""
Parkinson's MRI project: data setup, metadata dump, affine registration and QC.

Designed for a Kaggle notebook (Internet ON, Accelerator None).
Everything that must survive is written under BASE (/kaggle/working/pd_project),
which Kaggle keeps as the notebook Output after "Save & Run All (Commit)".

Usage (in a notebook cell):
    !python -u pd_pipeline.py --limit 2     # quick test: 2 scans per cohort
    !python -u pd_pipeline.py               # full run, all 83 scans

Stages (always in this order):
    1. setup     download tarballs, extract T1w + metadata, build master table
    2. metadata  copy *.tsv files, summarise acquisition parameters (JSON sidecars)
    3. register  N4 bias correction + affine registration to MNI152 (resumable)
    4. qc        MI table, robust-z outliers, outlier overlays, contact sheet
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BUCKET = "fcp-indi"
PREFIX = "data/Projects/INDI/umf_pd/"
COHORTS = ["neurocon", "taowu"]
EXPECTED = {("neurocon", 0): 16, ("neurocon", 1): 27,
            ("taowu", 0): 20, ("taowu", 1): 20}
TEMPLATE_URL = ("https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
                "tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz")
EXPECTED_TEMPLATE_SHAPE = (193, 229, 193)
SEED = 42
PROGRESS_COLS = ["cohort", "subject", "label", "mi", "ok", "seconds", "error"]

ON_KAGGLE = os.path.exists("/kaggle/working")
BASE = "/kaggle/working/pd_project" if ON_KAGGLE else os.path.abspath("pd_project")
TMP = "/tmp" if os.path.isdir("/tmp") else BASE
RAW = os.path.join(TMP, "pd_raw")          # scratch, rebuilt each run


def banner(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70, flush=True)


# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------
def ensure_packages():
    needed = {"boto3": "boto3", "nibabel": "nibabel", "ants": "antspyx",
              "skimage": "scikit-image", "tqdm": "tqdm", "pandas": "pandas",
              "matplotlib": "matplotlib"}
    missing = []
    for module, pip_name in needed.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print("Installing:", missing, flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", *missing])


# ----------------------------------------------------------------------------
# Stage 1: setup
# ----------------------------------------------------------------------------
def download_and_extract():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    for name in [f"{c}.tar.gz" for c in COHORTS]:
        archive = os.path.join(TMP, name)
        if not os.path.exists(archive):
            for attempt in range(1, 4):
                try:
                    print(f"Downloading {name} (attempt {attempt})", flush=True)
                    s3.download_file(BUCKET, PREFIX + name, archive)
                    break
                except Exception as exc:
                    print("  download failed:", exc, flush=True)
                    if os.path.exists(archive):
                        os.remove(archive)
                    if attempt == 3:
                        raise
                    time.sleep(5)
        print(f"{name}: {os.path.getsize(archive) / 1e6:.1f} MB", flush=True)

        cohort = name.split(".")[0]
        dest = os.path.join(RAW, cohort)
        if glob.glob(f"{dest}/**/anat/*T1w.nii*", recursive=True):
            continue                                   # already extracted
        os.makedirs(dest, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and (
                ("t1w" in m.name.lower() and (".nii" in m.name.lower()
                                              or m.name.endswith(".json")))
                or m.name.endswith((".tsv", ".csv", ".txt")))]
            print(f"{cohort}: extracting {len(members)} files", flush=True)
            try:
                tar.extractall(path=dest, members=members, filter="data")
            except TypeError:                          # Python < 3.12
                tar.extractall(path=dest, members=members)


def build_master_table():
    import nibabel as nib
    import pandas as pd

    rows = []
    for cohort in COHORTS:
        paths = sorted(glob.glob(f"{RAW}/{cohort}/**/anat/*T1w.nii*", recursive=True))
        for path in paths:
            sid = re.search(r"(sub-[^/_]+)", path).group(1)
            group = re.search(r"sub-(control|patient)", sid)
            if group is None:
                raise ValueError(f"Unrecognised subject ID: {sid}")
            img = nib.load(path)
            zooms = img.header.get_zooms()[:3]
            rows.append(dict(
                cohort=cohort, subject=sid, path=path, shape=img.shape,
                voxel=tuple(round(float(z), 3) for z in zooms),
                label=0 if group.group(1) == "control" else 1))
    df = pd.DataFrame(rows)

    counts = df.groupby(["cohort", "label"]).size()
    print(counts, flush=True)
    found = {(c, int(l)): int(n) for (c, l), n in counts.items()}
    if found != EXPECTED or len(df) != 83:
        raise RuntimeError(f"Counts differ from expected.\nFound: {found}\nExpected: {EXPECTED}")
    df.to_csv(os.path.join(BASE, "master_table.csv"), index=False)
    return df


# ----------------------------------------------------------------------------
# Stage 2: metadata
# ----------------------------------------------------------------------------
def dump_metadata(df):
    import pandas as pd

    meta_dir = os.path.join(BASE, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    # 1) any tsv/csv files shipped with the archives (e.g. *_patients.tsv)
    for cohort in COHORTS:
        files = [f for f in glob.glob(f"{RAW}/{cohort}/**/*", recursive=True)
                 if os.path.isfile(f) and f.lower().endswith((".tsv", ".csv"))]
        print(f"\n{cohort}: {len(files)} table file(s)")
        for f in files:
            shutil.copy(f, os.path.join(meta_dir, f"{cohort}__{os.path.basename(f)}"))
            sep = "\t" if f.lower().endswith(".tsv") else ","
            try:
                t = pd.read_csv(f, sep=sep)
                print(f"--- {os.path.basename(f)}  shape={t.shape}")
                print("columns:", t.columns.tolist())
                print(t.head(8).to_string())
            except Exception as exc:
                print(f"--- {os.path.basename(f)} could not be read: {exc}")

    # 2) acquisition parameters from the T1w JSON sidecars, one row per subject
    rows = []
    for _, r in df.iterrows():
        side = re.sub(r"\.nii(\.gz)?$", ".json", r.path)
        row = dict(cohort=r.cohort, subject=r.subject, label=r.label)
        if os.path.exists(side):
            with open(side) as fh:
                row.update(json.load(fh))
        rows.append(row)
    acq = pd.DataFrame(rows)
    acq.to_csv(os.path.join(meta_dir, "acquisition_params.csv"), index=False)
    print("\nAcquisition parameters (T1w JSON sidecars):")
    for col in [c for c in acq.columns if c not in ("cohort", "subject", "label")]:
        print(f"\n{col}")
        print(acq.groupby("cohort")[col].value_counts(dropna=False).to_string())


# ----------------------------------------------------------------------------
# Stage 3: registration
# ----------------------------------------------------------------------------
def get_template():
    tpl_dir = os.path.join(BASE, "templates")
    os.makedirs(tpl_dir, exist_ok=True)
    path = os.path.join(tpl_dir, "MNI152_T1_1mm.nii.gz")
    if not os.path.exists(path):
        print("Downloading MNI152 template", flush=True)
        urllib.request.urlretrieve(TEMPLATE_URL, path)
    return path


def register_all(df, transform, limit):
    import ants
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    reg_dir = os.path.join(BASE, "registered")
    xfm_dir = os.path.join(BASE, "transforms")
    os.makedirs(reg_dir, exist_ok=True)
    os.makedirs(xfm_dir, exist_ok=True)

    fixed = ants.image_read(get_template())
    print("template shape:", fixed.shape, flush=True)
    if tuple(fixed.shape) != EXPECTED_TEMPLATE_SHAPE:
        print(f"WARNING: expected {EXPECTED_TEMPLATE_SHAPE}", flush=True)

    todo = df
    if limit:
        todo = df.groupby("cohort", group_keys=False).head(limit)
    progress = os.path.join(BASE, "registration_progress.csv")

    for _, r in tqdm(list(todo.iterrows()), total=len(todo)):
        out_img = os.path.join(reg_dir, f"{r.cohort}_{r.subject}_mni.nii.gz")
        out_mat = os.path.join(xfm_dir, f"{r.cohort}_{r.subject}_affine.mat")
        if os.path.exists(out_img) and os.path.exists(out_mat):
            continue                                       # resume: already done
        row = dict(cohort=r.cohort, subject=r.subject, label=r.label,
                   mi=np.nan, ok=False, seconds=np.nan, error="")
        try:
            t0 = time.time()
            moving = ants.n4_bias_field_correction(ants.image_read(r.path))
            reg = ants.registration(fixed=fixed, moving=moving,
                                    type_of_transform=transform, random_seed=SEED)
            warped = reg["warpedmovout"]
            ants.image_write(warped, out_img)
            shutil.copy(reg["fwdtransforms"][0], out_mat)
            row.update(mi=ants.image_similarity(fixed, warped,
                                                metric_type="MattesMutualInformation"),
                       ok=True, seconds=round(time.time() - t0))
        except Exception as exc:                           # log and carry on
            row["error"] = str(exc)[:300]
        pd.DataFrame([row], columns=PROGRESS_COLS).to_csv(
            progress, mode="a", index=False, header=not os.path.exists(progress))


# ----------------------------------------------------------------------------
# Stage 4: quality control
# ----------------------------------------------------------------------------
def norm01(a):
    import numpy as np
    lo, hi = a.min(), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)


def edge_overlay(fixed_np, warped_np, title, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.feature import canny

    t, s = norm01(fixed_np), norm01(warped_np)
    cx, cy, cz = (d // 2 for d in t.shape)
    views = {"axial": (t[:, :, cz], s[:, :, cz]),
             "coronal": (t[:, cy, :], s[:, cy, :]),
             "sagittal": (t[cx, :, :], s[cx, :, :])}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, (ts, ss)) in zip(axes, views.items()):
        edges = np.rot90(canny(ss, sigma=2))
        ax.imshow(np.rot90(ts), cmap="gray")
        ax.imshow(np.ma.masked_where(~edges, edges), cmap="autumn")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close(fig)


def run_qc():
    import ants
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    progress = os.path.join(BASE, "registration_progress.csv")
    if not os.path.exists(progress):
        print("No progress file, nothing to QC.")
        return
    qc_dir = os.path.join(BASE, "qc")
    os.makedirs(qc_dir, exist_ok=True)

    prog = pd.read_csv(progress).drop_duplicates(["cohort", "subject"], keep="last")
    med = prog.mi.median()
    mad = (prog.mi - med).abs().median()
    prog["z"] = (prog.mi - med) / (1.4826 * mad + 1e-12)
    prog.to_csv(os.path.join(BASE, "registration_qc.csv"), index=False)

    print("\nRegistration status:\n", prog.ok.value_counts().to_string())
    print("\nMI by cohort (more negative = better match):")
    print(prog.groupby("cohort").mi.describe().to_string())
    print("\nimages:", len(glob.glob(f"{BASE}/registered/*.nii.gz")),
          "| transforms:", len(glob.glob(f"{BASE}/transforms/*.mat")))

    failed = prog[~prog.ok.astype(bool)]
    if len(failed):
        print("\nFAILED SCANS:\n", failed[["cohort", "subject", "error"]].to_string())
    flagged = prog[(prog.z.abs() > 3) & prog.ok.astype(bool)]
    print("\nFlagged by robust z (|z| > 3), inspect the overlays in qc/:")
    print(flagged[["cohort", "subject", "label", "mi", "z"]].to_string()
          if len(flagged) else "none")

    fixed = ants.image_read(os.path.join(BASE, "templates", "MNI152_T1_1mm.nii.gz"))
    fixed_np = fixed.numpy()

    for _, r in flagged.iterrows():
        f = os.path.join(BASE, "registered", f"{r.cohort}_{r.subject}_mni.nii.gz")
        if os.path.exists(f):
            edge_overlay(fixed_np, ants.image_read(f).numpy(),
                         f"FLAGGED {r.cohort} {r.subject} (z={r.z:.2f})",
                         os.path.join(qc_dir, f"flagged_{r.cohort}_{r.subject}.png"))

    ok_rows = prog[prog.ok.astype(bool)].sort_values(["cohort", "label", "subject"])
    n = len(ok_rows)
    if n:
        cols = 10
        rows = -(-n // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, (_, r) in zip(np.atleast_1d(axes).ravel(), ok_rows.iterrows()):
            f = os.path.join(BASE, "registered", f"{r.cohort}_{r.subject}_mni.nii.gz")
            a = ants.image_read(f).numpy()
            ax.imshow(np.rot90(a[:, :, a.shape[2] // 2]), cmap="gray")
            ax.set_title(f"{r.cohort[:2]} {'PD' if r.label else 'HC'} {r.subject[-3:]}",
                         fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(qc_dir, "contact_sheet.png"), dpi=100)
        plt.close(fig)
        print("saved qc/contact_sheet.png")


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="register only the first N scans per cohort (test run)")
    parser.add_argument("--transform", default="Affine",
                        help="ANTs transform type (Affine or AffineFast)")
    parser.add_argument("--skip-register", action="store_true")
    parser.add_argument("--skip-qc", action="store_true")
    args, _ = parser.parse_known_args()          # ignores Jupyter's extra argv

    os.makedirs(BASE, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    print("Saving outputs to:", BASE, "| on Kaggle:", ON_KAGGLE, flush=True)

    ensure_packages()

    banner("1/4  SETUP: download, extract, master table")
    download_and_extract()
    df = build_master_table()

    banner("2/4  METADATA: tables and acquisition parameters")
    dump_metadata(df)

    if not args.skip_register:
        banner(f"3/4  REGISTRATION ({args.transform}, seed {SEED})")
        register_all(df, args.transform, args.limit)

    if not args.skip_qc:
        banner("4/4  QUALITY CONTROL")
        run_qc()

    banner("DONE. Check that registered/ and transforms/ hold 83 files each "
           "(or 2 per cohort for a --limit test).")


if __name__ == "__main__":
    main()
