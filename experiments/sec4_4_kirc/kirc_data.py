#!/usr/bin/env python3
"""
Download and preprocess the TCGA-KIRC cohort of Sec. 4.4.

    python experiments/sec4_4_kirc/kirc_data.py        # writes data/kirc_OS.npz

Steps, as in the paper:

    gene-level STAR counts (UCSC Xena, GDC hub) and overall survival from the
        Pan-Cancer Atlas clinical data resource (Liu et al., Cell 2018)
    primary tumor samples only
    protein-coding genes only (Human Protein Atlas)
    drop genes with a total count below 10
    drop patients with zero follow-up time
    median-of-ratios size factors, log2(K/s + 1)
    center every gene and scale it to unit L2 norm

This gives n = 531 patients, p = 19 620 genes and 175 events. The Human Protein
Atlas list is downloaded fresh, so a later release can move p by a few genes;
--pc-file pins any other list. Downloads are cached in ~/.cache/vd_tcga (set
TCGA_CACHE to move it). No login needed.
"""
# NEW in this repository (Mehler, Koka, Muma, 2026). GPLv3, see NOTICE.
from __future__ import annotations

import argparse
import gzip
import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COHORT = "KIRC"
ENDPOINT = "OS"
PAPER = dict(n=531, p=19620, events=175)

CACHE = Path(os.environ.get("TCGA_CACHE", Path.home() / ".cache" / "vd_tcga"))
GDC = "https://gdc-hub.s3.us-east-1.amazonaws.com/download"
XENA = "https://gdc.xenahubs.net/download"
COUNTS_URLS = [
    (f"{GDC}/TCGA-{COHORT}.star_counts.tsv.gz", f"{COHORT}_star_counts.tsv.gz"),
    (f"{XENA}/TCGA-{COHORT}.star_counts.tsv.gz", f"{COHORT}_star_counts.tsv.gz"),
]
CDR_URLS = [
    ("https://pancanatlas.xenahubs.net/download/"
     "Survival_SupplementalTable_S1_20171025_xena_sp", "pancan_cdr.tsv"),
]
PROBEMAP_URLS = [
    (f"{GDC}/gencode.v36.annotation.gtf.gene.probemap", "probemap.tsv"),
    (f"{GDC}/gencode.v22.annotation.gene.probeMap", "probemap.tsv"),
]
HPA_TSV = "https://www.proteinatlas.org/download/proteinatlas.tsv.zip"


# ==========================================================================
# Download
# ==========================================================================
def fetch(url: str, fname: str, quiet: bool = False) -> Path:
    """Download into CACHE unless the file is already there."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / fname
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if not quiet:
        print(f"  [get] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "vd-tcga/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest


def fetch_any(candidates, quiet: bool = False) -> Path:
    """Try the URLs in order and return the first that works."""
    problems = []
    for url, fname in candidates:
        try:
            return fetch(url, fname, quiet=quiet)
        except Exception as e:  # noqa: BLE001
            problems.append(f"    {url}\n      {type(e).__name__}: {e}")
    raise RuntimeError(
        "None of the URLs could be reached:\n" + "\n".join(problems)
        + f"\n  Download the file from https://xenabrowser.net/datapages/ and put "
        f"it into {CACHE} as '{candidates[0][1]}'.")


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path, "rt")


# ==========================================================================
# Reading
# ==========================================================================
def load_counts(path: Path, verbose: bool = True) -> pd.DataFrame:
    """Raw STAR counts, genes x samples.

    Xena stores log2(count + 1); this is inverted exactly, because the count
    filter works on raw counts. float32 is exact for integers up to 16.7 million.
    """
    with _open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
    dtypes = {c: np.float32 for c in header[1:]}
    with _open(path) as f:
        df = pd.read_csv(f, sep="\t", index_col=0, dtype=dtypes)
    if verbose:
        print(f"[read] {df.shape[0]} genes x {df.shape[1]} samples")

    vals = df.to_numpy(dtype=np.float32, copy=False)
    counts = np.empty(vals.shape, dtype=np.float32)
    block = 4096
    for i in range(0, vals.shape[0], block):
        v = vals[i:i + block].astype(np.float64)
        counts[i:i + block] = np.rint(np.exp2(v) - 1.0).clip(min=0.0)

    idx = pd.Index([str(i).split(".")[0] for i in df.index], name="ensg")
    return pd.DataFrame(counts, index=idx, columns=df.columns)


def load_survival(verbose: bool = True) -> pd.DataFrame:
    """Overall survival of the cohort from the CDR: columns sample, time, event."""
    cdr = pd.read_csv(fetch_any(CDR_URLS, quiet=not verbose), sep="\t", low_memory=False)
    coh_col = next((c for c in cdr.columns if "cancer type" in c.lower()), None)
    if coh_col is not None:
        cdr = cdr[cdr[coh_col].astype(str).str.upper() == COHORT]
    if len(cdr) == 0:
        raise RuntimeError(f"no rows for {COHORT} in the CDR")
    out = cdr[["sample", f"{ENDPOINT}.time", ENDPOINT]].copy()
    out.columns = ["sample", "time", "event"]
    out["sample"] = out["sample"].astype(str)
    out = out.dropna().drop_duplicates("sample")
    out["event"] = out["event"].astype(float).round().astype(int)
    if verbose:
        print(f"[read] CDR {COHORT}/{ENDPOINT}: {len(out)} samples, "
              f"{int(out['event'].sum())} events")
    return out


def short_barcode(x) -> str:
    """TCGA-A1-A0SB-01A-11R-A144-07 -> TCGA-A1-A0SB-01 (project, site, patient, type)."""
    s = str(x)
    return s[:15] if s.startswith("TCGA-") and len(s) > 15 else s


def match_samples(counts: pd.DataFrame, surv: pd.DataFrame):
    """Common samples, exact first, otherwise on the 15-character barcode."""
    common = sorted(set(counts.columns) & set(surv["sample"]))
    if not common:
        counts = counts.rename(columns=short_barcode)
        counts = counts.loc[:, ~counts.columns.duplicated(keep="first")]
        surv = surv.copy()
        surv["sample"] = surv["sample"].map(short_barcode)
        surv = surv.drop_duplicates("sample")
        common = sorted(set(counts.columns) & set(surv["sample"]))
    if not common:
        raise RuntimeError("no common sample IDs between counts and survival")
    return counts[common], surv.set_index("sample").loc[common]


def load_symbols(ensg: pd.Index, verbose: bool = True) -> pd.Series:
    """Ensembl ID -> gene symbol via the Xena probe map."""
    if not str(ensg[0]).startswith("ENSG"):
        return pd.Series(ensg.values, index=ensg)
    pm = pd.read_csv(fetch_any(PROBEMAP_URLS, quiet=not verbose), sep="\t")
    pm["id"] = pm["id"].astype(str).str.split(".").str[0]
    m = pm.drop_duplicates("id").set_index("id")["gene"]
    return m.reindex(ensg)


def protein_coding_set(pc_file: str | None = None, verbose: bool = True) -> set[str]:
    """Protein-coding genes as a set of Ensembl IDs and symbols.

    Default is the Human Protein Atlas; pc_file takes one ID or symbol per line.
    """
    if pc_file:
        return {ln.strip().split(".")[0] for ln in Path(pc_file).read_text().splitlines()
                if ln.strip() and not ln.startswith("#")}
    path = fetch(HPA_TSV, "proteinatlas.tsv.zip", quiet=not verbose)
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".tsv"))
        with z.open(name) as f:
            hpa = pd.read_csv(io.TextIOWrapper(f, "utf-8"), sep="\t", low_memory=False,
                              usecols=lambda c: c in ("Gene", "Ensembl"))
    ids = set(hpa["Ensembl"].dropna().astype(str).str.split(".").str[0])
    ids |= set(hpa["Gene"].dropna().astype(str))
    return ids


# ==========================================================================
# Normalisation
# ==========================================================================
def size_factors(counts: np.ndarray) -> np.ndarray:
    """DESeq2 median-of-ratios size factors; counts is genes x samples."""
    c = np.asarray(counts, dtype=np.float64)
    with np.errstate(divide="ignore"):
        log_c = np.log(c)
    usable = np.all(np.isfinite(log_c), axis=1)
    if usable.sum() < 100:
        lib = c.sum(axis=0)
        return lib / np.median(lib)
    ref = log_c[usable].mean(axis=1, keepdims=True)
    return np.exp(np.median(log_c[usable] - ref, axis=0))


def vst_log2(counts: np.ndarray) -> np.ndarray:
    """log2(K/s + 1) with median-of-ratios size factors s."""
    s = size_factors(counts)
    return np.log2(np.asarray(counts, dtype=np.float64) / s[None, :] + 1.0)


def standardize(X: np.ndarray) -> np.ndarray:
    """Center every column and scale it to unit L2 norm."""
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    norms[norms == 0.0] = 1.0
    return X / norms


def check_h(X: np.ndarray, tol: float = 1e-10, verbose: bool = True) -> bool:
    """Check that every column is centered and has unit norm."""
    m = float(np.abs(X.mean(axis=0)).max())
    nrm = float(np.abs(np.linalg.norm(X, axis=0) - 1.0).max())
    ok = m < tol and nrm < tol
    if verbose:
        print(f"  centered and unit norm: max|mean| = {m:.2e}, "
              f"max|norm - 1| = {nrm:.2e} -> {'ok' if ok else 'FAILED'}")
    return ok


# ==========================================================================
# Result
# ==========================================================================
@dataclass
class CohortData:
    X: np.ndarray        # n x p, centered, unit-norm columns
    time: np.ndarray     # n, days, strictly positive
    event: np.ndarray    # n, 1 = death observed
    genes: np.ndarray    # p, gene symbols
    ensg: np.ndarray     # p, Ensembl IDs
    samples: np.ndarray  # n, TCGA barcodes
    var: np.ndarray | None = None   # p, gene variance before standardization

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def p(self) -> int:
        return self.X.shape[1]

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {} if self.var is None else {"var": self.var}
        np.savez_compressed(path, X=self.X, time=self.time, event=self.event,
                            genes=self.genes, ensg=self.ensg,
                            samples=self.samples, **extra)
        print(f"[save] {path}  ({path.stat().st_size / 1e6:.0f} MB)")
        return path

    @staticmethod
    def load(path) -> "CohortData":
        z = np.load(path, allow_pickle=False)
        return CohortData(X=z["X"], time=z["time"], event=z["event"],
                          genes=z["genes"], ensg=z["ensg"], samples=z["samples"],
                          var=z["var"] if "var" in z.files else None)

    def summary(self) -> str:
        return (f"n={self.n}  p={self.p}  Events={int(self.event.sum())} "
                f"({self.event.mean():.1%})  p/n={self.p / self.n:.1f}")


# ==========================================================================
# Pipeline
# ==========================================================================
def load_cohort(pc_file: str | None = None, min_count_sum: float = 10.0,
                verbose: bool = True) -> CohortData:
    say = print if verbose else (lambda *a, **k: None)

    say(f"\n[1/6] {COHORT}: counts and {ENDPOINT}")
    counts = load_counts(fetch_any(COUNTS_URLS, quiet=not verbose), verbose=verbose)
    surv = load_survival(verbose=verbose)

    # barcode positions 14-15: 01 primary solid tumor, 03 primary blood cancer
    keep = [c for c in counts.columns if str(c)[13:15] in ("01", "03")]
    counts = counts[keep]
    say(f"      primary tumors: {counts.shape[1]} samples")

    say("\n[2/6] matching samples")
    counts, surv = match_samples(counts, surv)
    say(f"      n={counts.shape[1]}  p={counts.shape[0]}")

    say("\n[3/6] protein-coding genes")
    counts = counts.loc[counts.index.isin(protein_coding_set(pc_file, verbose=verbose))]
    say(f"      p={counts.shape[0]}")

    say(f"\n[4/6] dropping genes with total count < {min_count_sum:g}")
    counts = counts.loc[counts.values.sum(axis=1, dtype=np.float64) >= min_count_sum]
    say(f"      p={counts.shape[0]}")

    say("\n[5/6] dropping zero times, variance stabilization")
    ok = surv["time"].values > 0
    counts, surv = counts.loc[:, ok], surv.loc[ok]
    say(f"      n={counts.shape[1]}")
    expr = pd.DataFrame(vst_log2(counts.values), index=counts.index,
                        columns=counts.columns)

    say("\n[6/6] standardizing")
    X = expr.values.T
    ensg = np.asarray(expr.index, dtype=str)
    genes = load_symbols(expr.index, verbose=verbose)
    genes = np.asarray(genes.fillna(pd.Series(ensg, index=expr.index)).values, dtype=str)
    data = CohortData(var=X.var(axis=0), X=standardize(X),
                      time=surv["time"].values.astype(np.float64),
                      event=surv["event"].values.astype(int),
                      genes=genes, ensg=ensg,
                      samples=np.asarray(expr.columns, dtype=str))

    say(f"\n  {COHORT}/{ENDPOINT}   {data.summary()}")
    check_h(data.X, verbose=verbose)
    if (data.n, data.p, int(data.event.sum())) != (PAPER["n"], PAPER["p"], PAPER["events"]):
        say(f"  [warn] the paper cohort is n={PAPER['n']}, p={PAPER['p']}, "
            f"{PAPER['events']} events. A newer Human Protein Atlas release can "
            f"move p by a few genes, anything else should not differ.")
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/kirc_OS.npz")
    ap.add_argument("--pc-file", default=None,
                    help="own list of protein-coding genes, one ID or symbol per line")
    args = ap.parse_args(argv)
    load_cohort(pc_file=args.pc_file).save(args.out)


if __name__ == "__main__":
    main()
