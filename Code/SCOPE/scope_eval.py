"""
scope_eval.py -- shared evaluation helpers used by the SCOPE study.

  split_by_seed       disjoint TRAIN/TEST partition of seeds, stratified by benchmark.
  behavioural_readout output-level metrics under (optional) erasure on held-out seeds:
                      slot-a disambiguated accuracy, and the answer-flip rate across the
                      slot-c demographic sub-variants (a metric that does not use the
                      activation commutator).
"""

import os
import random
import uuid
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import pandas as pd

import config_scope as C
import erase

BEHAV_MAXTOK = int(os.environ.get("SCOPE_BEHAV_MAXTOK", "48"))


def split_by_seed(pairs, frac_train=C.FRAC_TRAIN, seed=C.RANDOM_SEED):
    """Disjoint TRAIN/TEST partition of seeds, stratified by source benchmark."""
    seeds = sorted({p["seed_id"] for p in pairs})
    by_src = defaultdict(list)
    for s in seeds:
        by_src[s.split("_", 1)[0]].append(s)
    rng = random.Random(seed)
    train, test = set(), set()
    for src, ss in sorted(by_src.items()):
        ss = sorted(ss)
        rng.shuffle(ss)
        k = max(1, int(round(len(ss) * frac_train)))
        train |= set(ss[:k])
        test |= set(ss[k:])
    tr = [p for p in pairs if p["seed_id"] in train]
    te = [p for p in pairs if p["seed_id"] in test]
    return tr, te, train, test


def _acc(df):
    ok = df[df["success_flag"] == True]                              # noqa: E712
    if ok.empty:
        return float("nan")
    m = ok.apply(lambda r: str(r["parsed_answer"]).strip().lower() in str(r["gold_answer"]).strip().lower()
                 or str(r["gold_answer"]).strip().lower() in str(r["parsed_answer"]).strip().lower(), axis=1)
    return float(m.mean())


def behavioural_readout(model, tok, cfg, basis, test_seeds, max_tokens=BEHAV_MAXTOK, cap=C.BEHAV_SEED_CAP):
    """Output-level metrics on held-out TEST seeds under (optional) erasure.

    Returns (behav_accuracy, behav_flip_rate, n_acc_seeds, n_flip_seeds).
    """
    from osm_behavioral import evaluate_osm_model
    pentad = pd.read_parquet(C.PENTAD_PATH)
    test_seeds = set(test_seeds)
    acc_src = pentad[(pentad["slot"] == "a") & (pentad["subvariant"] == "surface")
                     & (pentad["seed_id"].isin(test_seeds))].copy()
    flip_src = pentad[(pentad["slot"] == "c") & (pentad["seed_id"].isin(test_seeds))].copy()
    for d in (acc_src, flip_src):
        d.drop(d[d["prompt_text"].astype(str).str.strip() == ""].index, inplace=True)
    if cap:
        keep = list(dict.fromkeys(flip_src["seed_id"].tolist()))[:cap]
        flip_src = flip_src[flip_src["seed_id"].isin(keep)]
        acc_src = acc_src[acc_src["seed_id"].isin(set(keep))]
    rid = f"scope-{uuid.uuid4().hex[:8]}"
    ctx = erase.ErasureContext(model, basis) if basis else nullcontext()
    with ctx:
        acc_df = evaluate_osm_model(cfg, model, tok, acc_src, rid + "a", temperature=0.0,
                                    sample_index=0, max_tokens=max_tokens) if len(acc_src) else pd.DataFrame()
        flip_df = evaluate_osm_model(cfg, model, tok, flip_src, rid + "c", temperature=0.0,
                                     sample_index=0, max_tokens=max_tokens) if len(flip_src) else pd.DataFrame()
    acc = _acc(acc_df) if len(acc_df) else float("nan")
    flip_rate, n_flip = float("nan"), 0
    if len(flip_df):
        ok = flip_df[flip_df["success_flag"] == True]                # noqa: E712
        if len(ok):
            g = ok.groupby("seed_id")["parsed_answer"].apply(
                lambda s: s.astype(str).str.strip().str.lower().nunique())
            g = g[g.index.map(lambda sid: (ok["seed_id"] == sid).sum() >= 2)]
            if len(g):
                flip_rate = float((g > 1).mean())
                n_flip = int(len(g))
    n_acc = int(acc_df["seed_id"].nunique()) if len(acc_df) else 0
    return acc, flip_rate, n_acc, n_flip
