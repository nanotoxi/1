from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DEFAULT_DATA_DIR
from .qa import answer_question

THRESHOLDS_CSV = DEFAULT_DATA_DIR / "thresholds_raw.csv"


NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def extract_number(text: str) -> float | None:
    if not text:
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*%", text)
    if m:
        return float(m.group(1))
    m = NUM_RE.search(text)
    return float(m.group(1)) if m else None


@dataclass(frozen=True)
class FieldSpec:
    name: str
    column: str
    template: str
    tol: float


FIELDS: list[FieldSpec] = [
    FieldSpec(
        name="Toxic %",
        column="Toxic %",
        template="Using ONLY the dataset context: What is the Toxic % for category '{cat}'? Answer with ONLY the number.",
        tol=0.6,
    ),
    FieldSpec(
        name="Zeta_mV_median",
        column="Zeta_mV_median",
        template="Using ONLY the dataset context: What is the median zeta potential (Zeta_mV_median) for category '{cat}'? Answer with ONLY the number in mV.",
        tol=0.6,
    ),
    FieldSpec(
        name="Size_nm_median",
        column="Size_nm_median",
        template="Using ONLY the dataset context: What is the median particle size (Size_nm_median) for category '{cat}'? Answer with ONLY the number in nm.",
        tol=0.6,
    ),
    FieldSpec(
        name="Hydro_size_nm_median",
        column="Hydro_size_nm_median",
        template="Using ONLY the dataset context: What is the median hydrodynamic size (Hydro_size_nm_median) for category '{cat}'? Answer with ONLY the number in nm.",
        tol=0.6,
    ),
    FieldSpec(
        name="Exposure_hrs_median",
        column="Exposure_hrs_median",
        template="Using ONLY the dataset context: What is the median exposure time (Exposure_hrs_median) for category '{cat}'? Answer with ONLY the number in hours.",
        tol=0.6,
    ),
    FieldSpec(
        name="LC50_val_median",
        column="LC50_val_median",
        template="Using ONLY the dataset context: What is the median LC50 value (LC50_val_median) for category '{cat}'? Answer with ONLY the number.",
        tol=1.0,
    ),
    FieldSpec(
        name="IC50_val_median",
        column="IC50_val_median",
        template="Using ONLY the dataset context: What is the median IC50 value (IC50_val_median) for category '{cat}'? Answer with ONLY the number.",
        tol=1.0,
    ),
]


def main() -> None:
    df = pd.read_csv(THRESHOLDS_CSV)
    df = df[df["Category"].notna()].copy()

    # Coerce numeric columns
    for f in FIELDS:
        df[f.column] = pd.to_numeric(df.get(f.column), errors="coerce")

    per_field = int(os.environ.get("RAG_BENCHMARK_PER_FIELD", "20"))
    rng_seed = 20260306
    random.seed(rng_seed)

    out = {"seed": rng_seed, "per_field": per_field, "fields": []}
    total = 0
    total_pass = 0

    for f in FIELDS:
        eligible = df[df[f.column].notna()].copy()
        n = min(per_field, len(eligible))
        sample = eligible.sample(n=n, random_state=rng_seed + hash(f.name) % 10000) if n else eligible

        rows = []
        passed = 0
        for _, row in sample.iterrows():
            cat = str(row["Category"]).strip()
            expected = float(row[f.column])
            q = f.template.format(cat=cat)
            resp = answer_question(q, top_k=8)
            ans = resp.get("answer", "")
            got = extract_number(ans)
            ok = got is not None and abs(got - expected) <= f.tol
            passed += int(ok)
            rows.append(
                {
                    "category": cat,
                    "expected": expected,
                    "answer": ans,
                    "parsed": got,
                    "pass": ok,
                    "answer_source": resp.get("answer_source", ""),
                    "threshold_field": resp.get("threshold_field", ""),
                }
            )

        total += len(rows)
        total_pass += passed
        out["fields"].append(
            {
                "field": f.name,
                "column": f.column,
                "n": len(rows),
                "passed": passed,
                "accuracy_pct": (passed / len(rows) * 100.0) if rows else 0.0,
                "examples": rows[:5],
            }
        )

    out["overall"] = {
        "total": total,
        "passed": total_pass,
        "accuracy_pct": (total_pass / total * 100.0) if total else 0.0,
    }

    if os.environ.get("RAG_BENCHMARK_JSON"):
        print(json.dumps(out, indent=2))
        return

    # Human-readable summary
    print("=" * 60)
    print("RAG Threshold Benchmark – Accuracy by Field")
    print("=" * 60)
    print(f"Total questions: {total}  |  Passed: {total_pass}  |  Overall accuracy: {out['overall']['accuracy_pct']:.1f}%\n")
    for f in out["fields"]:
        print(f"  {f['field']:25} {f['passed']:3}/{f['n']}  ({f['accuracy_pct']:.1f}%)")
    print("=" * 60)
    print("\nSet RAG_BENCHMARK_JSON=1 for full JSON output.")


if __name__ == "__main__":
    main()

