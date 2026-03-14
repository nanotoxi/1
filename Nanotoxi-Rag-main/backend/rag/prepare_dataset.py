"""
Prepare a clean RAG training CSV from the Mumbai Excel dataset.

Reads BOTH sheets:
  - "Classified Data": per-sample rows (nanoparticle, toxicity, size, zeta, etc.)
  - "Thresholds by Category": per-category summary stats (size/zeta/toxicity %)
and produces backend/data/training_data.csv with the columns expected by the RAG indexer:
  id, size, zeta_potential, concentration, composition, coating, toxicity
plus source_sheet so you can filter Classified Data vs Thresholds.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_XLSX = Path(__file__).resolve().parent.parent.parent / "Pravin sir 03_03_26 data set  mumbai- Final.xlsx"
SHEET_CLASSIFIED = "Classified Data"
SHEET_THRESHOLDS = "Thresholds by Category"
DEFAULT_OUT_CSV = Path(__file__).resolve().parent.parent / "data" / "training_data.csv"
DEFAULT_RAW_OUT_CSV = Path(__file__).resolve().parent.parent / "data" / "classified_data_raw.csv"
DEFAULT_THRESHOLDS_RAW_CSV = Path(__file__).resolve().parent.parent / "data" / "thresholds_raw.csv"


def _as_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_first_number(value: object) -> float | None:
    s = _as_str(value)
    if not s:
        return None
    m = _NUM_RE.search(s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _build_classified_df(xlsx_path: Path, raw_out_csv: Path | None) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=SHEET_CLASSIFIED)
    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]
    if raw_out_csv is not None:
        raw_out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_out_csv, index=False)

    def col(name: str) -> str:
        if name in df.columns:
            return name
        raise KeyError(f"Expected column not found in sheet '{SHEET_CLASSIFIED}': {name}")

    category_col = col("Nanoparticle Category")
    toxicity_col = col("Toxicity Classification")
    size_col = col("particle size (nm)")
    zeta_col = col("Zeta potential (mV)")
    surface_chem_col = col("Surface Chemistry")
    in_vitro_candidates = [c for c in df.columns if c.lower().startswith("dosage in vitro")]
    in_vivo_candidates = [c for c in df.columns if c.lower().startswith("dosage in vivo")]
    in_vitro_col = in_vitro_candidates[0] if in_vitro_candidates else None
    in_vivo_col = in_vivo_candidates[0] if in_vivo_candidates else None

    concentration = []
    for _, row in df.iterrows():
        v = _as_str(row.get(in_vitro_col)) if in_vitro_col else ""
        if not v and in_vivo_col:
            v = _as_str(row.get(in_vivo_col))
        concentration.append(v)

    out = df.copy()
    out["size"] = out[size_col].map(_as_str)
    out["size_nm"] = out[size_col].map(parse_first_number)
    out["zeta_potential"] = out[zeta_col].map(_as_str)
    out["zeta_potential_mv"] = out[zeta_col].map(parse_first_number)
    out["concentration"] = concentration
    out["composition"] = out[category_col].map(_as_str)
    out["coating"] = out[surface_chem_col].map(_as_str)
    out["toxicity"] = out[toxicity_col].map(_as_str)
    out["source_sheet"] = SHEET_CLASSIFIED
    return out


def _build_thresholds_df(xlsx_path: Path, thresholds_raw_csv: Path | None) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=SHEET_THRESHOLDS)
    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]
    if thresholds_raw_csv is not None:
        thresholds_raw_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(thresholds_raw_csv, index=False)

    if "Category" not in df.columns:
        return pd.DataFrame()

    def row_size(r):
        parts = []
        if "Size_nm_median" in r.index and pd.notna(r.get("Size_nm_median")):
            parts.append(f"median {r['Size_nm_median']} nm")
        if "Size_nm_min" in r.index and "Size_nm_max" in r.index and pd.notna(r.get("Size_nm_min")) and pd.notna(r.get("Size_nm_max")):
            parts.append(f"range {r['Size_nm_min']}-{r['Size_nm_max']} nm")
        return "; ".join(parts) if parts else "category summary"

    def row_zeta(r):
        if "Zeta_mV_median" in r.index and pd.notna(r.get("Zeta_mV_median")):
            return f"{r['Zeta_mV_median']} mV"
        return "category summary"

    def row_toxicity(r):
        parts = []
        if "Toxic %" in r.index and pd.notna(r.get("Toxic %")):
            parts.append(f"Toxic % {r['Toxic %']}")
        if "Toxic Count" in r.index and "Non-toxic Count" in r.index:
            parts.append(f"Toxic: {r.get('Toxic Count', '')}, Non-toxic: {r.get('Non-toxic Count', '')}")
        return "; ".join(str(p) for p in parts) if parts else "category summary"

    out = pd.DataFrame()
    out["composition"] = df["Category"].map(_as_str)
    out["size"] = df.apply(row_size, axis=1)
    out["size_nm"] = df["Size_nm_median"] if "Size_nm_median" in df.columns else None
    out["zeta_potential"] = df.apply(row_zeta, axis=1)
    out["zeta_potential_mv"] = df["Zeta_mV_median"] if "Zeta_mV_median" in df.columns else None
    out["concentration"] = ""
    out["coating"] = ""
    out["toxicity"] = df.apply(row_toxicity, axis=1)
    out["source_sheet"] = SHEET_THRESHOLDS
    extra = df[[c for c in df.columns if c not in out.columns]]
    if not extra.empty:
        out = pd.concat([out, extra], axis=1)
    return out


def prepare_from_excel(
    xlsx_path: Path,
    sheet_name: str | None = None,
    out_csv: Path = DEFAULT_OUT_CSV,
    raw_out_csv: Path | None = DEFAULT_RAW_OUT_CSV,
    thresholds_raw_csv: Path | None = DEFAULT_THRESHOLDS_RAW_CSV,
    use_both_sheets: bool = True,
) -> dict[str, object]:
    """
    Build training_data.csv from Excel. If use_both_sheets=True (default), uses
    both "Classified Data" and "Thresholds by Category". Otherwise only the
    sheet given by sheet_name (default "Classified Data").
    """
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"Excel file not found: {xlsx_path}")

    if use_both_sheets:
        classified = _build_classified_df(xlsx_path, raw_out_csv)
        thresholds = _build_thresholds_df(xlsx_path, thresholds_raw_csv)
        # Align columns: standard cols first, then union of extras
        standard_cols = [
            "id", "size", "zeta_potential", "concentration", "composition", "coating", "toxicity",
            "size_nm", "zeta_potential_mv", "source_sheet",
        ]
        classified["id"] = [str(i + 1) for i in range(len(classified))]
        if not thresholds.empty:
            thresholds["id"] = [str(len(classified) + i + 1) for i in range(len(thresholds))]
            missing_cols = [c for c in standard_cols if c not in thresholds.columns and c != "id"]
            if missing_cols:
                thresholds = thresholds.assign(**{c: "" for c in missing_cols})
            out = pd.concat([classified, thresholds], ignore_index=True, sort=False)
        else:
            out = classified.copy()
        out["id"] = [str(i + 1) for i in range(len(out))]
        cols_first = [c for c in standard_cols if c in out.columns]
        remaining = [c for c in out.columns if c not in cols_first]
        out = out[cols_first + remaining]
    else:
        sheet = sheet_name or SHEET_CLASSIFIED
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        df = df.dropna(how="all")
        df.columns = [str(c).strip() for c in df.columns]
        if raw_out_csv is not None:
            raw_out_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(raw_out_csv, index=False)
        if sheet == SHEET_CLASSIFIED:
            out = _build_classified_df(xlsx_path, None)
            out["id"] = [str(i + 1) for i in range(len(out))]
            cols_first = [
                "id", "size", "zeta_potential", "concentration", "composition", "coating", "toxicity",
                "size_nm", "zeta_potential_mv", "source_sheet",
            ]
            out["source_sheet"] = SHEET_CLASSIFIED
            remaining = [c for c in out.columns if c not in cols_first]
            out = out[[c for c in cols_first if c in out.columns] + remaining]
        else:
            out = _build_thresholds_df(xlsx_path, thresholds_raw_csv)
            if out.empty:
                raise ValueError(f"Sheet '{sheet}' could not be converted (missing 'Category'?).")
            out["id"] = [str(i + 1) for i in range(len(out))]
            cols_first = [
                "id", "size", "zeta_potential", "concentration", "composition", "coating", "toxicity",
                "size_nm", "zeta_potential_mv", "source_sheet",
            ]
            remaining = [c for c in out.columns if c not in cols_first]
            out = out[[c for c in cols_first if c in out.columns] + remaining]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    n_classified = int((out.get("source_sheet", pd.Series()) == SHEET_CLASSIFIED).sum()) if "source_sheet" in out.columns else len(out)
    n_thresholds = int((out.get("source_sheet", pd.Series()) == SHEET_THRESHOLDS).sum()) if "source_sheet" in out.columns else 0

    return {
        "xlsx": str(xlsx_path),
        "rows": int(len(out)),
        "columns": int(len(out.columns)),
        "rows_classified": n_classified,
        "rows_thresholds": n_thresholds,
        "out_csv": str(out_csv),
        "raw_out_csv": str(raw_out_csv) if raw_out_csv is not None else "",
        "thresholds_raw_csv": str(thresholds_raw_csv) if thresholds_raw_csv is not None else "",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare RAG training_data.csv from Excel (both sheets by default)")
    p.add_argument("--xlsx", type=str, default=str(DEFAULT_XLSX), help="Path to source .xlsx")
    p.add_argument("--sheet", type=str, default=None, help="Single sheet to use (default: both sheets)")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT_CSV), help="Output CSV path")
    p.add_argument("--raw-out", type=str, default=str(DEFAULT_RAW_OUT_CSV), help="Raw Classified Data CSV (set empty to disable)")
    p.add_argument("--thresholds-raw-out", type=str, default=str(DEFAULT_THRESHOLDS_RAW_CSV), help="Raw Thresholds CSV (set empty to disable)")
    p.add_argument("--single-sheet", action="store_true", help="Use only --sheet (default: use both sheets)")
    args = p.parse_args()

    xlsx = Path(args.xlsx).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    raw_out = Path(args.raw_out).expanduser().resolve() if args.raw_out else None
    thresholds_raw = Path(args.thresholds_raw_out).expanduser().resolve() if args.thresholds_raw_out else None

    use_both = not args.single_sheet
    sheet_name = args.sheet if args.single_sheet else (args.sheet or SHEET_CLASSIFIED)
    summary = prepare_from_excel(
        xlsx_path=xlsx,
        sheet_name=sheet_name,
        out_csv=out,
        raw_out_csv=raw_out,
        thresholds_raw_csv=thresholds_raw,
        use_both_sheets=use_both,
    )
    print(pd.Series(summary).to_json(indent=2))


if __name__ == "__main__":
    main()

