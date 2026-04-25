"""
EPD Bull Optimizer Dashboard
Streamlit app that mirrors the logic in separateforheifers.qmd.
Upload an EPD file (CSV / Excel / TSV), tune weights, criteria,
bull caps, and power function — then run the optimizer.
"""

import io
import math
import streamlit as st
import pandas as pd
import polars as pl
import pulp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EPD Bull Optimizer",
    page_icon="🐂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS  (mirrored from the .qmd)
# ─────────────────────────────────────────────────────────────────────────────

ALL_TRAITS = ["PROS","CED","Milk","WW","YW","ADG","HPG","CEM","Marb","STAY",
              "CW","REA","HB","GM","BW","DMI","ME","YG","FAT"]

HIGHER_COLS = ["PROS","CED","WW","YW","ADG","Milk","HPG","CEM","STAY","Marb",
               "CW","REA","HB","GM"]
LOWER_COLS  = ["BW","DMI","ME","YG","FAT"]

DEFAULT_CRITERIA_B = {
    "PROS": 120,
    "CED":  15,
    "Milk": 29,
    "WW":   73,
    "YW":   115,
    "ADG":  0.28,
    "HPG":  13,
    "CEM":  9,
    "Marb": 0.54,
    "STAY": 17,
    "CW":   29,
    "REA":  0.25,
    "HB":   70,
    "GM":   61,
    "BW":  -3,
    "DMI":  0.43,
    "ME":  -2,
    "YG":   0.01,
    "FAT":  0,
}

DEFAULT_TRAIT_WEIGHTS = {
    "PROS": 1.0,
    "CED":  1.5,
    "Milk": 1.3,
    "WW":   0.5,
    "YW":   0.5,
    "ADG":  1.0,
    "HPG":  1.0,
    "CEM":  1.0,
    "Marb": 1.0,
    "STAY": 1.0,
    "CW":   1.0,
    "REA":  1.0,
    "HB":   1.0,
    "GM":   1.0,
    "BW":   1.0,
    "DMI":  1.0,
    "ME":   1.0,
    "YG":   1.0,
    "FAT":  1.0,
}

DEFAULT_COMBINED_COUNT_WEIGHT = 1.0
DEFAULT_BULL_MAX               = 25
DEFAULT_POWER_FUNCTION         = "linear"

POWER_OPTIONS = ["linear", "squared", "cubed", "exponential"]

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────

def _init():
    if "criteria_b"     not in st.session_state:
        st.session_state.criteria_b = DEFAULT_CRITERIA_B.copy()
    if "trait_weights"  not in st.session_state:
        st.session_state.trait_weights = DEFAULT_TRAIT_WEIGHTS.copy()
    if "power_fn"       not in st.session_state:
        st.session_state.power_fn = DEFAULT_POWER_FUNCTION
    if "cc_weight"      not in st.session_state:
        st.session_state.cc_weight = DEFAULT_COMBINED_COUNT_WEIGHT
    if "epd_df"         not in st.session_state:
        st.session_state.epd_df = None
    if "bull_caps"      not in st.session_state:
        st.session_state.bull_caps = {}
    if "result"         not in st.session_state:
        st.session_state.result = None

_init()

# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC  (mirrors .qmd functions exactly)
# ─────────────────────────────────────────────────────────────────────────────

def strip_column_names(df: pl.DataFrame) -> pl.DataFrame:
    return df.rename({col: col.strip() for col in df.columns})


def compute_pct_diff(df: pl.DataFrame, criteria_b: dict, lower_cols: list) -> pl.DataFrame:
    exprs = []
    for trait, benchmark in criteria_b.items():
        if trait not in df.columns:
            continue
        if trait in lower_cols:
            if benchmark == 0:
                raw = -(pl.col(trait).cast(pl.Float64) - benchmark)
            else:
                raw = -(pl.col(trait).cast(pl.Float64) - benchmark) / abs(benchmark) * 100
        else:
            if benchmark == 0:
                raw = pl.col(trait).cast(pl.Float64) - benchmark
            else:
                raw = (pl.col(trait).cast(pl.Float64) - benchmark) / abs(benchmark) * 100
        exprs.append(raw.alias(f"{trait}_pct"))
    return df.with_columns(exprs)


def normalize_pct_diff(df: pl.DataFrame) -> pl.DataFrame:
    pct_cols = [c for c in df.columns if c.endswith("_pct")]
    exprs = []
    for col in pct_cols:
        mean_sq = (df[col].fill_null(0.0) ** 2).mean()
        rms = mean_sq ** 0.5 if (mean_sq is not None and mean_sq > 0) else None
        if rms is None or rms == 0:
            exprs.append(pl.col(col))
        else:
            exprs.append((pl.col(col) / rms).alias(col))
    return df.with_columns(exprs)
 



def compute_pair_score(df: pl.DataFrame, trait_weights: dict, cc_weight: float) -> pl.DataFrame:
    pct_cols = [c for c in df.columns if c.endswith("_pct")]
    if not pct_cols:
        raise ValueError("No _pct columns found — run compute_pct_diff first.")
    score_expr = sum(
        pl.col(c).fill_null(0.0) * trait_weights.get(c.replace("_pct", ""), 1.0)
        for c in pct_cols
    )
    if "combined_count" in df.columns:
        score_expr = score_expr + pl.col("combined_count").fill_null(0.0) * cc_weight
    return df.with_columns(score_expr.alias("pair_score"))


def apply_power(x: float, power_fn: str) -> float:
    if power_fn == "linear":
        return x
    elif power_fn == "squared":
        return math.copysign(x ** 2, x)
    elif power_fn == "cubed":
        return x ** 3
    elif power_fn == "exponential":
        return math.exp(x)
    raise ValueError(f"Unknown power function: {power_fn}")


def generic_solve(df: pl.DataFrame, cow_col: str, bull_col: str,
                  bull_caps: dict, power_fn: str,
                  require_all_assigned: bool = True) -> dict:
    """
    General solver that works for herd / heifer / president modes.
    require_all_assigned=True  → each cow must be assigned (herd / heifer)
    require_all_assigned=False → each cow assigned at most once (president)
    """
    df = df.with_columns(pl.col(bull_col).str.strip_chars())

    pair_scores = {
        (row[cow_col], row[bull_col]): row["pair_score"]
        for row in df.select([cow_col, bull_col, "pair_score"]).to_dicts()
    }

    pairs    = list(pair_scores.keys())
    cow_ids  = sorted({c for c, _ in pairs})
    bull_ids = list(bull_caps.keys())

    scores = {(c, b): apply_power(float(pair_scores[c, b]), power_fn)
              for c, b in pairs}

    prob = pulp.LpProblem("Cattle_Pairing", pulp.LpMaximize)
    x    = {(c, b): pulp.LpVariable(f"x_{c}_{b}", cat="Binary") for c, b in pairs}

    prob += pulp.lpSum(scores[c, b] * x[c, b] for c, b in pairs), "Total_Score"

    for c in cow_ids:
        constraint = pulp.lpSum(x[c, b] for b in bull_ids if (c, b) in x)
        if require_all_assigned:
            prob += (constraint == 1, f"cow_once_{c}")
        else:
            prob += (constraint <= 1, f"cow_once_{c}")

    for b in bull_ids:
        prob += (
            pulp.lpSum(x[c, b] for c in cow_ids if (c, b) in x) <= bull_caps[b],
            f"bull_limit_{b}",
        )

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    assignments = []
    for c, b in pairs:
        val = pulp.value(x[c, b])
        if val is not None and float(val) > 0.5:
            assignments.append({
                "cow":        c,
                "bull":       b,
                "pair_score": round(float(pair_scores[c, b]), 4),
                "score":      round(float(scores[c, b]), 4),
            })

    bull_usage    = {b: sum(1 for a in assignments if a["bull"] == b) for b in bull_ids}
    assigned_cows = {a["cow"] for a in assignments}
    unassigned    = [c for c in cow_ids if c not in assigned_cows]

    return {
        "status":      pulp.LpStatus[prob.status],
        "total_score": round(float(pulp.value(prob.objective) or 0), 4),
        "assignments": sorted(assignments, key=lambda a: -a["score"]),
        "bull_usage":  bull_usage,
        "bull_max":    bull_caps,
        "unassigned":  unassigned,
        "power_fn":    power_fn,
    }

# ─────────────────────────────────────────────────────────────────────────────
# FILE PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_uploaded_file(uploaded_file) -> pl.DataFrame | None:
    fname = uploaded_file.name.lower()
    try:
        if fname.endswith(".csv"):
            df = pl.read_csv(uploaded_file)
        elif fname.endswith((".xlsx", ".xls")):
            df = pl.read_excel(uploaded_file)
        else:
            content = uploaded_file.read().decode("utf-8", errors="replace")
            df = pl.read_csv(io.StringIO(content), separator="\t")
    except Exception as e:
        st.error(f"Could not parse file: {e}")
        return None
    return strip_column_names(df)


def extract_bulls_from_df(df: pl.DataFrame) -> list[str]:
    """
    Pull unique bull IDs from an AnimalID column (format COW-BULL)
    or from a dedicated Bull/Sire column.
    """
    if "AnimalID" in df.columns:
        bull_series = (
            df["AnimalID"]
            .str.split_exact("-", 1)
            .struct.field("field_1")
            .drop_nulls()
            .unique()
        )
        return sorted(bull_series.to_list())
    for candidate in ["Bull", "Sire", "bull", "sire", "BULL", "SIRE"]:
        if candidate in df.columns:
            return sorted(df[candidate].drop_nulls().unique().to_list())
    return []


def clean_epd_df(df: pl.DataFrame, higher_cols: list, lower_cols: list,
                 criteria_b: dict) -> pl.DataFrame:
    """Parse AnimalID → cow / bull, compute higher/lower counts."""
    if "AnimalID" in df.columns:
        df = df.with_columns(
            cow  = pl.col("AnimalID").str.split_exact("-", 1).struct.field("field_0"),
            bull = pl.col("AnimalID").str.split_exact("-", 1).struct.field("field_1"),
        )
        drop_cols = [c for c in ["AnimalID","Sex","BirthDate","BrdCds","DOB"] if c in df.columns]
        df = df.drop(drop_cols)

    present_higher = [c for c in higher_cols if c in df.columns]
    present_lower  = [c for c in lower_cols  if c in df.columns]

    df = df.with_columns(
        higher_count = pl.sum_horizontal(
            [(pl.col(c) >= criteria_b[c]).cast(pl.UInt8) for c in present_higher]
        ),
        lower_count = pl.sum_horizontal(
            [(pl.col(c) <= criteria_b[c]).cast(pl.UInt8) for c in present_lower]
        ),
    ).with_columns(
        combined_count = pl.col("higher_count") + pl.col("lower_count")
    )

    if "cow" in df.columns:
        df = df.with_columns(pl.col("cow").str.strip_chars())
    if "bull" in df.columns:
        df = df.with_columns(pl.col("bull").str.strip_chars())

    return df

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🐂 EPD Optimizer")
    st.markdown("---")

    # ── File upload ──────────────────────────────────────────────────────────
    st.subheader("📂 Upload EPD File")
    uploaded = st.file_uploader(
        "CSV, Excel, or tab-delimited",
        type=["csv", "xlsx", "xls", "txt", "tsv"],
        help="Expected columns: AnimalID (COW-BULL format) + EPD trait columns.",
    )
    if uploaded:
        parsed = parse_uploaded_file(uploaded)
        if parsed is not None:
            st.session_state.epd_df = parsed
            bulls = extract_bulls_from_df(parsed)
            # Initialise caps for newly seen bulls
            for b in bulls:
                if b not in st.session_state.bull_caps:
                    st.session_state.bull_caps[b] = DEFAULT_BULL_MAX
            st.success(f"✅ {len(parsed):,} rows · {len(bulls)} bulls detected")

    st.markdown("---")

    # ── Solver mode ──────────────────────────────────────────────────────────
    st.subheader("⚙️ Solver Mode")
    solver_mode = st.radio(
        "Select mode",
        ["Herd (all assigned)", "Heifer (all assigned)", "President (optional)"],
        index=0,
    )
    require_all = "optional" not in solver_mode.lower()

    st.markdown("---")

    # ── Power function ───────────────────────────────────────────────────────
    st.subheader("🔢 Power Function")
    st.session_state.power_fn = st.selectbox(
        "Shape of scoring curve",
        POWER_OPTIONS,
        index=POWER_OPTIONS.index(st.session_state.power_fn),
        help="Linear = default. Squared / Cubed / Exponential reward top pairs more aggressively.",
    )

    st.markdown("---")

    # ── Combined count weight ────────────────────────────────────────────────
    st.subheader("🔗 Combined Count Weight")
    st.session_state.cc_weight = st.number_input(
        "Weight applied to combined_count bonus",
        min_value=0.0, max_value=10.0,
        value=float(st.session_state.cc_weight),
        step=0.1,
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────

tabs = st.tabs(["📊 Criteria & Weights", "🐄 Bull Caps", "🚀 Run Optimizer", "📈 Results"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Criteria & Weights
# ─────────────────────────────────────────────────────────────────────────────

with tabs[0]:
    st.header("Benchmark (criteria_B) & Trait Weights")
    st.caption(
        "Edit the benchmark value each trait is measured against "
        "and the relative weight given to each trait in the composite score."
    )

    col_reset, _ = st.columns([1, 5])
    with col_reset:
        if st.button("↺ Reset to defaults"):
            st.session_state.criteria_b    = DEFAULT_CRITERIA_B.copy()
            st.session_state.trait_weights = DEFAULT_TRAIT_WEIGHTS.copy()
            st.rerun()

    st.markdown("---")

    # Display as an editable table-like grid
    header_cols = st.columns([2, 1.8, 1.8, 2])
    header_cols[0].markdown("**Trait**")
    header_cols[1].markdown("**Benchmark (criteria_B)**")
    header_cols[2].markdown("**Weight**")
    header_cols[3].markdown("**Direction**")

    for trait in ALL_TRAITS:
        c0, c1, c2, c3 = st.columns([2, 1.8, 1.8, 2])
        c0.markdown(f"`{trait}`")

        new_bench = c1.number_input(
            f"bench_{trait}",
            value=float(st.session_state.criteria_b.get(trait, 0.0)),
            step=0.01,
            format="%.4f",
            label_visibility="collapsed",
            key=f"bench_{trait}",
        )
        st.session_state.criteria_b[trait] = new_bench

        new_wt = c2.number_input(
            f"wt_{trait}",
            value=float(st.session_state.trait_weights.get(trait, 1.0)),
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            format="%.2f",
            label_visibility="collapsed",
            key=f"wt_{trait}",
        )
        st.session_state.trait_weights[trait] = new_wt

        direction = "⬇️ lower is better" if trait in LOWER_COLS else "⬆️ higher is better"
        c3.markdown(direction)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Bull Caps
# ─────────────────────────────────────────────────────────────────────────────

with tabs[1]:
    st.header("Bull Breeding Caps")
    st.caption(
        "Maximum number of cows each bull can be assigned. "
        "Bulls are detected automatically from the uploaded file. "
        "Default cap is **25** if not changed."
    )

    if not st.session_state.bull_caps:
        st.info("Upload an EPD file first — bulls will be auto-detected.")
    else:
        col_reset2, col_zero, _ = st.columns([1, 1, 4])
        with col_reset2:
            if st.button("↺ Reset all caps to 25"):
                for b in st.session_state.bull_caps:
                    st.session_state.bull_caps[b] = DEFAULT_BULL_MAX
                st.rerun()
        with col_zero:
            if st.button("⛔ Set all caps to 0"):
                for b in st.session_state.bull_caps:
                    st.session_state.bull_caps[b] = 0
                st.rerun()

        st.markdown("---")

        # Manual add bull
        with st.expander("➕ Add a bull manually"):
            ma_col1, ma_col2, ma_col3 = st.columns([2, 1, 1])
            new_bull_name = ma_col1.text_input("Bull ID", key="new_bull_name")
            new_bull_cap  = ma_col2.number_input("Cap", min_value=0, value=25, key="new_bull_cap")
            if ma_col3.button("Add"):
                if new_bull_name.strip():
                    st.session_state.bull_caps[new_bull_name.strip()] = new_bull_cap
                    st.rerun()

        st.markdown("---")

        # Grid of bull caps
        bulls = sorted(st.session_state.bull_caps.keys())
        n_cols = 4
        rows = [bulls[i:i+n_cols] for i in range(0, len(bulls), n_cols)]
        for row in rows:
            cols = st.columns(n_cols)
            for i, bull in enumerate(row):
                new_cap = cols[i].number_input(
                    bull,
                    min_value=0,
                    max_value=500,
                    value=int(st.session_state.bull_caps[bull]),
                    step=1,
                    key=f"cap_{bull}",
                )
                st.session_state.bull_caps[bull] = new_cap

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Run Optimizer
# ─────────────────────────────────────────────────────────────────────────────

with tabs[2]:
    st.header("Run Optimizer")

    if st.session_state.epd_df is None:
        st.warning("⚠️ Please upload an EPD file in the sidebar first.")
    elif not st.session_state.bull_caps:
        st.warning("⚠️ No bulls detected. Check your file and the Bull Caps tab.")
    else:
        # Preview
        with st.expander("🔍 Preview uploaded data"):
            st.dataframe(st.session_state.epd_df.to_pandas(), width=True)

        # Active caps summary
        active_caps = {b: cap for b, cap in st.session_state.bull_caps.items() if cap > 0}
        st.info(
            f"**Active bulls:** {len(active_caps)}   |   "
            f"**Total capacity:** {sum(active_caps.values())} cows   |   "
            f"**Power function:** `{st.session_state.power_fn}`"
        )

        if not active_caps:
            st.error("All bull caps are set to 0 — nothing to solve.")
        else:
            if st.button("▶ Run Optimization", type="primary", width=True):
                with st.spinner("Cleaning data, scoring pairs, solving…"):
                    try:
                        # Clean
                        cleaned = clean_epd_df(
                            st.session_state.epd_df,
                            HIGHER_COLS,
                            LOWER_COLS,
                            st.session_state.criteria_b,
                        )

                        # Filter to bulls with active caps
                        if "bull" in cleaned.columns:
                            cleaned = cleaned.filter(
                                pl.col("bull").str.strip_chars().is_in(list(active_caps.keys()))
                            )

                        # Score
                        diff   = compute_pct_diff(cleaned, st.session_state.criteria_b, LOWER_COLS)
                        normed = normalize_pct_diff(diff)
                        scored = compute_pair_score(
                            normed,
                            st.session_state.trait_weights,
                            st.session_state.cc_weight,
                        )

                        # Solve
                        result = generic_solve(
                            scored,
                            cow_col="cow",
                            bull_col="bull",
                            bull_caps=active_caps,
                            power_fn=st.session_state.power_fn,
                            require_all_assigned=require_all,
                        )
                        st.session_state.result = result
                        st.success(f"✅ Solver status: **{result['status']}** — "
                                   f"{len(result['assignments'])} pairs assigned, "
                                   f"total score {result['total_score']:.2f}")
                        if result["unassigned"]:
                            st.warning(f"Unassigned cows: {', '.join(result['unassigned'])}")
                    except Exception as e:
                        st.error(f"Optimization failed: {e}")
                        import traceback; st.code(traceback.format_exc())

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Results
# ─────────────────────────────────────────────────────────────────────────────

with tabs[3]:
    st.header("Results")

    result = st.session_state.result
    if result is None:
        st.info("Run the optimizer first.")
    else:
        # ── Summary metrics ──────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Status",          result["status"])
        m2.metric("Pairs Assigned",  len(result["assignments"]))
        m3.metric("Unassigned Cows", len(result["unassigned"]))
        m4.metric("Total Score",     f"{result['total_score']:.2f}")

        st.markdown("---")

        # ── Assignments table ────────────────────────────────────────────────
        st.subheader("📋 Assignments")
        assignments_df = pd.DataFrame(result["assignments"])
        st.dataframe(assignments_df, width=True, hide_index=True)

        # Download assignments
        csv_bytes = assignments_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download assignments CSV",
            data=csv_bytes,
            file_name="assignments.csv",
            mime="text/csv",
        )

        st.markdown("---")

        # ── Bull usage chart ─────────────────────────────────────────────────
        st.subheader("🐂 Bull Usage")

        usage_data = []
        for bull, used in result["bull_usage"].items():
            cap  = result["bull_max"].get(bull, 0)
            usage_data.append({"Bull": bull, "Used": used, "Max": cap, "Remaining": cap - used})
        usage_df = pd.DataFrame(usage_data).sort_values("Used", ascending=False)

        fig, ax = plt.subplots(figsize=(max(6, len(usage_df) * 0.8), 4))
        x     = np.arange(len(usage_df))
        width = 0.4
        ax.bar(x - width/2, usage_df["Used"],  width, label="Used",      color="#3B82F6")
        ax.bar(x + width/2, usage_df["Remaining"], width, label="Remaining", color="#E5E7EB")
        ax.set_xticks(x)
        ax.set_xticklabels(usage_df["Bull"], rotation=30, ha="right")
        ax.set_ylabel("Cows")
        ax.set_title("Bull Usage vs Capacity")
        ax.legend()
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig, width=True)

        # Usage table
        st.dataframe(usage_df, width=True, hide_index=True)

        usage_csv = usage_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download bull usage CSV",
            data=usage_csv,
            file_name="bull_usage.csv",
            mime="text/csv",
        )

        st.markdown("---")

        # ── Score distribution ───────────────────────────────────────────────
        if assignments_df is not None and "pair_score" in assignments_df.columns:
            st.subheader("📊 Pair Score Distribution")
            fig2, ax2 = plt.subplots(figsize=(7, 3))
            ax2.hist(assignments_df["pair_score"], bins=20, color="#3B82F6", edgecolor="white")
            ax2.set_xlabel("Pair Score")
            ax2.set_ylabel("Count")
            ax2.set_title("Distribution of Assigned Pair Scores")
            ax2.spines[["top","right"]].set_visible(False)
            st.pyplot(fig2, width=True)