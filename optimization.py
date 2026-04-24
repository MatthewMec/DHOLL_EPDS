Import necessary libraries and define helper functions for data manipulation and optimization.
```{python}

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import lets_plot as lp
import pulp
import math
from itertools import product

```


#Define functions for computing percentage differences, scores, and solving the optimization problem for pairing cows and bulls based on their traits and the criteria_B benchmarks.


```{python}

def strip_column_names(df: pl.DataFrame) -> pl.DataFrame:
    return df.rename({col: col.strip() for col in df.columns})

def compute_pct_diff(df: pl.DataFrame) -> pl.DataFrame:
    """
    For each trait in criteria_B that exists as a column in df,
    adds a new column  <trait>_pct  containing:
 
    Higher-is-better traits:
        pct = (value - benchmark) / |benchmark| * 100
 
    Lower-is-better traits (sign baked in, not a separate flip):
        pct = -(value - benchmark) / |benchmark| * 100
 
    In both cases a value exactly equal to criteria_B produces
    exactly 0, so after dividing by std in normalize_pct_diff
    it stays exactly 0 — i.e. criteria_B is always the zero point.
 
    Rows where the trait column is null are left as null in pct.
    Traits where benchmark == 0 use raw difference instead of
    percentage (avoids division by zero).
    """
    exprs = []
    for trait, benchmark in criteria_B.items():
        if trait not in df.columns:
            continue
 
        if trait in lower_cols:
            # Lower is better: flip numerator so beating criteria_B is positive
            # A value exactly equal to benchmark always produces exactly 0
            if benchmark == 0:
                raw = -(pl.col(trait).cast(pl.Float64) - benchmark)
            else:
                raw = -(pl.col(trait).cast(pl.Float64) - benchmark) / abs(benchmark) * 100
        else:
            # Higher is better: standard % difference from criteria_B
            if benchmark == 0:
                raw = pl.col(trait).cast(pl.Float64) - benchmark
            else:
                raw = (pl.col(trait).cast(pl.Float64) - benchmark) / abs(benchmark) * 100
 
        exprs.append(raw.alias(f"{trait}_pct"))
 
    return df.with_columns(exprs)
 
 
 
def compute_pair_score(df: pl.DataFrame, use_combined_count:bool=True) -> pl.DataFrame:
    """
    Sums all <trait>_pct columns into a single pair_score column,
    multiplying each by its TRAIT_WEIGHTS value before summing.
    Traits not in TRAIT_WEIGHTS default to weight 1.0.
    Null trait values are treated as 0.
        if "ME" in df.columns:
        score_expr = score_expr + (
            pl.when(pl.col("ME").cast(pl.Float64) > 0)
            .then(-pl.col("ME").cast(pl.Float64))
            .otherwise(0.0)
            )

    """
    pct_cols = [c for c in df.columns if c.endswith("_pct")]
    if not pct_cols:
        raise ValueError("No _pct columns found — run compute_pct_diff first.")
        
 
    score_expr = sum(
        pl.col(c).fill_null(0.0) * TRAIT_WEIGHTS.get(c.replace("_pct", ""), 1.0)
        for c in pct_cols
    )
    if use_combined_count and "combined_count" in df.columns:
        score_expr = score_expr + pl.col("combined_count").fill_null(0.0) * COMBINED_COUNT_WEIGHT



    return df.with_columns(score_expr.alias("pair_score"))


def apply_power(x: float) -> float:
    if POWER_FUNCTION == "linear":
        return x
    elif POWER_FUNCTION == "squared":
        return math.copysign(x ** 2, x)
    elif POWER_FUNCTION == "cubed":
        return x ** 3
    elif POWER_FUNCTION == "exponential":
        return math.exp(x)
    else:
        raise ValueError(f"Unknown POWER_FUNCTION: {POWER_FUNCTION}")
 
def president_solve(df: pl.DataFrame, verbose: bool = True) -> dict:
    """
    Parameters
    ----------
    df : pl.DataFrame
        Must contain cow_col, bull_col, and pair_score columns.
        Run compute_pct_diff() and compute_pair_score() first.
    """
 
    # Strip whitespace from bull column to avoid key mismatches
    df = df.with_columns(pl.col(bull_col).str.strip_chars())
 
    required = [cow_col, bull_col, "pair_score"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Dataframe missing column: '{col}'")
 
    # Build score lookup from dataframe
    pair_scores = {
        (row[cow_col], row[bull_col]): row["pair_score"]
        for row in df.select([cow_col, bull_col, "pair_score"]).to_dicts()
    }
 
    pairs    = list(pair_scores.keys())
    cow_ids  = sorted({c for c, _ in pairs})
    bull_ids = list(president_max.keys())
 
    # Pre-compute nonlinear coefficients
    scores = {(c, b): apply_power(float(pair_scores[c, b])) for c, b in pairs}
 
    prob = pulp.LpProblem("Cattle_Pct_Pairing", pulp.LpMaximize)
 
    x = {(c, b): pulp.LpVariable(f"x_{c}_{b}", cat="Binary") for c, b in pairs}
 
    # Objective
    prob += pulp.lpSum(scores[c, b] * x[c, b] for c, b in pairs), "Total_Score"
 
    # Each cow breeds at most once
    for c in cow_ids:
        prob += (
            pulp.lpSum(x[c, b] for b in bull_ids if (c, b) in x) <= 1,
            f"cow_once_{c}"
        )
 
    # Each bull respects its breeding limit
    for b in bull_ids:
        prob += (
            pulp.lpSum(x[c, b] for c in cow_ids if (c, b) in x) <= president_max[b],
            f"bull_limit_{b}"
        )
 
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
 
    # Extract assignments
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
 
    bull_usage    = {b: int(sum(1 for a in assignments if a["bull"] == b)) for b in bull_ids}
    assigned_cows = {a["cow"] for a in assignments}
    unassigned    = [c for c in cow_ids if c not in assigned_cows]
 
    result = {
        "status":      pulp.LpStatus[prob.status],
        "total_score": round(float(pulp.value(prob.objective) or 0), 4),
        "assignments": sorted(assignments, key=lambda a: -a["score"]),
        "bull_usage":  bull_usage,
        "bull_max":    president_max,
        "unassigned":  unassigned,
        "power_fn":    POWER_FUNCTION,
    }
 
    if verbose:
        _print_results(result)
 
    return result

def export_csv(result: dict, path: str = "cattle_assignments.csv") -> None:
    pl.DataFrame(result["assignments"]).write_csv(path)
    print(f"  Assignments written to: {path}")
 
    bull_rows = [
        {
            "bull":      b,
            "used":      used,
            "max":       result["bull_max"][b],
            "remaining": result["bull_max"][b] - used,
        }
        for b, used in result["bull_usage"].items()
    ]
    usage_path = "bull_usage_" + path.split("/")[-1]
    pl.DataFrame(bull_rows).write_csv(usage_path)
    print(f"  Bull usage written to:  {usage_path}")


def _print_results(result: dict) -> None:
    print("=" * 58)
    print(f"  CATTLE PAIRING  |  power: {result['power_fn']}")
    print("=" * 58)
    print(f"  Status      : {result['status']}")
    print(f"  Total score : {result['total_score']}")
    print(f"  Assigned    : {len(result['assignments'])} pairs")
    if result["unassigned"]:
        print(f"  Unassigned  : {', '.join(result['unassigned'])}")
    print()
 
    print(f"  {'Cow':<12} {'Bull':<8} {'Pct Score':>10} {'Adj Score':>10}")
    print("  " + "-" * 44)
    for a in result["assignments"]:
        print(f"  {a['cow']:<12} {a['bull']:<8} {a['pair_score']:>10.2f} {a['score']:>10.2f}")
 
    print()
    print(f"  {'Bull':<8} {'Used':>5} {'Max':>5} {'Left':>6}")
    print("  " + "-" * 28)
    for b, used in result["bull_usage"].items():
        cap  = result["bull_max"][b]
        rem  = cap - used
        flag = " FULL" if rem == 0 else ""
        print(f"  {b:<8} {used:>5} {cap:>5} {rem:>6}{flag}")
    print("=" * 58)

def normalize_pct_diff(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalizes each <trait>_pct column using criteria_B as the fixed zero point.
 
    Uses Root Mean Square (RMS) instead of std so the scaling is measured
    around 0 (criteria_B), not around the herd mean:
 
        rms = sqrt(mean(pct_diff²))
        z   = pct_diff / rms
 
    This guarantees:
        - A value exactly matching criteria_B always scores exactly 0
        - Better than criteria_B is always positive
        - Worse than criteria_B is always negative
        - Every trait is scaled equally regardless of magnitude
 
    Traits with zero RMS (all values identical to criteria_B) are left as-is.
    """
    pct_cols = [c for c in df.columns if c.endswith("_pct")]
    if not pct_cols:
        raise ValueError("No _pct columns found — run compute_pct_diff first.")
 
    exprs = []
    for col in pct_cols:
        rms = (df[col].fill_null(0.0) ** 2).mean() ** 0.5
        if rms is None or rms == 0:
            exprs.append(pl.col(col))
        else:
            exprs.append(
                (pl.col(col) / rms).alias(col)
            )
 
    return df.with_columns(exprs)
 

def heifer_solve(df: pl.DataFrame, verbose: bool = True) -> dict:
    """
    Parameters
    ----------
    df : pl.DataFrame
        Must contain cow_col, bull_col, and pair_score columns.
        Run compute_pct_diff() and compute_pair_score() first.
    """
 
    # Strip whitespace from bull column to avoid key mismatches
    df = df.with_columns(pl.col(bull_col).str.strip_chars())
 
    required = [cow_col, bull_col, "pair_score"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Dataframe missing column: '{col}'")
 
    # Build score lookup from dataframe
    pair_scores = {
        (row[cow_col], row[bull_col]): row["pair_score"]
        for row in df.select([cow_col, bull_col, "pair_score"]).to_dicts()
    }
 
    pairs    = list(pair_scores.keys())
    cow_ids  = sorted({c for c, _ in pairs})
    bull_ids = list(heifer_max.keys())
 
    # Pre-compute nonlinear coefficients
    scores = {(c, b): apply_power(float(pair_scores[c, b])) for c, b in pairs}
 
    prob = pulp.LpProblem("Cattle_Pct_Pairing", pulp.LpMaximize)
 
    x = {(c, b): pulp.LpVariable(f"x_{c}_{b}", cat="Binary") for c, b in pairs}
 
    # Objective
    prob += pulp.lpSum(scores[c, b] * x[c, b] for c, b in pairs), "Total_Score"
 
    # Each cow breeds at most once
    for c in cow_ids:
        prob += (
            pulp.lpSum(x[c, b] for b in bull_ids if (c, b) in x) == 1,
            f"cow_once_{c}"
        )
 
    # Each bull respects its breeding limit
    for b in bull_ids:
        prob += (
            pulp.lpSum(x[c, b] for c in cow_ids if (c, b) in x) <= heifer_max[b],
            f"bull_limit_{b}"
        )
 
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
 
    # Extract assignments
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
 
    bull_usage    = {b: int(sum(1 for a in assignments if a["bull"] == b)) for b in bull_ids}
    assigned_cows = {a["cow"] for a in assignments}
    unassigned    = [c for c in cow_ids if c not in assigned_cows]
 
    result = {
        "status":      pulp.LpStatus[prob.status],
        "total_score": round(float(pulp.value(prob.objective) or 0), 4),
        "assignments": sorted(assignments, key=lambda a: -a["score"]),
        "bull_usage":  bull_usage,
        "bull_max":    heifer_max,
        "unassigned":  unassigned,
        "power_fn":    POWER_FUNCTION,
    }
 
    if verbose:
        _print_results(result)
 
    return result

def herd_solve(df: pl.DataFrame, verbose: bool = True) -> dict:
    """
    Parameters
    ----------
    df : pl.DataFrame
        Must contain cow_col, bull_col, and pair_score columns.
        Run compute_pct_diff() and compute_pair_score() first.
    """
 
    # Strip whitespace from bull column to avoid key mismatches
    df = df.with_columns(pl.col(bull_col).str.strip_chars())
 
    required = [cow_col, bull_col, "pair_score"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Dataframe missing column: '{col}'")
 
    # Build score lookup from dataframe
    pair_scores = {
        (row[cow_col], row[bull_col]): row["pair_score"]
        for row in df.select([cow_col, bull_col, "pair_score"]).to_dicts()
    }
 
    pairs    = list(pair_scores.keys())
    cow_ids  = sorted({c for c, _ in pairs})
    bull_ids = list(bull_max.keys())
 
    # Pre-compute nonlinear coefficients
    scores = {(c, b): apply_power(float(pair_scores[c, b])) for c, b in pairs}
 
    prob = pulp.LpProblem("Cattle_Pct_Pairing", pulp.LpMaximize)
 
    x = {(c, b): pulp.LpVariable(f"x_{c}_{b}", cat="Binary") for c, b in pairs}
 
    # Objective
    prob += pulp.lpSum(scores[c, b] * x[c, b] for c, b in pairs), "Total_Score"
 
    # Each cow breeds at most once
    for c in cow_ids:
        prob += (
            pulp.lpSum(x[c, b] for b in bull_ids if (c, b) in x) == 1,
            f"cow_once_{c}"
        )
 
    # Each bull respects its breeding limit
    for b in bull_ids:
        prob += (
            pulp.lpSum(x[c, b] for c in cow_ids if (c, b) in x) <= bull_max[b],
            f"bull_limit_{b}"
        )
 
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
 
    # Extract assignments
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
 
    bull_usage    = {b: int(sum(1 for a in assignments if a["bull"] == b)) for b in bull_ids}
    assigned_cows = {a["cow"] for a in assignments}
    unassigned    = [c for c in cow_ids if c not in assigned_cows]
 
    result = {
        "status":      pulp.LpStatus[prob.status],
        "total_score": round(float(pulp.value(prob.objective) or 0), 4),
        "assignments": sorted(assignments, key=lambda a: -a["score"]),
        "bull_usage":  bull_usage,
        "bull_max":    bull_max,
        "unassigned":  unassigned,
        "power_fn":    POWER_FUNCTION,
    }
 
    if verbose:
        _print_results(result)
 
    return result
def export_csv(result: dict, path: str = "cattle_assignments.csv") -> None:
    pl.DataFrame(result["assignments"]).write_csv(path)
    print(f"  Assignments written to: {path}")
 
    bull_rows = [
        {
            "bull":      b,
            "used":      used,
            "max":       result["bull_max"][b],
            "remaining": result["bull_max"][b] - used,
        }
        for b, used in result["bull_usage"].items()
    ]
    usage_path = "bull_usage_" + path.split("/")[-1]
    pl.DataFrame(bull_rows).write_csv(usage_path)
    print(f"  Bull usage written to:  {usage_path}")
```

```{python}


```

```{python}
president_heifer_csv = strip_column_names(pl.read_csv("president_heifers.csv"))

president_cow_csv = strip_column_names(pl.read_csv("president_cows.csv"))

newbulls_heifer_csv = strip_column_names(pl.read_csv("heifer_planned.csv"))


higher_cols = ["PROS", "CED", "WW", "YW", "ADG", "Milk", "HPG", "CEM", "STAY", "Marb", "CW", "REA", "HB", "GM"]
lower_cols = ["BW", "DMI", "ME", "YG", "FAT"]

criteria_B = {
    "PROS": 120, #good
    "CED": 15, #good
    "Milk": 29, #reasonable
    "WW": 73, #behind
    "YW": 115, #behind
    "ADG": 0.28, #behind
    "HPG": 13, #behind/On the verge
    "CEM": 9, #good
    "Marb": 0.54, #good
    "STAY": 17, #good
    "CW": 29, # good
    "REA": 0.25, # good
    "HB": 70, #great
    "GM": 61, #reasonable
    "BW": -3, # good
    "DMI": 0.43, #behind
    "ME": -2, #good
    "YG": 0.01, #good
    "FAT": 0 #good
} 

TRAIT_WEIGHTS = { 
    "PROS": 1,
    "CED": 1.5, #Don't Drop
    "Milk": 1.3, 
    "WW": .5, 
    "YW": .5, 
    "ADG": 1,
    "HPG": 1, 
    "CEM": 1,
    "Marb": 1, #Don't Drop
    "STAY": 1,
    "CW": 1,
    "REA": 1,
    "HB": 1,
    "GM": 1,
    "BW": 1,
    "DMI": 1,
    "ME": 1, #Don't Drop
    "YG": 1,
    "FAT": 1
    }
COMBINED_COUNT_WEIGHT = 1
bull_max = {
    "Z124G": 8,
    "8269":  8,
    "D172":   20,
    "1181J":  20,
    "1278J":  26,
    "4454M":  26,
    "J490":   25,
    "N17":   0,
    "N54": 17,
    "H33": 0,
    "K20B": 25
}

president_max = {
    "8177F": 10,
}

heifer_max = {
    "N5243": 17,
    "57": 17,
    "N17": 17
}

president_heifer = pl.DataFrame(president_heifer_csv)
newbulls_heifer = pl.DataFrame(newbulls_heifer_csv)
president_cow = pl.DataFrame(president_cow_csv)

#Combine president cows and heifers for analysis
president_all = pl.concat([president_heifer, president_cow])


cow_col = "cow"
bull_col = "bull" 
score_col = "combined_count"

POWER_FUNCTION = "squared"

lower_cols = ["BW", "DMI", "ME", "YG", "FAT"]
```





```{python}

#Read Data In





#Combine Heifers
newbulls_heifer = pl.concat([ newbulls_heifer])

#Clean heifers
heifer_clean = newbulls_heifer.with_columns(
    cow = pl.col("AnimalID").str.split_exact("-", 1).struct.field("field_0"),
    bull = pl.col("AnimalID").str.split_exact("-", 1).struct.field("field_1"),
    higher_count = pl.sum_horizontal([ (pl.col(col) >= criteria_B[col]).cast(pl.UInt8) for col in higher_cols ]).alias("higher_count"),
    lower_count = pl.sum_horizontal([ (pl.col(col) <= criteria_B[col]).cast(pl.UInt8) for col in lower_cols ]).alias("lower_count")
).drop("AnimalID", "Sex", "BirthDate", "BrdCds", "DOB")
heifer_clean = heifer_clean.with_columns(
    combined_count = pl.col("higher_count") + pl.col("lower_count")
)

#Clean president cows and combine with heifers
president_clean = president_all.with_columns(
    cow = pl.col("AnimalID").str.split_exact("-", 1).struct.field("field_0"),
    bull = pl.col("AnimalID").str.split_exact("-", 1).struct.field("field_1"),
    higher_count = pl.sum_horizontal([ (pl.col(col) >= criteria_B[col]).cast(pl.UInt8) for col in higher_cols ]).alias("higher_count"),
    lower_count = pl.sum_horizontal([ (pl.col(col) <= criteria_B[col]).cast(pl.UInt8) for col in lower_cols ]).alias("lower_count")
).drop("AnimalID", "Sex", "BirthDate", "BrdCds", "DOB")

president_clean = president_clean.with_columns(
    combined_count = pl.col("higher_count") + pl.col("lower_count")
)

president_cows = ["N70", "N18", "N28", "N55", "N65", "N41"]
heifer_clean = heifer_clean.with_columns(pl.col("cow").str.strip_chars())
heifer_clean = heifer_clean.filter(~pl.col("cow").is_in(president_cows))

```
```{python}
if __name__ == "__main__":
    president_diff = compute_pct_diff(president_clean)
    president_score_unnorm = compute_pair_score(president_diff)
    president_norm = normalize_pct_diff(president_diff)
    president_score = compute_pair_score(president_norm)
    result = president_solve(president_score, verbose=True)
    export_csv(result, path="president_10.csv")
    result_unnorm = president_solve(president_score_unnorm, verbose=True)

```


```{python}
if __name__ == "__main__":
    newbulls_diff = compute_pct_diff(heifer_clean)
    newbulls_norm = normalize_pct_diff(newbulls_diff)
    newbulls_score = compute_pair_score(newbulls_norm)
    newbulls_result = heifer_solve(newbulls_score)

```


```{python}
newbull_assignments = pl.DataFrame(newbulls_result["assignments"])
newbulls_score.write_csv("newbulls_score.csv")
newbull_assignments.write_csv("newbull_assignment.csv")
```


```{python}
newbulls_score.select("ME","ME_pct","cow","bull")
```



```{python}

exclude_cows = ["M35", "M65", "M60", "M38", "M78"] #Avoid Inbreeding


#Handling N17

nbull = strip_column_names(pl.read_csv("n17_cows.csv"))
nbull54 = strip_column_names(pl.read_csv("n54.csv"))
nweights = strip_column_names(pl.read_csv("weights.csv"))
k20b = strip_column_names(pl.read_excel('K20b.xlsx'))
h33 = strip_column_names(pl.read_excel('H33.xlsx'))

nbull = pl.concat([nbull, nbull54])
#Clean nbull 

nbull_clean = nbull.with_columns(
    cow = pl.col("AnimalID").str.split_exact("-", 1).struct.field("field_0"),
    bull = pl.col("AnimalID").str.split_exact("-", 1).struct.field("field_1"),
    higher_count = pl.sum_horizontal([ (pl.col(col) >= criteria_B[col]).cast(pl.UInt8) for col in higher_cols ]).alias("higher_count"),
    lower_count = pl.sum_horizontal([ (pl.col(col) <= criteria_B[col]).cast(pl.UInt8) for col in lower_cols ]).alias("lower_count")
).drop("AnimalID", "Sex", "BirthDate", "BrdCds", "DOB")
nbull_clean.with_columns(
    combined_count = pl.col("higher_count") + pl.col("lower_count")
)

nbull_clean = nbull_clean.with_columns(pl.col("cow").str.strip_chars())
nweights = nweights.with_columns(pl.col("Tag").str.strip_chars())

nbull_clean = nbull_clean.join(nweights, left_on= "cow", right_on="Tag", how="left")


nbull_clean = nbull_clean.filter(pl.col("Weight") <= 1100)

nbull_clean = nbull_clean.with_columns(
    combined_count = pl.col("higher_count") + pl.col("lower_count")
)


#Get Herd Ready
herd_epd = strip_column_names(pl.read_excel('2026 Planned Mating.xlsx'))
herd_epd =pl.concat([herd_epd, h33, k20b])
herd_clean = herd_epd.with_columns(
    cow = pl.col("AnimalID").str.split_exact("-", 1).struct.field("field_0"),
    bull = pl.col("AnimalID").str.split_exact("-", 1).struct.field("field_1"),
    higher_count = pl.sum_horizontal([ (pl.col(col) >= criteria_B[col]).cast(pl.UInt8) for col in higher_cols ]).alias("higher_count"),
    lower_count = pl.sum_horizontal([ (pl.col(col) <= criteria_B[col]).cast(pl.UInt8) for col in lower_cols ]).alias("lower_count")
).drop("AnimalID", "Sex", "BirthDate", "BrdCds", "DOB")

herd_clean = herd_clean.with_columns(pl.col("cow").str.strip_chars())
herd_clean = herd_clean.join(nweights, left_on= "cow", right_on="Tag", how="left")

herd_clean = herd_clean.with_columns(
    combined_count = pl.col("higher_count") + pl.col("lower_count")
)

#Join Herd and others
herd_clean = pl.concat([herd_clean, nbull_clean])

herd_clean = herd_clean.with_columns(pl.col("bull").str.strip_chars())
herd_clean = herd_clean.with_columns(pl.col("cow").str.strip_chars())
herd_clean = herd_clean.filter(
    ~((pl.col("bull") == "N54") & (pl.col("cow").is_in(exclude_cows)))
)

herd_clean = herd_clean.unique(subset=["cow", "bull"])
herd_delete = ["J490", "N17", "H33"]
herd_clean = herd_clean.filter(~pl.col("bull").is_in(herd_delete)
)

```

```{python}
if __name__ == "__main__":
    herd_diff = compute_pct_diff(herd_clean)
    herd_norm = normalize_pct_diff(herd_diff)
    herd_score = compute_pair_score(herd_norm)
    herd_result = herd_solve(herd_score)

```



```{python}
#Simple Bull Comparison

bull_comp = pl.read_excel('bull_comparison.xlsx')

bull_comp
```


```{python}
if __name__ == "__main__":
    bull_diff = compute_pct_diff(bull_comp)
    bull_norm = normalize_pct_diff(bull_diff)
    bull_score = compute_pair_score(bull_norm)


```


```{python}
herd_assignments = pl.DataFrame(herd_result["assignments"])
herd_assignments = herd_assignments.join(herd_clean, on=["cow", "bull"], how="left")
```