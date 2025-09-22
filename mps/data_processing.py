from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


MAP_FILENAME = "Date Code Mapping (Sat) 2025.csv"


@dataclass(frozen=True)
class MappingColumns:
	date_code: str = "Date_Code"
	wk_code: str = "Wk_Code"
	quarter: str = "Quarter"
	cymw: str = "Cymw"  # e.g., 2025-07 W5
	fymw: str = "Fymw"  # e.g., FY24Q4Aug W1


M = MappingColumns()


def load_mapping(base_dir: str) -> pd.DataFrame:
	"""Load the embedded date mapping CSV and prepare types and sort orders.

	Args:
		base_dir: Directory containing the CSV mapping file.

	Returns:
		DataFrame with parsed date column and unique week identifiers.
	"""
	mapping_path = f"{base_dir}/{MAP_FILENAME}"
	mapping = pd.read_csv(mapping_path)

	# Normalize and parse
	mapping[M.date_code] = pd.to_datetime(mapping[M.date_code], errors="coerce")
	mapping = mapping.dropna(subset=[M.date_code, M.wk_code, M.quarter, M.fymw])

	# Ensure uniqueness for Fymw and Cymw to support joins
	mapping = mapping.sort_values(M.date_code)
	mapping = mapping.drop_duplicates(subset=[M.fymw], keep="first")
	# Keep a separate lookup for Cymw (version labels)
	cymw_lookup = pd.read_csv(mapping_path)
	cymw_lookup[M.date_code] = pd.to_datetime(cymw_lookup[M.date_code], errors="coerce")
	cymw_lookup = cymw_lookup.dropna(subset=[M.date_code, M.cymw])
	cymw_lookup = cymw_lookup.sort_values(M.date_code).drop_duplicates(subset=[M.cymw], keep="first")

	# Attach a suffix to avoid column clashes on merge
	cymw_lookup = cymw_lookup[[M.cymw, M.date_code, M.wk_code, M.quarter]].rename(
		columns={
			M.date_code: "Ver_Date",
			M.wk_code: "Ver_Wk_Code",
			M.quarter: "Ver_Quarter",
		}
	)

	# Merge the version lookup into the primary mapping on nearest week code label equality when possible
	# We will keep them separate and join later on demand.
	mapping = mapping.reset_index(drop=True)
	mapping.attrs["cymw_lookup"] = cymw_lookup
	return mapping


def week_columns_order(mapping: pd.DataFrame, restrict_to: Optional[Sequence[str]] = None) -> List[str]:
	"""Return ordered list of week display headers (Wk_Code) by ascending Date_Code.

	If restrict_to provided, the list is filtered to those week codes only.
	"""
	ordered = mapping.sort_values(M.date_code)[M.wk_code].tolist()
	if restrict_to is None:
		return ordered
	allowed = set(restrict_to)
	return [w for w in ordered if w in allowed]


def prepare_long_data(
	mps_df: pd.DataFrame,
	mapping: pd.DataFrame,
	programs: Optional[Sequence[str]] = None,
	config1: Optional[Sequence[str]] = None,
	config2: Optional[Sequence[str]] = None,
	metric: str = "Incremental",
	ttl_config1: bool = False,
	ttl_config2: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
	"""Reshape wide MPS into long with mapping join and compute cumulative.

	Returns the long dataframe and a dict mapping version raw -> version display label (Ver_Wk_Code).
	"""
	required_id_cols = ["Program", "Config 1", "Config 2", "MPS Ver"]
	missing = [c for c in required_id_cols if c not in mps_df.columns]
	if missing:
		raise ValueError(f"Missing required columns in uploaded file: {missing}")
	# Optional: MPS Type. If missing, assume POR for all rows
	has_mps_type = "MPS Type" in mps_df.columns
	if not has_mps_type:
		mps_df = mps_df.copy()
		mps_df["MPS Type"] = "POR"

	# Identify weekly columns by Fymw presence in mapping
	week_cols = [c for c in mps_df.columns if c in set(mapping[M.fymw].unique())]
	if not week_cols:
		raise ValueError("No weekly columns in the uploaded MPS match the mapping Fymw labels.")

	id_df = mps_df[[*required_id_cols, "MPS Type"]].copy()
	values_df = mps_df[week_cols].copy()

	# Melt to long
	long_df = pd.melt(
		pd.concat([id_df, values_df], axis=1),
		id_vars=[*required_id_cols, "MPS Type"],
		value_vars=week_cols,
		var_name=M.fymw,
		value_name="Incremental",
	)

	# Coerce numeric values, treat blanks as 0 for cumulative math but keep NaN for display if needed
	long_df["Incremental"] = pd.to_numeric(long_df["Incremental"], errors="coerce").fillna(0.0)

	# Join on Fymw to get Date_Code / Wk_Code / Quarter
	long_df = long_df.merge(
		mapping[[M.fymw, M.date_code, M.wk_code, M.quarter]],
		how="left",
		on=M.fymw,
	)

	# Version metadata: map MPS Ver (Cymw) to version date and display label
	cymw_lookup = mapping.attrs.get("cymw_lookup")
	long_df = long_df.merge(
		cymw_lookup[[M.cymw, "Ver_Date", "Ver_Wk_Code", "Ver_Quarter"]],
		how="left",
		left_on="MPS Ver",
		right_on=M.cymw,
	)
	long_df = long_df.drop(columns=[M.cymw])

	# Compute cumulative within each Program/Config/MPS Ver sorted by Date_Code
	long_df = long_df.sort_values(["Ver_Date", "Program", "Config 1", "Config 2", "MPS Type", M.date_code])
	long_df["Cumulative"] = (
		long_df.groupby(["Program", "Config 1", "Config 2", "MPS Ver", "MPS Type"])  # type: ignore
		["Incremental"].cumsum()
	)

	# Optional filters (pre-aggregation)
	if programs:
		long_df = long_df[long_df["Program"].isin(programs)]
	if config1:
		long_df = long_df[long_df["Config 1"].isin(config1)]
	if config2:
		long_df = long_df[long_df["Config 2"].isin(config2)]

	version_label_map = (
		long_df.drop_duplicates(subset=["MPS Ver"])  # each version
			.set_index("MPS Ver")["Ver_Wk_Code"].to_dict()
	)
	return long_df, version_label_map


def _aggregate_for_ttl(df: pd.DataFrame, ttl_config1: bool, ttl_config2: bool) -> pd.DataFrame:
	res = df.copy()
	if ttl_config1:
		res["Config 1"] = "TTL"
	if ttl_config2:
		res["Config 2"] = "TTL"
	if ttl_config1 or ttl_config2:
		group_cols = ["MPS Ver", "Ver_Date", "Ver_Wk_Code", "Program", "Config 1", "Config 2", "MPS Type", M.date_code, M.wk_code, M.quarter]
		res = (
			res.groupby(group_cols, as_index=False)[["Incremental", "Cumulative"]]
			.sum()
		)
	return res


def _pivot_display(
	df: pd.DataFrame,
	metric: str,
	mapping: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
	"""Pivot long df to wide for display; returns df and mapping of week column -> quarter."""
	if metric not in ("Incremental", "Cumulative"):
		raise ValueError("metric must be 'Incremental' or 'Cumulative'")

	# Empty guard: return an empty frame with key columns so UI doesn't break
	index_cols = ["Ver_Wk_Code", "MPS Type", "Program", "Config 1", "Config 2"]
	if df.empty:
		return pd.DataFrame(columns=index_cols), {}

	# Determine week columns in order
	ordered_weeks = week_columns_order(mapping, restrict_to=df[M.wk_code].unique())
	# Build pivot so that columns are week codes
	pivot = (
		pd.pivot_table(
			df,
			index=index_cols,
			columns=M.wk_code,
			values=metric,
			aggfunc="sum",
			fill_value=0.0,
		)
	)
	# Ensure all expected columns in order
	pivot = pivot.reindex(columns=ordered_weeks, fill_value=0.0)

	# Reset index and sort versions by Ver_Date (desc)
	pivot = pivot.reset_index()
	version_order = (
        df.drop_duplicates(subset=["Ver_Wk_Code", "Ver_Date"]).sort_values("Ver_Date", ascending=False)["Ver_Wk_Code"].tolist()
	)
	pivot["_ver_order"] = pivot["Ver_Wk_Code"].map({v: i for i, v in enumerate(version_order)})
	pivot = pivot.sort_values(["_ver_order", "MPS Type", "Program", "Config 1", "Config 2"]).drop(columns=["_ver_order"])

	# Quarter banding metadata
	week_to_quarter = (
		mapping.drop_duplicates(subset=[M.wk_code]).set_index(M.wk_code)[M.quarter].to_dict()
	)
	return pivot, week_to_quarter


# ------------------------- Simulation helpers (v1.6) -------------------------

def get_latest_version_for_program(long_df: pd.DataFrame, program: str, prefer_type: str = "POR") -> Optional[Tuple[str, pd.Timestamp, str]]:
    """Return (MPS Ver, Ver_Date, Ver_Wk_Code) for the latest version for a program.

    Prefer rows with MPS Type == prefer_type. Fallback to any type if none.
    """
    sub = long_df[long_df["Program"] == program]
    if sub.empty:
        return None
    prefer = sub[sub["MPS Type"] == prefer_type]
    use = prefer if not prefer.empty else sub
    meta = (
        use.dropna(subset=["Ver_Date"]).drop_duplicates(subset=["MPS Ver", "Ver_Date", "Ver_Wk_Code"]).sort_values("Ver_Date", ascending=False)
    )
    if meta.empty:
        return None
    row = meta.iloc[0]
    return str(row["MPS Ver"]), pd.to_datetime(row["Ver_Date"]), str(row["Ver_Wk_Code"])


def build_inc_pivot_for_horizon(long_df: pd.DataFrame, mapping: pd.DataFrame, program: str, version: str, ver_date: pd.Timestamp, ttl_config1: bool, ttl_config2: bool) -> Tuple[pd.DataFrame, List[str]]:
    """Return incremental pivot for a program+version limited to weeks on/after the version date (horizon)."""
    sub = long_df[(long_df["Program"] == program) & (long_df["MPS Ver"] == version)].copy()
    sub = _aggregate_for_ttl(sub, ttl_config1, ttl_config2)
    pivot, _ = _pivot_display(sub, "Incremental", mapping)
    # Filter to horizon weeks
    wk_to_date = mapping.drop_duplicates(subset=[M.wk_code]).set_index(M.wk_code)[M.date_code].to_dict()
    key_cols = [c for c in ["Ver_Wk_Code", "MPS Type", "Program", "Config 1", "Config 2"] if c in pivot.columns]
    week_cols = [c for c in pivot.columns if c not in key_cols]
    week_cols_hz = [wk for wk in week_cols if pd.notna(wk_to_date.get(wk)) and wk_to_date[wk] >= ver_date]
    filtered = pivot[[*key_cols, *week_cols_hz]].copy()
    return filtered, week_cols_hz


def compute_to_go_by_bucket(inc_pivot: pd.DataFrame, week_cols: List[str]) -> pd.DataFrame:
    """Compute to-go totals per (Config1, Config2) row from an incremental pivot."""
    base_cols = [c for c in ["Ver_Wk_Code", "MPS Type", "Program", "Config 1", "Config 2"] if c in inc_pivot.columns]
    res = inc_pivot[base_cols].copy()
    res["ToGo"] = inc_pivot[week_cols].sum(axis=1)
    total = res["ToGo"].sum()
    res["Mix"] = np.where(total > 0, res["ToGo"] / total, 0.0)
    return res


def split_request_across_buckets(to_go_df: pd.DataFrame, signed_amount: float, targeted: Optional[Dict[Tuple[str, str], float]] = None) -> Dict[Tuple[str, str], float]:
    """Split a signed request (negative=cut, positive=add) across Config1xConfig2 buckets.

    If targeted provided, use exact values per (Config1, Config2). Otherwise, split by Mix.
    """
    allocations: Dict[Tuple[str, str], float] = {}
    if targeted:
        allocations = dict(targeted)
    else:
        for _, row in to_go_df.iterrows():
            key = (str(row.get("Config 1")), str(row.get("Config 2")))
            allocations[key] = float(signed_amount) * float(row.get("Mix", 0.0))
    return allocations


def lifo_apply_allocation(inc_pivot: pd.DataFrame, week_cols: List[str], allocations: Dict[Tuple[str, str], float]) -> pd.DataFrame:
    """Apply per-bucket signed allocation amounts across weeks using LIFO (latest week backward). No negatives allowed."""
    result = inc_pivot.copy()
    # Ensure numeric
    result[week_cols] = result[week_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    idx_map = {(str(row["Config 1"]), str(row["Config 2"])): i for i, row in result.iterrows()}
    # Iterate allocations
    for key, amt in allocations.items():
        if key not in idx_map:
            continue
        i = idx_map[key]
        remaining = float(amt)
        # Negative: cut. Positive: add
        if remaining < 0:
            need = -remaining
            for wk in reversed(week_cols):
                cur = float(result.at[i, wk])
                if cur <= 0 or need <= 0:
                    continue
                delta = min(cur, need)
                result.at[i, wk] = cur - delta
                need -= delta
            # If need remains, we've cut as much as possible; rest is ignored to preserve non-negativity
        elif remaining > 0:
            add_left = remaining
            for wk in reversed(week_cols):
                if add_left <= 0:
                    break
                # Add greedily to latest weeks
                result.at[i, wk] = float(result.at[i, wk]) + add_left
                add_left = 0.0
        # else zero -> no-op
    return result


def build_simulation_export(original_inc: pd.DataFrame, simulated_inc: pd.DataFrame, week_cols: List[str], mps_ver: str, program: str, action_label: str) -> pd.DataFrame:
    """Return export df with schema: Program, Config 1, Config 2, MPS Ver, MPS Type, weeks.

    Simulated rows appended with MPS Type label like "Simulation <X> cut on <Program>".
    """
    # Original POR rows (take from original_inc, keep only POR rows if present)
    meta_cols = [c for c in ["Program", "Config 1", "Config 2"] if c in original_inc.columns]
    por_type_col = "MPS Type" if "MPS Type" in original_inc.columns else None
    por_rows = original_inc.copy()
    if por_type_col:
        # keep POR only for original
        por_rows = por_rows[por_rows[por_type_col] == "POR"] if (por_type_col in por_rows.columns) else por_rows
    por_rows = por_rows.rename(columns={})
    por_rows_out = por_rows[meta_cols].copy()
    por_rows_out["MPS Ver"] = mps_ver
    por_rows_out["MPS Type"] = "POR"
    for wk in week_cols:
        por_rows_out[wk] = por_rows[wk].values if wk in por_rows.columns else 0.0

    # Simulated rows
    sim_rows_out = simulated_inc[meta_cols].copy()
    sim_rows_out["MPS Ver"] = mps_ver
    sim_rows_out["MPS Type"] = action_label
    for wk in week_cols:
        sim_rows_out[wk] = simulated_inc[wk].values if wk in simulated_inc.columns else 0.0

    export_df = pd.concat([por_rows_out, sim_rows_out], ignore_index=True)
    return export_df


def build_comparison_table(
	long_df: pd.DataFrame,
	mapping: pd.DataFrame,
	metric: str,
	selected_versions: Sequence[str],
	ttl_config1: bool,
	ttl_config2: bool,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
	"""Create display table for comparison view including only selected versions."""
	subset = long_df[long_df["MPS Ver"].isin(selected_versions)].copy()
	subset = _aggregate_for_ttl(subset, ttl_config1, ttl_config2)
	return _pivot_display(subset, metric, mapping)


def build_delta_table(
	long_df: pd.DataFrame,
	mapping: pd.DataFrame,
	metric: str,
	anchor_version: str,
	comparison_versions: Sequence[str],
	ttl_config1: bool,
	ttl_config2: bool,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
	"""Compute Anchor - Comparison deltas for each comparison version.

	Rows correspond to each comparison version with same Program/Configs.
	"""
	anchor_df = long_df[long_df["MPS Ver"] == anchor_version].copy()
	anchor_df = _aggregate_for_ttl(anchor_df, ttl_config1, ttl_config2)
	anchor_pivot, _ = _pivot_display(anchor_df, metric, mapping)

	all_delta_rows: List[pd.DataFrame] = []
	for ver in comparison_versions:
		comp_df = long_df[long_df["MPS Ver"] == ver].copy()
		comp_df = _aggregate_for_ttl(comp_df, ttl_config1, ttl_config2)
		comp_pivot, _ = _pivot_display(comp_df, metric, mapping)

		# Align on Program/Config rows
		join_keys = ["Program", "Config 1", "Config 2"]
		merged = pd.merge(
			anchor_pivot,
			comp_pivot,
			how="outer",
			on=join_keys,
			suffixes=("_anch", "_comp"),
		)

		# derive week columns from anchor pivot excluding key columns (and MPS Type)
		key_cols = ["Ver_Wk_Code", "MPS Type", *join_keys]
		week_cols = [c for c in anchor_pivot.columns if c not in key_cols]
		# Compute all week deltas at once to avoid fragmented writes
		anch_cols = [f"{wk}_anch" for wk in week_cols]
		comp_cols = [f"{wk}_comp" for wk in week_cols]
		anch_vals = merged[anch_cols].fillna(0.0).to_numpy(copy=False)
		comp_vals = merged[comp_cols].fillna(0.0).to_numpy(copy=False)
		deltas = anch_vals - comp_vals
		# Assign back in one go
		for i, wk in enumerate(week_cols):
			merged[wk] = deltas[:, i]

		# Set display version label to the comparison version display
		ver_label = (
			long_df.loc[long_df["MPS Ver"] == ver, "Ver_Wk_Code"].dropna().unique()
		)
		ver_label = ver_label[0] if len(ver_label) else ver
		# Include MPS Type from comparison side if present, else default to POR
		mps_type_col = "MPS Type_comp" if "MPS Type_comp" in merged.columns else None
		if mps_type_col is not None:
			merged["MPS Type"] = merged[mps_type_col]
		else:
			merged["MPS Type"] = "POR"
		result_cols = ["MPS Type", *join_keys, *week_cols]
		delta_rows = merged[result_cols].copy()
		delta_rows.insert(0, "Ver_Wk_Code", ver_label)
		all_delta_rows.append(delta_rows)

	delta_df = pd.concat(all_delta_rows, ignore_index=True) if all_delta_rows else pd.DataFrame()

	# Sort by Program/Config and maintain comparison version order
	order_map = {v: i for i, v in enumerate(comparison_versions)}
	delta_df["_order"] = delta_df["Ver_Wk_Code"].map(order_map)
	delta_df = delta_df.sort_values(["_order", "Program", "Config 1", "Config 2"]).drop(columns=["_order"])

	week_to_quarter = (
		mapping.drop_duplicates(subset=[M.wk_code]).set_index(M.wk_code)[M.quarter].to_dict()
	)
	return delta_df, week_to_quarter


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
	buffer = BytesIO()
	with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
		df.to_excel(writer, index=False, sheet_name=sheet_name)
	buffer.seek(0)
	return buffer.getvalue()
