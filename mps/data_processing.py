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

	# Identify weekly columns by Fymw presence in mapping
	week_cols = [c for c in mps_df.columns if c in set(mapping[M.fymw].unique())]
	if not week_cols:
		raise ValueError("No weekly columns in the uploaded MPS match the mapping Fymw labels.")

	id_df = mps_df[required_id_cols].copy()
	values_df = mps_df[week_cols].copy()

	# Melt to long
	long_df = pd.melt(
		pd.concat([id_df, values_df], axis=1),
		id_vars=required_id_cols,
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
	long_df = long_df.sort_values(["Ver_Date", "Program", "Config 1", "Config 2", M.date_code])
	long_df["Cumulative"] = (
		long_df.groupby(["Program", "Config 1", "Config 2", "MPS Ver"])  # type: ignore
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
		group_cols = ["MPS Ver", "Ver_Date", "Ver_Wk_Code", "Program", "Config 1", "Config 2", M.date_code, M.wk_code, M.quarter]
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
	index_cols = ["Ver_Wk_Code", "Program", "Config 1", "Config 2"]
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
	pivot = pivot.sort_values(["_ver_order", "Program", "Config 1", "Config 2"]).drop(columns=["_ver_order"])

	# Quarter banding metadata
	week_to_quarter = (
		mapping.drop_duplicates(subset=[M.wk_code]).set_index(M.wk_code)[M.quarter].to_dict()
	)
	return pivot, week_to_quarter


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

		week_cols = [c for c in anchor_pivot.columns if c not in ("Ver_Wk_Code", *join_keys)]
		for wk in week_cols:
			merged[wk] = merged[f"{wk}_anch"].fillna(0.0) - merged[f"{wk}_comp"].fillna(0.0)

		# Set display version label to the comparison version display
		ver_label = (
			long_df.loc[long_df["MPS Ver"] == ver, "Ver_Wk_Code"].dropna().unique()
		)
		ver_label = ver_label[0] if len(ver_label) else ver
		result_cols = [*join_keys, *week_cols]
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
	writer.close()
	buffer.seek(0)
	return buffer.read()
