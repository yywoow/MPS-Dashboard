import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from mps import (
	load_mapping,
	prepare_long_data,
	build_comparison_table,
	build_delta_table,
	week_columns_order,
)
from mps.data_processing import dataframe_to_excel_bytes

try:
	from st_aggrid import AgGrid
except Exception:  # pragma: no cover
	AgGrid = None  # type: ignore


st.set_page_config(page_title="MPS Dashboard", layout="wide")

BASE_DIR = str(Path(__file__).resolve().parent)
MAPPING_NOTE = "✅ Date mapping is already embedded. No need to upload."

st.title("MPS Dashboard Tool")
st.caption(MAPPING_NOTE)

# --- Load embedded mapping ---
@st.cache_data(show_spinner=False)
def get_mapping() -> pd.DataFrame:
	return load_mapping(BASE_DIR)

mapping = get_mapping()

# --- File input (with optional sample loader) ---
col_u1, col_u2 = st.columns([4, 1])
with col_u1:
	uploaded_file = st.file_uploader("Upload MPS Excel (.xlsx)", type=["xlsx"], accept_multiple_files=False)
with col_u2:
	use_sample = False
	if Path(BASE_DIR, "MPS Example.xlsx").exists():
		use_sample = st.checkbox("Use sample", key="use_sample", value=st.session_state.get("use_sample", False))

# If a file is uploaded, ignore sample mode for this run (avoid mutating widget state)
if uploaded_file is not None:
	use_sample = False


def read_mps_excel(file) -> pd.DataFrame:
	return pd.read_excel(file, engine="openpyxl")

mps_df: pd.DataFrame
if uploaded_file is None and not use_sample:
	st.info("Drag-and-drop your MPS Excel file or click 'Use sample' if available.")
	st.stop()
elif use_sample:
	mps_df = read_mps_excel(Path(BASE_DIR, "MPS Example.xlsx"))
else:
	mps_df = read_mps_excel(uploaded_file)

# Validate core columns
required_cols = ["Program", "Config 1", "Config 2", "MPS Ver"]
missing = [c for c in required_cols if c not in mps_df.columns]
if missing:
	st.error(f"Missing required columns: {missing}")
	st.stop()

# --- Build version order (latest -> oldest) using mapping cymw_lookup ---
cymw_lookup = mapping.attrs.get("cymw_lookup")
ver_meta = pd.DataFrame({"MPS Ver": mps_df["MPS Ver"].dropna().astype(str).unique()})
ver_meta = ver_meta.merge(
	cymw_lookup[["Cymw", "Ver_Date", "Ver_Wk_Code"]].rename(columns={"Cymw": "MPS Ver"}),
	how="left",
	on="MPS Ver",
)
# Fallback: if Ver_Date is missing, sort by string descending
ver_meta["Ver_Date_fill"] = pd.to_datetime(ver_meta["Ver_Date"], errors="coerce")
ver_meta = ver_meta.sort_values(["Ver_Date_fill", "MPS Ver"], ascending=[False, False])
version_sorted = ver_meta["MPS Ver"].tolist()
version_label_map = dict(zip(ver_meta["MPS Ver"], ver_meta["Ver_Wk_Code"].fillna(ver_meta["MPS Ver"])) )

if not version_sorted:
	st.error("No valid 'MPS Ver' values found in the uploaded file.")
	st.stop()

# Defaults: latest as anchor, 3 most recent prior as comparisons
anchor_default = version_sorted[0]
comparison_defaults = version_sorted[1:5]

# --- Filters ---
st.subheader("Filters")

# Options
program_options = sorted(mps_df["Program"].dropna().astype(str).unique())
config1_options = sorted(mps_df["Config 1"].dropna().astype(str).unique())
config2_options = sorted(mps_df["Config 2"].dropna().astype(str).unique())

# Initialize applied state
if "applied_programs" not in st.session_state:
	st.session_state["applied_programs"] = program_options
if "applied_config1" not in st.session_state:
	st.session_state["applied_config1"] = config1_options
if "applied_config2" not in st.session_state:
	st.session_state["applied_config2"] = config2_options
if "applied_metric" not in st.session_state:
	st.session_state["applied_metric"] = "Incremental"

# Preview versions based on applied filters
try:
	_preview_long, _ = prepare_long_data(
		mps_df=mps_df,
		mapping=mapping,
		programs=st.session_state["applied_programs"],
		config1=st.session_state["applied_config1"],
		config2=st.session_state["applied_config2"],
		metric=st.session_state["applied_metric"],
	)
except Exception:
	_preview_long = pd.DataFrame()

if not _preview_long.empty:
	versions_in_applied = (
		_preview_long.drop_duplicates(subset=["MPS Ver", "Ver_Date"]).sort_values("Ver_Date", ascending=False)["MPS Ver"].tolist()
	)
else:
	versions_in_applied = version_sorted

if "applied_anchor" not in st.session_state:
	st.session_state["applied_anchor"] = (versions_in_applied[0] if versions_in_applied else version_sorted[0])
if "applied_comparisons" not in st.session_state:
	st.session_state["applied_comparisons"] = [v for v in versions_in_applied[1:5] if v != st.session_state["applied_anchor"]]
if "applied_ttl1" not in st.session_state:
	st.session_state["applied_ttl1"] = False
if "applied_ttl2" not in st.session_state:
	st.session_state["applied_ttl2"] = False

# Controls with Apply and Refresh
with st.form("filters_form"):
	c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
	with c1:
		ui_programs = st.multiselect("Program", options=program_options, default=st.session_state["applied_programs"])
	with c2:
		ui_config1 = st.multiselect("Config 1", options=config1_options, default=st.session_state["applied_config1"])
	with c3:
		ui_config2 = st.multiselect("Config 2", options=config2_options, default=st.session_state["applied_config2"])
	with c4:
		metric_options = ["Incremental", "Cumulative"]
		ui_metric = st.radio("Metric", options=metric_options, horizontal=True, index=metric_options.index(st.session_state["applied_metric"]))

	c5, c6, c7 = st.columns([2, 2, 4])
	with c5:
		ui_anchor = st.selectbox(
			"Anchor Version",
			options=versions_in_applied,
			format_func=lambda v: version_label_map.get(v, v),
			index=(versions_in_applied.index(st.session_state["applied_anchor"]) if st.session_state["applied_anchor"] in versions_in_applied else 0),
		)
	with c6:
		avail_comps = [v for v in versions_in_applied if v != ui_anchor]
		default_comps = [v for v in st.session_state["applied_comparisons"] if v in avail_comps]
		ui_comparisons = st.multiselect(
			"Comparison Versions",
			options=avail_comps,
			format_func=lambda v: version_label_map.get(v, v),
			default=default_comps,
		)
	with c7:
		ui_ttl1 = st.checkbox("TTL Config 1", value=st.session_state["applied_ttl1"])
		ui_ttl2 = st.checkbox("TTL Config 2", value=st.session_state["applied_ttl2"])

	apply_clicked = st.form_submit_button("Apply")

col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
	if st.button("Refresh"):
		st.session_state["applied_programs"] = program_options
		st.session_state["applied_config1"] = config1_options
		st.session_state["applied_config2"] = config2_options
		st.session_state["applied_metric"] = "Incremental"
		st.session_state["applied_anchor"] = version_sorted[0]
		st.session_state["applied_comparisons"] = version_sorted[1:5]
		st.session_state["applied_ttl1"] = False
		st.session_state["applied_ttl2"] = False
		st.rerun()
with col_btn2:
	st.write("")

if apply_clicked:
	st.session_state["applied_programs"] = ui_programs or program_options
	st.session_state["applied_config1"] = ui_config1 or config1_options
	st.session_state["applied_config2"] = ui_config2 or config2_options
	st.session_state["applied_metric"] = ui_metric
	st.session_state["applied_anchor"] = ui_anchor
	st.session_state["applied_comparisons"] = [v for v in ui_comparisons if v != ui_anchor]
	st.session_state["applied_ttl1"] = ui_ttl1
	st.session_state["applied_ttl2"] = ui_ttl2
	st.rerun()

# --- Prepare long data (using applied state) ---
try:
	long_df, ver_label_map_from_long = prepare_long_data(
		mps_df=mps_df,
		mapping=mapping,
		programs=st.session_state["applied_programs"],
		config1=st.session_state["applied_config1"],
		config2=st.session_state["applied_config2"],
		metric=st.session_state["applied_metric"],
	)
except Exception as e:
	st.exception(e)
	st.stop()

# If filtering removes all rows, show empty-state and stop before building tables
if long_df.empty:
    st.warning("No data after applying filters. Adjust filters to see results.")
    st.stop()

# Constrain versions to those present after filtering
versions_in_filtered = (
	long_df.drop_duplicates(subset=["MPS Ver", "Ver_Date"]).sort_values("Ver_Date", ascending=False)["MPS Ver"].tolist()
)
if not versions_in_filtered:
	st.warning("No versions available after filters.")
	st.stop()

anchor_version = st.session_state["applied_anchor"]
comparison_versions = st.session_state["applied_comparisons"]
ttl_config1 = st.session_state["applied_ttl1"]
ttl_config2 = st.session_state["applied_ttl2"]

# --- Build tables ---
comp_df, comp_week_quarter = build_comparison_table(
	long_df=long_df,
	mapping=mapping,
	metric=st.session_state["applied_metric"],
	selected_versions=[anchor_version, *comparison_versions],
	ttl_config1=ttl_config1,
	ttl_config2=ttl_config2,
)

delta_df, delta_week_quarter = build_delta_table(
	long_df=long_df,
	mapping=mapping,
	metric=st.session_state["applied_metric"],
	anchor_version=anchor_version,
	comparison_versions=comparison_versions,
	ttl_config1=ttl_config1,
	ttl_config2=ttl_config2,
)

# --- AgGrid rendering helpers ---
def build_quarter_grouped_column_defs(df: pd.DataFrame, week_quarter: Dict[str, str], colorize_delta: bool) -> List[dict]:
	key_cols = ["Ver_Wk_Code", "Program", "Config 1", "Config 2"]
	week_cols = [c for c in df.columns if c not in key_cols]

	# Group weeks by quarter in order of appearance
	quarter_order: List[str] = []
	for wk in week_cols:
		q = week_quarter.get(wk, "?")
		if q not in quarter_order:
			quarter_order.append(q)
	quarter_to_weeks: Dict[str, List[str]] = {q: [wk for wk in week_cols if week_quarter.get(wk, "?") == q] for q in quarter_order}

	# Key columns pinned left
	col_defs: List[dict] = []
	for k in key_cols:
		col_defs.append({
			"field": k,
			"headerName": k,
			"pinned": "left",
			"sortable": True,
			"filter": True,
			"minWidth": 140 if k == "Ver_Wk_Code" else 160,
		})

	# Style function for delta colors
	cell_style_js = None
	if colorize_delta:
		cell_style_js = {
			"function": "params => { const v = Number(params.value); if (isNaN(v)) return {}; if (v > 0) return {color: '#1a7f37'}; if (v < 0) return {color: '#d32f2f'}; return {}; }"
		}

	# Alternating quarter header classes
	for i, q in enumerate(quarter_order):
		children = []
		for wk in quarter_to_weeks[q]:
			children.append({
				"field": wk,
				"headerName": wk,
				"type": "rightAligned",
				"valueFormatter": "(p.value===undefined||p.value===null)?'':new Intl.NumberFormat().format(p.value)",
				"cellStyle": cell_style_js,
				"minWidth": 110,
			})
		col_defs.append({
			"headerName": q,
			"headerClass": "quarter-even" if i % 2 == 0 else "quarter-odd",
			"children": children,
		})
	return col_defs


def render_aggrid(df: pd.DataFrame, week_quarter: Dict[str, str], is_delta: bool, key: str):
	if AgGrid is None:
		st.dataframe(df, use_container_width=True)
		return
	col_defs = build_quarter_grouped_column_defs(df, week_quarter, colorize_delta=is_delta)
	grid_options = {
		"defaultColDef": {
			"resizable": True,
			"sortable": False,
			"filter": False,
			"suppressMenu": True,
		},
		"columnDefs": col_defs,
		"suppressHorizontalScroll": False,
		"domLayout": "normal",
	}
	# Inject CSS for quarter banding
	st.markdown(
		"""
		<style>
		.ag-theme-streamlit .ag-header-group-cell.quarter-even { background-color: #f5f7fa; font-weight: 600; }
		.ag-theme-streamlit .ag-header-group-cell.quarter-odd { background-color: #eef1f5; font-weight: 600; }
		</style>
		""",
		unsafe_allow_html=True,
	)
	AgGrid(
		df,
		gridOptions=grid_options,
		height=min(72 + 28 * max(len(df), 8), 700),
		fit_columns_on_grid_load=False,
		allow_unsafe_jscode=True,
		key=key,
	)


def render_aggrid_grouped(
	df: pd.DataFrame,
	week_quarter: Dict[str, str],
	group_cols: List[str],
	is_delta: bool,
	key: str,
):
	if AgGrid is None:
		st.dataframe(df, use_container_width=True)
		return

	# Build column defs with row grouping on requested columns, keep Measurement visible and pinned
	week_cols = [c for c in df.columns if c not in (group_cols + ["Measurement"]) ]
	quarter_order: List[str] = []
	for wk in week_cols:
		q = week_quarter.get(wk, "?")
		if q not in quarter_order:
			quarter_order.append(q)
	quarter_to_weeks: Dict[str, List[str]] = {q: [wk for wk in week_cols if week_quarter.get(wk, "?") == q] for q in quarter_order}

	col_defs: List[dict] = []
	# Group columns: hidden, grouped
	for g in group_cols:
		col_defs.append({
			"field": g,
			"headerName": g,
			"rowGroup": True,
			"hide": True,
		})
	# Measurement pinned and visible
	col_defs.append({
		"field": "Measurement",
		"headerName": "Measurement",
		"pinned": "left",
		"minWidth": 160,
		"sortable": True,
		"filter": True,
	})

	cell_style_js = None
	if is_delta:
		cell_style_js = {
			"function": "params => { const v = Number(params.value); if (isNaN(v)) return {}; if (v > 0) return {color: '#1a7f37'}; if (v < 0) return {color: '#d32f2f'}; return {}; }"
		}

	for i, q in enumerate(quarter_order):
		children = []
		for wk in quarter_to_weeks[q]:
			children.append({
				"field": wk,
				"headerName": wk,
				"type": "rightAligned",
				"valueFormatter": "(p.value===undefined||p.value===null)?'':new Intl.NumberFormat().format(p.value)",
				"cellStyle": cell_style_js,
				"minWidth": 110,
			})
		col_defs.append({
			"headerName": q,
			"headerClass": "quarter-even" if i % 2 == 0 else "quarter-odd",
			"children": children,
		})

	grid_options = {
		"defaultColDef": {
			"resizable": True,
			"sortable": False,
			"filter": False,
			"suppressMenu": True,
		},
		"columnDefs": col_defs,
		"suppressHorizontalScroll": False,
		"domLayout": "normal",
		"groupDisplayType": "groupRows",
		"autoGroupColumnDef": {
			"headerName": "Group",
			"minWidth": 260,
		},
	}

	# Inject CSS for quarter banding
	st.markdown(
		"""
		<style>
		.ag-theme-streamlit .ag-header-group-cell.quarter-even { background-color: #f5f7fa; font-weight: 600; }
		.ag-theme-streamlit .ag-header-group-cell.quarter-odd { background-color: #eef1f5; font-weight: 600; }
		</style>
		""",
		unsafe_allow_html=True,
	)
	AgGrid(
		df,
		gridOptions=grid_options,
		height=min(72 + 28 * max(len(df), 8), 700),
		fit_columns_on_grid_load=False,
		allow_unsafe_jscode=True,
		key=key,
	)

# --- Tabs with exports ---
comp_tab, delta_tab, sim_tab = st.tabs(["Comparison Table", "Delta Table", "Simulation Table"])


with comp_tab:
	st.caption("Anchor table (top) and Comparisons (bottom). Left columns are frozen.")
	# Split comp_df into anchor-only and comparison-only
	anchor_label = (
		long_df.loc[long_df["MPS Ver"] == anchor_version, "Ver_Wk_Code"].dropna().unique()
	)
	anchor_label = anchor_label[0] if len(anchor_label) else anchor_version
	anchor_only = comp_df[comp_df["Ver_Wk_Code"] == anchor_label].copy()
	comparisons_only = comp_df[comp_df["Ver_Wk_Code"] != anchor_label].copy()

	st.markdown("**Anchor**")
	render_aggrid(anchor_only, comp_week_quarter, is_delta=False, key="comp_anchor")
	st.divider()
	st.markdown("**Comparisons**")
	render_aggrid(comparisons_only, comp_week_quarter, is_delta=False, key="comp_comparisons")
	# Exports
	# Exports
	csv_anchor = anchor_only.to_csv(index=False).encode("utf-8")
	st.download_button("Export Anchor CSV", data=csv_anchor, file_name="anchor.csv", mime="text/csv")
	xlsx_anchor = dataframe_to_excel_bytes(anchor_only, sheet_name="Anchor")
	st.download_button("Export Anchor XLSX", data=xlsx_anchor, file_name="anchor.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

	csv_comp = comparisons_only.to_csv(index=False).encode("utf-8")
	st.download_button("Export Comparisons CSV", data=csv_comp, file_name="comparisons.csv", mime="text/csv")
	xlsx_comp = dataframe_to_excel_bytes(comparisons_only, sheet_name="Comparisons")
	st.download_button("Export Comparisons XLSX", data=xlsx_comp, file_name="comparisons.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with delta_tab:
	st.caption("Rows = each Comparison vs Anchor (Anchor − Comparison). Green=positive, Red=negative")
	render_aggrid(delta_df, delta_week_quarter, is_delta=True, key="delta")
	# Exports
	csv_d = delta_df.to_csv(index=False).encode("utf-8")
	st.download_button("Export Delta CSV", data=csv_d, file_name="delta.csv", mime="text/csv")
	xlsx_d = dataframe_to_excel_bytes(delta_df, sheet_name="Delta")
	st.download_button("Export Delta XLSX", data=xlsx_d, file_name="delta.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with sim_tab:
    st.caption("Simulation: Anchor version to-go values (weeks ≥ anchor week)")
    # Filter long_df to anchor version only
    anchor_long = long_df[long_df["MPS Ver"] == anchor_version].copy()

    # Build pivots for cumulative and incremental
    anchor_pivot_cum, _ = build_comparison_table(
        long_df=anchor_long,
        mapping=mapping,
        metric="Cumulative",
        selected_versions=[anchor_version],
        ttl_config1=ttl_config1,
        ttl_config2=ttl_config2,
    )
    anchor_pivot_inc, _ = build_comparison_table(
        long_df=anchor_long,
        mapping=mapping,
        metric="Incremental",
        selected_versions=[anchor_version],
        ttl_config1=ttl_config1,
        ttl_config2=ttl_config2,
    )

    # Determine anchor date threshold
    anchor_ver_date = (
        anchor_long.drop_duplicates(subset=["Ver_Date"])["Ver_Date"].iloc[0]
        if not anchor_long.empty else None
    )
    # Map weeks to Date_Code and keep only >= anchor date
    wk_to_date = mapping.drop_duplicates(subset=["Wk_Code"]).set_index("Wk_Code")["Date_Code"].to_dict()
    week_cols_all = [c for c in anchor_pivot_cum.columns if c not in ("Ver_Wk_Code", "Program", "Config 1", "Config 2")]
    if anchor_ver_date is not None:
        week_cols_togo = [wk for wk in week_cols_all if wk_to_date.get(wk) is not None and wk_to_date[wk] >= anchor_ver_date]
    else:
        week_cols_togo = week_cols_all

    sim_cum = anchor_pivot_cum[["Ver_Wk_Code", "Program", "Config 1", "Config 2", *week_cols_togo]].copy()
    sim_inc = anchor_pivot_inc[["Ver_Wk_Code", "Program", "Config 1", "Config 2", *week_cols_togo]].copy()

    # Render cumulative first, then incremental
    # Combine into one table with Measurement column (Cumulative first, then Incremental)
    sim_cum.insert(0, "Measurement", "Cumulative")
    sim_inc.insert(0, "Measurement", "Incremental")
    sim_df = pd.concat([sim_cum, sim_inc], ignore_index=True)

    # Group by first four columns visually
    render_aggrid_grouped(
        sim_df,
        {wk: delta_week_quarter.get(wk) or comp_week_quarter.get(wk) for wk in week_cols_togo},
        group_cols=["Ver_Wk_Code", "Program", "Config 1", "Config 2"],
        is_delta=False,
        key="sim_grouped",
    )
    csv_s = sim_df.to_csv(index=False).encode("utf-8")
    st.download_button("Export Simulation CSV", data=csv_s, file_name="simulation.csv", mime="text/csv")
    xlsx_s = dataframe_to_excel_bytes(sim_df, sheet_name="Simulation")
    st.download_button("Export Simulation XLSX", data=xlsx_s, file_name="simulation.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
