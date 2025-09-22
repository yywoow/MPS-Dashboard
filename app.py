import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from mps import (
	load_mapping,
	prepare_long_data,
	build_comparison_table,
	build_delta_table,
	week_columns_order,
)
from mps.data_processing import dataframe_to_excel_bytes
from mps.data_processing import (
    get_latest_version_for_program,
    build_inc_pivot_for_horizon,
    compute_to_go_by_bucket,
    split_request_across_buckets,
    lifo_apply_allocation,
    build_simulation_export,
)

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

# Defaults will be set after we know POR preference from preview long
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
	# Prefer POR versions for defaults if present
	if "MPS Type" in _preview_long.columns and (_preview_long["MPS Type"] == "POR").any():
		por_meta = (
			_preview_long[_preview_long["MPS Type"] == "POR"]
			.drop_duplicates(subset=["MPS Ver", "Ver_Date"]).sort_values("Ver_Date", ascending=False)
		)
		versions_in_applied = por_meta["MPS Ver"].tolist()
		if not versions_in_applied:
			versions_in_applied = (
				_preview_long.drop_duplicates(subset=["MPS Ver", "Ver_Date"]).sort_values("Ver_Date", ascending=False)["MPS Ver"].tolist()
			)
	else:
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

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
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
	if st.button("Reset All"):
		# Clear all session state keys used by this app
		for k in list(st.session_state.keys()):
			if k.startswith("applied_") or k.startswith("sim_"):
				del st.session_state[k]
		# Clear cached data (mapping)
		get_mapping.clear()
		st.rerun()
with col_btn3:
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
	key_cols = ["Ver_Wk_Code", "MPS Type", "Program", "Config 1", "Config 2"]
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
			"minWidth": 140 if k in ("Ver_Wk_Code", "MPS Type") else 160,
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
		st.dataframe(df, width='stretch')
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
		st.dataframe(df, width='stretch')
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
comp_tab, delta_tab, sim_tab = st.tabs(["Comparison Table", "Delta Table", "Simulation"])


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
	st.caption("Simulate cut/add on the latest POR for a selected Program. Applies to weeks ≥ that version's date (LIFO).")

	# Choose program (single)
	prog_default = st.session_state["applied_programs"][0] if st.session_state.get("applied_programs") else (program_options[0] if program_options else "")
	program_for_sim = st.selectbox("Program for Simulation", options=program_options, index=(program_options.index(prog_default) if prog_default in program_options else 0))

	# Find latest version (prefer POR) for program
	latest_meta = get_latest_version_for_program(long_df, program_for_sim, prefer_type="POR")
	if latest_meta is None:
		st.warning("No versions found for the selected program.")
		st.stop()
	mps_ver_latest, ver_date_latest, ver_wk_label = latest_meta
	st.info(f"Latest version for {program_for_sim}: {ver_wk_label}")

	# Build incremental pivot for horizon and To-Go by bucket
	inc_pivot_hz, week_cols_hz = build_inc_pivot_for_horizon(
		long_df=long_df, mapping=mapping, program=program_for_sim, version=mps_ver_latest, ver_date=ver_date_latest, ttl_config1=ttl_config1, ttl_config2=ttl_config2
	)
	to_go_df = compute_to_go_by_bucket(inc_pivot_hz, week_cols_hz)
	if not week_cols_hz:
		st.warning("No horizon weeks on/after the latest POR date for the selected program. Nothing to simulate.")
		st.markdown("**Current To-Go (Program latest POR)**")
		render_aggrid(to_go_df[["Config 1", "Config 2", "ToGo", "Mix"]], {**comp_week_quarter, **delta_week_quarter}, is_delta=False, key="sim_togo_nowks")
		st.stop()

	# Simulation inputs
	with st.form("sim_form"):
		c1, c2, c3 = st.columns([1.2, 1, 2])
		with c1:
			action = st.radio("Action", options=["Cut", "Add"], horizontal=True)
		with c2:
			amount = st.number_input("Amount", min_value=0.0, value=0.0, step=10.0, format="%0.1f")
		with c3:
			st.checkbox("Targeted allocation (edit per bucket)", key="sim_targeted", value=st.session_state.get("sim_targeted", False))

		# Editable targeted table when enabled
		edited_allocs: Optional[pd.DataFrame] = None
		if st.session_state.get("sim_targeted", False):
			edit_df = to_go_df[["Config 1", "Config 2", "ToGo", "Mix"]].copy()
			edit_df["Requested"] = 0.0
			if AgGrid is not None:
				grid_options = {
					"defaultColDef": {"editable": True, "resizable": True},
					"columnDefs": [
						{"field": "Config 1", "editable": False},
						{"field": "Config 2", "editable": False},
						{"field": "ToGo", "editable": False, "type": "rightAligned"},
						{"field": "Mix", "editable": False, "type": "rightAligned"},
						{"field": "Requested", "editable": True, "type": "rightAligned"},
					],
				}
				ag_res = AgGrid(edit_df, gridOptions=grid_options, height=240, allow_unsafe_jscode=True, key="sim_edit_grid")
				if isinstance(ag_res, dict) and "data" in ag_res:
					edited_allocs = pd.DataFrame(ag_res["data"])  # streamlit-aggrid dict return
				elif hasattr(ag_res, "data"):
					edited_allocs = pd.DataFrame(getattr(ag_res, "data"))  # object-style return
				else:
					edited_allocs = edit_df
			else:
				st.markdown("Enter per-bucket values in the Requested column, then click Apply Simulation.")
				edited_allocs = st.data_editor(edit_df, use_container_width=True, key="sim_edit_editor")

		apply_sim = st.form_submit_button("Apply Simulation")

	if amount <= 0 and not st.session_state.get("sim_targeted", False):
		st.info("Enter a positive amount or enable targeted allocation and set per-bucket Requested values.")
		# Show current to-go for context
		st.markdown("**Current To-Go (Program latest POR)**")
		render_aggrid(to_go_df[["Config 1", "Config 2", "ToGo", "Mix"]], {**comp_week_quarter, **delta_week_quarter}, is_delta=False, key="sim_togo_table")
		st.stop()

	if apply_sim:
		try:
			# Build allocations
			signed_amount = -float(amount) if action == "Cut" else float(amount)
			targeted_map: Optional[Dict[Tuple[str, str], float]] = None
			if st.session_state.get("sim_targeted", False) and edited_allocs is not None:
				allocs = {}
				for _, r in edited_allocs.iterrows():
					val = float(r.get("Requested", 0.0))
					if val == 0:
						continue
					key = (str(r.get("Config 1")), str(r.get("Config 2")))
					allocs[key] = (-val if action == "Cut" else val)
				targeted_map = allocs

			allocations = split_request_across_buckets(to_go_df, signed_amount, targeted=targeted_map)

			# Pre-check for cuts: cannot exceed to-go
			violations: List[Tuple[str, str, float, float]] = []
			to_go_lookup = { (str(r["Config 1"]), str(r["Config 2"])): float(r["ToGo"]) for _, r in to_go_df.iterrows() }
			for key, amt in allocations.items():
				if amt < 0:
					requested_cut = -amt
					available = to_go_lookup.get(key, 0.0)
					if requested_cut > available + 1e-9:
						violations.append((key[0], key[1], requested_cut, available))
			if violations:
				st.error("Simulation can’t be applied: requested cut exceeds remaining to-go.")
				viol_df = pd.DataFrame(violations, columns=["Config 1", "Config 2", "Requested", "Available"])
				st.dataframe(viol_df, use_container_width=True)
				st.stop()

			# Apply LIFO allocation
			sim_inc = lifo_apply_allocation(inc_pivot_hz, week_cols_hz, allocations)

			# Build Original / Simulated / Delta tables (show incremental only with MPS Type keys)
			key_cols = ["Ver_Wk_Code", "MPS Type", "Program", "Config 1", "Config 2"]
			orig_inc = inc_pivot_hz.copy()
			orig_inc = orig_inc[key_cols + week_cols_hz]
			# Ensure MPS Type exists
			if "MPS Type" not in orig_inc.columns:
				orig_inc.insert(1, "MPS Type", "POR")
			# Simulated rows: set MPS Type label per PRD
			action_label = f"Simulation {abs(signed_amount):.0f} {'cut' if signed_amount < 0 else 'add'} on {program_for_sim}"
			sim_rows = sim_inc.copy()
			if "MPS Type" in sim_rows.columns:
				sim_rows["MPS Type"] = action_label
			else:
				sim_rows.insert(1, "MPS Type", action_label)
			sim_rows = sim_rows[key_cols + week_cols_hz]

			# Delta = Simulated - Original (positive=green, negative=red)
			delta_rows = sim_rows.copy()
			for wk in week_cols_hz:
				delta_rows[wk] = sim_rows[wk].fillna(0.0) - orig_inc[wk].fillna(0.0)
			delta_rows["MPS Type"] = "Δ (Sim - Orig)"

			# Show tabs
			st.markdown("**Original (Incremental, horizon)**")
			render_aggrid(orig_inc, {wk: comp_week_quarter.get(wk) or delta_week_quarter.get(wk) for wk in week_cols_hz}, is_delta=False, key="sim_orig")
			st.divider()
			st.markdown("**Simulated (Incremental, horizon)**")
			render_aggrid(sim_rows, {wk: comp_week_quarter.get(wk) or delta_week_quarter.get(wk) for wk in week_cols_hz}, is_delta=False, key="sim_sim")
			st.divider()
			st.markdown("**Delta (Simulated − Original)**")
			render_aggrid(delta_rows, {wk: comp_week_quarter.get(wk) or delta_week_quarter.get(wk) for wk in week_cols_hz}, is_delta=True, key="sim_delta")

			# Exports per PRD: POR + Simulated together
			export_df = build_simulation_export(orig_inc, sim_rows, week_cols_hz, mps_ver_latest, program_for_sim, action_label)
			csv_sim = export_df.to_csv(index=False).encode("utf-8")
			st.download_button("Export Simulation CSV", data=csv_sim, file_name="simulation_export.csv", mime="text/csv")
			xlsx_sim = dataframe_to_excel_bytes(export_df, sheet_name="Simulation")
			st.download_button("Export Simulation XLSX", data=xlsx_sim, file_name="simulation_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
		except Exception as e:
			st.exception(e)
