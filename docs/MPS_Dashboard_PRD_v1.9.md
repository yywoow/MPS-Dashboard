# MPS Playground – Product Requirements Document (v1.9)

**Date:** 2025-09-22  
**Type:** Web App (Python + Streamlit)

---

## 1) Overview
Web-based dashboard to review Master Production Schedule (MPS) data. Users upload **only the MPS Excel**; a pre-loaded mapping (Date_Code → Wk_Code → Quarter) is embedded. The app standardizes weeks, computes incremental/cumulative series, supports TTL aggregation, version-to-version comparisons, and **MPS Simulation** (cut/add). Layout: **table view only** with quarter banding, horizontally scrollable weeks, robust tab switching, and formatted numbers (e.g., `1,000,000`) and config percents (`x.xx%`).

---

## 2) Inputs
- Columns: `Program`, `Config 1`, `Config 2`, `MPS Ver`, `MPS Type`, weekly incremental columns (`Wk_Code` headers).  
- Mapping (embedded): `Date_Code`, `Wk_Code`, `Quarter`.

---

## 3) Data Transformation
- Normalize `MPS Ver` and weekly headers using mapping.  
- Reshape wide → long; compute cumulative per `(Program, Config1, Config2, MPS Ver, MPS Type)` sorted by `Date_Code`.  
- TTL aggregation; quarter banding metadata.

---

## 4) Dashboard Features
- **Filters**: Program, Config 1, Config 2, MPS Ver (default Anchor = latest **POR**, plus 3 recent PORs as comparisons), Metric (Incremental/Cumulative).  
- **Table view**: Frozen ID columns (`MPS Ver`, `MPS Type`, `Program`, `Config 1`, `Config 2`), scrollable week columns ordered by `Date_Code`, quarter banding.  
- **Tabs**: Comparison Table, Delta Table, Simulation.  
- **Formatting**: numbers with thousands separators; config % as `x.xx%`.  
- **Robustness**: safe state handling—free tab switching, filter changes + Apply/Refresh, and Reset without corrupting state.

---

## 5) Simulation

### 5.1 Scope (unchanged)
- **Program-specific** and **latest-POR-specific**: simulation always applies to the selected **Program** and its **latest POR** only.  
- Horizon = weeks with `Date_Code ≥ latest POR week`.

### 5.2 Granularity Options (new)
User chooses a **simulation granularity** when creating a scenario:

1) **Program level (Top Line)**  
   - User enters one number (e.g., cut 100).  
   - Tool splits across all `(Config 1 × Config 2)` buckets by **overall to-go mix** of the selected Program, then applies **LIFO** per bucket.

2) **Config 1 level (e.g., 16GB)**  
   - User enters a number per **Config 1** (e.g., cut 100 on 16GB).  
   - Tool splits across that Config 1’s **Config 2** children by their **to-go mix**, then LIFO per child bucket.

3) **Config 1 × Config 2 level**  
   - User enters a number for specific sub-config(s) (e.g., cut 80 on 16GB·MID).  
   - LIFO in those exact buckets.

> Policy: proportional split uses **overall to-go mix on the horizon**, not per-week mixes.

### 5.3 Validation (blocking)
- For each targeted bucket (Program, Config 1, or Config1×Config2 after proportional split), ensure **requested ≤ available to-go**.  
- On violation: **STOP**, show error table `{Target | Requested | Available To-Go | Excess}`, ask user to revise. No partial applies.

### 5.4 Allocation
- **LIFO**: apply from last week backward within each targeted bucket; never produce negatives.  
- Integer rounding via largest-remainder where needed.

### 5.5 Outputs in UI
- Updated Incremental, Cumulative, To-go totals, Config %.  
- Config % displayed as `x.xx%`.  
- Toggle: **Original / Simulated / Δ**.  
- Summary banner: “Simulation applied: Cut X at <granularity> on <Program>”.

### 5.6 **Export (updated)**
- **Only export the simulation version** to Excel/CSV (i.e., **do not include POR** rows in the export).  
- Schema identical to input: `Program`, `Config 1`, `Config 2`, `MPS Ver` (same as latest POR), `MPS Type = "Simulation <X> cut|add on <Program>"`, followed by weekly incremental columns.  
- Number formatting preserved in file where applicable (CSV numeric prints raw; Excel can use cell formatting).

---

## 6) Worked Examples

### 6.1 Program level (Top Line) – *Cut 100 on Program A715*
- Latest POR = **2025-09 W1**; horizon = weeks ≥ that week.  
- Program A715 to-go mix by (Config1×Config2):  
  - 16GB·MID 37%, 16GB·SIL 23%, 24GB·MID 27%, 24GB·SIL 13% (example shares).  
- Split 100 as 37/23/27/13; apply LIFO per bucket.  
- **Export** contains **only** the simulated rows for A715 with `MPS Type = "Simulation 100 cut on A715"`.

### 6.2 **Config 1 level** – *Cut 100 on 16GB (in A715)*
- Tool splits 100 across 16GB’s Config 2 children by to-go mix (e.g., MID 61.67%, SIL 38.33% → 62/38).  
- LIFO within 16GB·MID and 16GB·SIL; 24GB rows unchanged.  
- **Export**: only the simulated rows for A715 (all impacted buckets), tagged with `MPS Type = "Simulation 100 cut on A715"`.

### 6.3 Config 1 × Config 2 level – *Cut 80 on 16GB·MID (in A715)*
- LIFO within 16GB·MID only; SIL and 24GB untouched.  
- Validation blocks if 80 > to-go of 16GB·MID.  
- **Export**: only the simulation rows for affected buckets with proper `MPS Type` tag.

---

## 7) Web Implementation
- Python + Streamlit; pandas for processing (Polars optional for speed).  
- AgGrid/Streamlit DataFrame with frozen columns + horizontal scroll.  
- Apply/Refresh/Reset buttons; robust session state.  
- Number formatting: thousands separator; percentages `x.xx%` (UI).

---

## 8) UX Flow
1) Upload POR → validation (mapping present).  
2) Dashboard → choose Program, filters, tabs.  
3) Simulation tab → pick **granularity**, enter quantities, **Validate & Apply**.  
4) Review results (Original/Sim/Δ), then **Export Simulation** (simulation-only file).  
5) **Reset** to clear scenario.

---

## 9) Non-Functional
- Performance: ≤2s update for typical datasets (~100k rows, 200+ weeks).  
- Stability: resilient to rapid tab/filter changes; no cross-state bleed.  
- Clear error states (invalid mapping, over-cut).

---
