# MPS Dashboard Tool – Product Requirements Document (v1.6)

**Date:** 2025-09-21  
**Type:** Web App (Python + Streamlit)

---

## 1) Overview
The **MPS Dashboard Tool** is a **web-based application** for reviewing Master Production Schedule (MPS) data.  
Users upload only their **MPS Excel file**. A pre-loaded **date mapping table** (Date_Code → Wk_Code → Quarter) is embedded in the backend, so no extra input is required.  
The app standardizes weeks, computes incremental & cumulative metrics, supports TTL aggregation, enables version-to-version comparisons, and provides **MPS Simulation** for cut/add scenarios.  
UI layout: **Table view only**, with **quarter-based banding** and **horizontal scroll**.

---

## 2) Inputs

### 2.1 User-Uploaded (MPS Excel)
- Columns:
  - `Program`  
  - `Config 1`  
  - `Config 2`  
  - `MPS Ver` (e.g., `2025-07 W5`)  
  - `MPS Type` (e.g., `POR`, `Forecast`, `Simulation …`)  
  - Weekly incremental columns: `FY24Q4Aug W1`, `FY24Q4Aug W2`, … `FY26Q3Jun W4`

### 2.2 Embedded (Background Mapping)
- Pre-loaded CSV (not uploaded by user) with:  
  - `Date_Code` (sortable date)  
  - `Wk_Code` (week label, e.g., `FY24Q4Aug W1`)  
  - `Quarter` (e.g., `CQ4'24`)  

---

## 3) Data Transformation
1. Use embedded mapping to standardize `MPS Ver` and week headers.  
2. Reshape wide → long format with fields:  
   - Program, Config1, Config2, MPS Ver, MPS Type, Date_Code, Wk_Code, Quarter, Incremental  
3. Compute **Cumulative** by group `(Program, Config1, Config2, MPS Ver, MPS Type)` sorted by `Date_Code`.  
4. TTL Aggregation: sum across configs when TTL selected.  
5. Add Quarter Banding metadata for UI display.  

---

## 4) Dashboard Features

### 4.1 Filters
- Program (multi-select)  
- Config 1 / Config 2 (multi-select + TTL option)  
- MPS Ver:  
  - Sorted by **Date_Code** (latest → oldest)  
  - **Default selection**: most recent POR = Anchor, 3 most recent prior PORs = Comparisons  
- Metric toggle: Incremental | Cumulative  
- **Apply button**: apply current filter selections  
- **Refresh button**: recalculate tables (after file reload or changes)  

### 4.2 Table View
- **Column order (from left):**  
  1. **MPS Ver** (displayed as Wk_Code, sorted latest → oldest)  
  2. **MPS Type** (POR, Simulation, Forecast, etc.)  
  3. **Program**  
  4. **Config 1**  
  5. **Config 2**  
  6. Weekly values (`Wk_Code`, ordered by `Date_Code`)  

- **Scrollability:**  
  - Left columns (`MPS Ver`, `MPS Type`, `Program`, `Config 1`, `Config 2`) are **frozen**.  
  - Weekly columns (`Wk_Code`) are **horizontally scrollable**.  
  - Weeks ordered by `Date_Code` (earliest → latest).  
  - Quarter banding applied across the scrollable section.  

- **TTL Aggregation:**  
  - If TTL selected for Config 1 or Config 2, the table displays `"TTL"` in that column.  
  - Values aggregated across selected configs before display.  

### 4.3 Table Tabs
- **Comparison Table**: Anchor + Comparisons wide view (all POR rows).  
- **Delta Table**: Anchor – Comparison differences, with positive = green, negative = red.  

### 4.4 Exports
- Export current table (Comparison, Delta, Simulation) to CSV/XLSX  

---

## 5) Simulation

### 5.1 Goal
Simulate **cut/add scenarios** on the **latest POR horizon** (weeks ≥ that version’s `Date_Code`).  

### 5.2 Simulation Rules
1. **Scope:**  
   - Simulation is always **program-specific** **and version-specific**.  
   - User must select a **Program** when creating a scenario.  
   - Simulation applies only to the **latest POR** for the selected program.  
   - All calculations (to-go totals, % mix, validation, allocation) apply only to that Program + latest POR.  
2. **To-go totals & mix:** Compute to-go totals across the horizon for the selected program; derive % mix for Config 1, Config 2, and Config1×Config2.  
3. **Split:**  
   - **Default (Program-level):** Requested cut/add split across configs by **overall to-go mix**.  
   - **Targeted configs:** If user specifies (e.g., 20 on 16GB, 80 on 32GB), apply exactly to those configs.  
4. **Pre-check validation:**  
   - If any requested cut/add > available to-go for that config, **STOP** and show error.  
   - **Error message:** “Simulation can’t be applied: requested cut exceeds remaining to-go.”  
   - Show violating buckets with Requested, Available, Excess.  
   - User must correct before Apply.  
5. **Allocation:**  
   - LIFO (latest week backward).  
   - Within each config (or config×config2), cut/add placed proportionally by that bucket’s mix.  
   - No negatives allowed.  
6. **Outputs:**  
   - Updated incremental, cumulative, to-go totals, and config %.  
   - Toggle between **Original / Simulated / Δ**.  
   - Export simulated tables to CSV/XLSX.  

### 5.3 Simulation Export Format
- Same schema as input (`Program`, `Config 1`, `Config 2`, `MPS Ver`, `MPS Type`, weeks).  
- **MPS Type** for simulated rows = `"Simulation <X> cut on <Program>"` or `"Simulation <X> add on <Program>"`.  
- Weekly incremental values = post-simulation results.  
- Original POR rows are preserved; simulated rows appended.  

### 5.4 Example – Simulation Export (Program A715, latest POR = 2025-09 W1, cut 100)
Original POR rows (Program A715, latest version only):  

| Program | Config 1 | Config 2 | MPS Ver   | MPS Type | FY24Q4Sep W1 | FY24Q4Sep W2 | FY24Q4Sep W3 |
|---------|----------|----------|-----------|----------|--------------|--------------|--------------|
| A715    | 16GB     | MID      | 2025-09 W1| POR      | 36           | 45           | 30           |
| A715    | 16GB     | SIL      | 2025-09 W1| POR      | 24           | 25           | 20           |
| A715    | 24GB     | MID      | 2025-09 W1| POR      | 28           | 35           | 18           |
| A715    | 24GB     | SIL      | 2025-09 W1| POR      | 12           | 15           | 12           |

Simulated rows (Program A715, 100 cut applied to latest POR):  

| Program | Config 1 | Config 2 | MPS Ver   | MPS Type                     | FY24Q4Sep W1 | FY24Q4Sep W2 | FY24Q4Sep W3 |
|---------|----------|----------|-----------|------------------------------|--------------|--------------|--------------|
| A715    | 16GB     | MID      | 2025-09 W1| Simulation 100 cut on A715   | 36           | 38           |  0           |
| A715    | 16GB     | SIL      | 2025-09 W1| Simulation 100 cut on A715   | 24           | 22           |  0           |
| A715    | 24GB     | MID      | 2025-09 W1| Simulation 100 cut on A715   | 28           | 26           |  0           |
| A715    | 24GB     | SIL      | 2025-09 W1| Simulation 100 cut on A715   | 12           | 14           |  0           |

---

## 6) Web Implementation
- **Framework**: Python + Streamlit  
- **Data processing**: pandas (Polars optional)  
- **Tables**: Streamlit DataFrame/AgGrid (frozen columns + horizontal scroll)  
- **Exports**: pandas `.to_csv` / `.to_excel`  
- **Deployment**:  
  - Local: `http://localhost:8501`  
  - Optional: Internal server or Streamlit Cloud  

---

## 7) UX Flow

### Initial Upload Screen
- Drag-and-drop MPS Excel file  
- “Proceed to Dashboard” after validation  
- Message: “✅ Date mapping is already embedded. No need to upload.”  

### Dashboard Screen
- Filter panel: Program, Configs, Versions, Metric toggle  
- Apply / Refresh buttons  
- Table view (Comparison + Delta tabs)  
- Simulation tab: Add Scenario (Program + latest POR only), enter quantities, run Apply with validation  
- Export buttons (POR + Simulation together)  

---

## 8) Non-Functional Requirements
- Handle 100k+ rows, 200+ week columns  
- Local processing (no external upload)  
- Clear error handling (invalid versions, unmapped weeks, over-cut scenarios)  
- UI responsive: <2s updates for typical dataset after Apply  

---

## 9) Future Enhancements
- Drill-down TTL → Config1 → Config2  
- Monthly/Quarterly aggregation  
- Auto-select latest vs previous version  
- Threshold alerts for deltas  
- Alternative simulation policies (front-load, equal split)  

