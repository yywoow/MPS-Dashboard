# MPS Dashboard Tool – Product Requirements Document (v1.8)

**Date:** 2025-09-22  
**Type:** Web App (Python + Streamlit)

---

## 1) Overview
The **MPS Dashboard Tool** is a **web-based application** for reviewing Master Production Schedule (MPS) data.  
Users upload only their **MPS Excel file**. A pre-loaded **date mapping table** (Date_Code → Wk_Code → Quarter) is embedded in the backend.  
The app standardizes weeks, computes incremental & cumulative metrics, supports TTL aggregation, enables version-to-version comparisons, and provides **MPS Simulation** for cut/add scenarios.  
UI layout: **Table view only**, with **quarter-based banding**, **scrollable weeks**, and **robust tab switching**.

---

## 2) Inputs
- `Program`, `Config 1`, `Config 2`, `MPS Ver`, `MPS Type`, weekly incremental values.  
- Mapping file (embedded): `Date_Code`, `Wk_Code`, `Quarter`.  

---

## 3) Data Transformation
- Normalize `MPS Ver` via mapping.  
- Reshape wide → long.  
- Compute cumulative per `(Program, Config1, Config2, MPS Ver, MPS Type)`.  
- TTL aggregation support.  
- Quarter banding metadata.  

---

## 4) Dashboard Features
- **Filters:** Program, Configs, Versions (default: latest POR as Anchor, 3 most recent PORs as Comparisons).  
- **Table View:** Frozen ID columns; horizontally scrollable weeks; quarter banding.  
- **Tabs:** Comparison Table, Delta Table, Simulation.  
- **Exports:** CSV/XLSX with number formatting (1,000,000).  
- **Robustness:** User can switch freely between tabs/filters/reset without breaking state.  

---

## 5) Simulation

### 5.1 Scope
- Always **program-specific** and **latest POR-specific**.  
- Horizon = weeks ≥ latest POR week.  

### 5.2 Simulation Granularity
- **Case 1: Top-Line Cut/Add**  
  - User enters a total (e.g., cut 100).  
  - Split proportionally by current to-go config % mix.  
  - Apply LIFO within each config.  

- **Case 2: Config-Specific Cut/Add**  
  - User specifies cut/add per config directly (e.g., “Cut 80 from 16GB”).  
  - Apply LIFO only to that config’s rows.  
  - Other configs unchanged.  

### 5.3 Validation
- **Top-Line:** All allocated cuts ≤ available to-go per config. Otherwise STOP.  
- **Config-Specific:** Each specified config independently validated. If request > available → STOP with error message.  

### 5.4 Outputs
- Updated incremental, cumulative, to-go totals, and config %.  
- Config % displayed as `x.xx%`.  
- Toggle: Original / Simulated / Δ.  
- Export with `MPS Type = Simulation <X> cut on <Program>`.  

---

## 6) Example – Config-Specific Simulation

**Context:**  
- Program = A715  
- Latest POR = 2025-09 W1  
- Horizon subset: FY24Q4Sep W1–W3  

**Original POR (16GB + 24GB):**

| Program | Config 1 | Config 2 | MPS Ver   | MPS Type | FY24Q4Sep W1 | FY24Q4Sep W2 | FY24Q4Sep W3 | To-Go Total |
|---------|----------|----------|-----------|----------|--------------|--------------|--------------|-------------|
| A715    | 16GB     | MID      | 2025-09 W1| POR      | 36           | 45           | 30           | 111 |
| A715    | 16GB     | SIL      | 2025-09 W1| POR      | 24           | 25           | 20           | 69  |
| A715    | 24GB     | MID      | 2025-09 W1| POR      | 28           | 35           | 18           | 81  |
| A715    | 24GB     | SIL      | 2025-09 W1| POR      | 12           | 15           | 12           | 39  |

**Scenario:** User requests **Cut 80 from 16GB**.  

- To-go total for 16GB = 180 (≥ 80) ✅ valid.  
- LIFO applied within 16GB:  
  - W3: 30 + 20 = 50 → remove 50 (set to 0).  
  - W2: 45 + 25 = 70 → remove 30 (reduce to 40).  
- Remaining = 0.  
- 24GB untouched.  

**Simulated Output:**

| Program | Config 1 | Config 2 | MPS Ver   | MPS Type                     | FY24Q4Sep W1 | FY24Q4Sep W2 | FY24Q4Sep W3 |
|---------|----------|----------|-----------|------------------------------|--------------|--------------|--------------|
| A715    | 16GB     | MID      | 2025-09 W1| Simulation 80 cut on A715    | 36           | 40           |  0 |
| A715    | 16GB     | SIL      | 2025-09 W1| Simulation 80 cut on A715    | 24           | 20           |  0 |
| A715    | 24GB     | MID      | 2025-09 W1| Simulation 80 cut on A715    | 28           | 35           | 18 |
| A715    | 24GB     | SIL      | 2025-09 W1| Simulation 80 cut on A715    | 12           | 15           | 12 |

**Config Mix Before vs After:**  
- Before: 16GB = 180 (60.00%), 24GB = 120 (40.00%)  
- After: 16GB = 100 (45.45%), 24GB = 120 (54.55%)  

---

## 7) Web Implementation
- Python + Streamlit, pandas backend.  
- Streamlit DataFrame/AgGrid for table (frozen ID cols, scrollable weeks).  
- Export CSV/XLSX.  
- Number formatting applied in UI + export.  

---

## 8) UX Flow
- Upload POR → Dashboard.  
- Filters (Program, Configs, Versions, Metric).  
- Tabs: Comparison, Delta, Simulation.  
- Simulation Tab:  
  - User chooses **Cut by: [Top Line | Config]**.  
  - Input box for total OR config-level table with inputs.  
  - Apply/Refresh.  
- Reset returns to POR baseline.  

---

## 9) Non-Functional
- Robustness: no crashes from tab switching, filter changes, reset.  
- Performance: ≤2s refresh on 100k+ rows.  

---
