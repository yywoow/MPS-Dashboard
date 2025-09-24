# MPS Dashboard Tool – Product Requirements Document (v2.0)

**Date:** 2025-09-23  
**Type:** Web App (Python + Streamlit)

---

## 1) Overview
The MPS Dashboard Tool is a web-based application for reviewing Master Production Schedule (MPS) data.  
Users upload their **MPS Excel file**; a pre-loaded mapping (Date_Code → Wk_Code → Quarter) is embedded.  
The app standardizes weeks, computes incremental/cumulative series, supports TTL aggregation across **Config1–Config5**, enables version-to-version comparisons, and provides **MPS Simulation** (cut/add) at different granularities.  
Layout: **table view only** with quarter banding, horizontally scrollable weeks, robust tab switching, and formatted numbers (e.g., `1,000,000`) and config percents (`x.xx%`).

---

## 2) Inputs

### 2.1 User File (MPS Example 0923)
- Columns:  
  - `Program`  
  - `Config 1`  
  - `Config 2`  
  - `Config 3`  
  - `Config 4`  
  - `Config 5`  
  - `MPS Ver` (e.g., 2025-09 W1)  
  - `MPS Type` (POR, Forecast, Simulation …)  
  - Weekly incremental columns (Wk_Code headers)  

### 2.2 Mapping File (embedded)
- `Date_Code` (sortable)  
- `Wk_Code` (week label)  
- `Quarter`  

---

## 3) Data Transformation
- Normalize `MPS Ver` and week headers using mapping.  
- Reshape wide → long, including Program + Config1–5.  
- Compute cumulative per (Program + Config1–5 + MPS Ver + MPS Type).  
- TTL aggregation allowed at **any config level (1–5)**.  
- Quarter banding metadata applied to weekly columns.  

---

## 4) Dashboard Features
- **Filters:** Program, Config1–5 (multi-select with TTL option at any level), Versions (Anchor = latest POR, plus 3 prior PORs), Metric toggle.  
- **Table view:** Frozen ID columns (Program, Config1–5, MPS Ver, MPS Type), horizontally scrollable weekly columns ordered by Date_Code.  
- **Tabs:** Comparison Table, Delta Table, Simulation.  
- **Formatting:** thousands separators for numbers, percentages as x.xx%.  
- **Robustness:** safe state handling for tab/filter/reset.  

---

## 5) Simulation

### 5.1 Scope
- Simulation is always **Program-specific** and **latest POR-specific**.  
- Horizon = weeks ≥ latest POR week.  

### 5.2 Granularity Options (extended for Config1–5)
User chooses the **cut/add granularity**:  

1. **Program level (Top Line)**  
   - Cut/add applies across all Config1–5 combinations by overall to-go mix.  
   - Example: *Cut 100 at Program A715* → split across all leaves by to-go % mix.  

2. **Config N level (N = 1 to 5)**  
   - Cut/add applies across all child configs below the chosen level.  
   - Example: *Cut 200 at Config1=16GB* → auto-split across Config2–5 under 16GB.  

3. **Leaf level (full Config1–5 path)**  
   - Cut/add applies directly to the specified leaf bucket.  
   - Example: *Cut 80 at 16GB·MID·VAR1·A·X*.  

### 5.3 Validation
- For each target (Program, ConfigN, or leaf bucket), ensure **requested ≤ available to-go**.  
- If violation, STOP and show error table with Requested, Available, Excess.  

### 5.4 Allocation
- Proportional split by **to-go mix** at the chosen level.  
- Apply **LIFO** within each affected leaf bucket.  
- Never produce negatives.  

### 5.5 Outputs
- Updated incremental, cumulative, to-go totals, and config %.  
- Export = **only simulated rows**, schema identical to input with `MPS Type = Simulation …`.  

---

## 6) Worked Examples

### 6.1 Program-level Cut – *Cut 100 on Program A715*
- Latest POR = 2025-09 W1; horizon = weeks ≥ that week.  
- A715 to-go total = 1,000 units across all Config1–5 leaves.  
- Split 100 across all leaves by % mix, then LIFO in each.  
- Export shows simulated rows with `MPS Type = "Simulation 100 cut on A715"`.  

### 6.2 Config1-level Cut – *Cut 200 on 16GB (in A715)*

**To-Go Totals under 16GB:**  
- 16GB·MID = 250  
- 16GB·SIL = 150  
- **Total = 400**  

**Step 1 – Allocation:**  
- Requested cut = 200 on 16GB  
- Split 200 by share: MID 62.5% (125), SIL 37.5% (75)  

**Step 2 – Cascade deeper (Config3–5):**  
- MID (125 cut): VAR1=150 (60%) → 75, VAR2=100 (40%) → 50  
- SIL (75 cut): VAR1=90 (60%) → 45, VAR2=60 (40%) → 30  

**Step 3 – LIFO:** apply week-by-week until target cuts absorbed.  

**Simulated Output Example:**  

| Program | Config1 | Config2 | Config3 | Config4 | Config5 | MPS Ver   | MPS Type                     | FY24Q4Sep W1 | FY24Q4Sep W2 | FY24Q4Sep W3 | To-Go |
|---------|---------|---------|---------|---------|---------|-----------|------------------------------|--------------|--------------|--------------|-------|
| A715    | 16GB    | MID     | VAR1    | A       | X       | 2025-09 W1| Simulation 200 cut on A715   | 50           | 25           | 0            | 75    |
| A715    | 16GB    | MID     | VAR2    | A       | Y       | 2025-09 W1| Simulation 200 cut on A715   | 40           | 10           | 0            | 50    |
| A715    | 16GB    | SIL     | VAR1    | B       | Z       | 2025-09 W1| Simulation 200 cut on A715   | 30           | 15           | 0            | 45    |
| A715    | 16GB    | SIL     | VAR2    | B       | W       | 2025-09 W1| Simulation 200 cut on A715   | 20           | 10           | 0            | 30    |

### 6.3 Leaf-level Cut – *Cut 80 on 16GB·MID·VAR1 (in A715)*
- To-go under 16GB·MID·VAR1 = 150  
- Requested cut = 80 → valid  
- Apply LIFO within that exact path, no impact to other configs  
- Export shows only simulated row for that path  

---

## 7) Web Implementation
- Python + Streamlit; pandas for processing.  
- AgGrid/Streamlit DataFrame with frozen columns + horizontal scroll.  
- Apply/Refresh/Reset with session state.  
- Export simulation-only Excel/CSV.  

---

## 8) UX Flow
- Upload POR → validation.  
- Dashboard → filter Program + configs.  
- Simulation tab → choose granularity (Program / Config1–5 / Leaf), enter cut/add.  
- Validate & Apply → results in table with Original/Sim/Δ toggle.  
- Export simulation-only file.  
- Reset clears scenario.  

---

## 9) Non-Functional
- Handle 100k+ rows × 5 configs × 200 weeks.  
- Robust tab switching & filter changes.  
- Performance target ≤2s refresh.  

---
