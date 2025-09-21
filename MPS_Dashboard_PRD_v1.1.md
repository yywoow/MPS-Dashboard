# MPS Dashboard Tool – Product Requirements Document (v1.1)

**Date:** 2025-09-21  
**Type:** Web App (Python + Streamlit)

---

## 1) Overview
The **MPS Dashboard Tool** is a **web-based application** for reviewing Master Production Schedule (MPS) data.  
Users upload only their **MPS Excel file**. A pre-loaded **date mapping table** (Date_Code → Wk_Code → Quarter) is embedded in the backend, so no extra input is required.  
The app standardizes weeks, computes incremental & cumulative metrics, supports TTL aggregation, and enables version-to-version comparisons.  
UI layout: **Table view only** (top), with **quarter-based banding** and **horizontal scroll**.

---

## 2) Inputs

### 2.1 User-Uploaded (MPS Excel)
- Columns:
  - `Program`  
  - `Config 1`  
  - `Config 2`  
  - `MPS Ver` (e.g., `2025-07 W5`)  
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
   - Program, Config1, Config2, MPS Ver (Date/Wk/Quarter), Date_Code, Wk_Code, Quarter, Incremental  
3. Compute **Cumulative** by group `(Program, Config1, Config2, MPS Ver)` sorted by `Date_Code`.  
4. TTL Aggregation: sum across configs when TTL selected.  
5. Add Quarter Banding metadata for UI display.  

---

## 4) Dashboard Features

### 4.1 Filters
- Program (multi-select)  
- Config 1 / Config 2 (multi-select + TTL option)  
- MPS Ver:  
  - Sorted by **Date_Code** (latest → oldest)  
  - **Default selection**: most recent version = Anchor, 3 most recent prior versions = Comparisons  
- Metric toggle: Incremental | Cumulative  

### 4.2 Table View (Top Section Only)
- **Column order (from left):**  
  1. **MPS Ver** (displayed as Wk_Code, sorted latest → oldest)  
  2. **Program**  
  3. **Config 1**  
  4. **Config 2**  
  5. Weekly values (`Wk_Code`, ordered by `Date_Code`)  

- **Scrollability:**  
  - Left columns (`MPS Ver`, `Program`, `Config 1`, `Config 2`) are **frozen**.  
  - Weekly columns (`Wk_Code`) are **horizontally scrollable**.  
  - Weeks ordered by `Date_Code` (earliest → latest).  
  - Quarter banding applied across the scrollable section (alternating shading with labels `CQ4'24`, `CQ1'25`, etc).  

- **TTL Aggregation:**  
  - If TTL selected for Config 1 or Config 2, the table displays `"TTL"` in that column.  
  - Values aggregated across selected configs before display.  

---

## 5) Table Tabs

### 5.1 Comparison Table
- Rows = Anchor version (top) + Comparison versions (below)  
- Columns = Weekly values (scrollable)  
- Quarter banding visible across weeks  

### 5.2 Delta Table (Anchor – Comparison)
- Rows = Comparisons relative to Anchor  
- Values = Anchor – Comparison deltas  
- Positive values = green, Negative values = red  

---

## 6) Exports
- Export current table (Comparison or Delta) to CSV/XLSX  

---

## 7) Web Implementation
- **Framework**: Python + Streamlit  
- **Data processing**: pandas (Polars optional for performance)  
- **Tables**: Streamlit DataFrame/AgGrid (supports frozen columns + horizontal scroll)  
- **Exports**: pandas `.to_csv` / `.to_excel`  
- **Deployment**:  
  - Local: `http://localhost:8501`  
  - Optional: Internal server or Streamlit Cloud  

---

## 8) UX Flow

### Initial Upload Screen
- Drag-and-drop MPS Excel file  
- "Proceed to Dashboard" after validation  
- Message: "✅ Date mapping is already embedded. No need to upload."  

### Dashboard Screen
- Filter panel: Program, Configs, Versions, Metric toggle  
- Top section: **Table view only** (Comparison Table + Delta Table tabs)  
- Export buttons for both tables  

---

## 9) Non-Functional Requirements
- Handle 100k+ rows, 200+ week columns  
- Local processing (no external upload)  
- Clear error handling (invalid versions, unmapped weeks)  
- UI responsive: <2s updates for typical dataset  

---

## 10) Future Enhancements
- Drill-down TTL → Config1 → Config2  
- Monthly/Quarterly aggregation  
- Auto-select latest vs previous version  
- Threshold alerts for deltas  
