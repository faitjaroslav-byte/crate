import io
import math
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, GridUpdateMode
from st_aggrid.shared import JsCode

# -----------------------------
# Helpers
# -----------------------------
def parse_dims(dim_str: str):
    """
    Parses "25x100" -> (25,100) as ints (mm).
    Accepts spaces, uppercase X.
    """
    if pd.isna(dim_str):
        return None
    s = str(dim_str).strip().lower().replace(" ", "").replace("×", "x")
    parts = s.split("x")
    if len(parts) != 2:
        raise ValueError(f"Bad dimension format: {dim_str!r}. Expected like '25x100'.")
    a, b = int(parts[0]), int(parts[1])
    return a, b  # (thickness,width) for boards; (w,h) for beams

def ceil_div(a, b):
    return int(math.ceil(a / b)) if b > 0 else 0

def safe_int(x):
    if pd.isna(x): 
        return 0
    return int(str(x).strip())

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # fix known columns with trailing spaces in your file
    rename = {}
    for c in df.columns:
        if c.lower() in ["w", "w ", "w  "]:
            rename[c] = "W"
        if c.lower() in ["h", "h ", "h  "]:
            rename[c] = "H"
    # In your file, columns are 'W ' and 'H '
    if "W " in df.columns: rename["W "] = "W"
    if "H " in df.columns: rename["H "] = "H"
    df = df.rename(columns=rename)
    return df

def add_piece(rows, crate_id, part, item, material_type, dims, length_mm, qty):
    """
    rows: list of dicts
    dims: string like '25x100' or '120x80'
    length_mm: piece length in mm
    qty: integer
    """
    if qty <= 0:
        return
    rows.append({
        "Crate": crate_id,
        "Part": part,
        "Item": item,
        "MaterialType": material_type,  # Board / Beam
        "Dimensions": dims,
        "Length_mm": int(round(length_mm)),
        "Qty": int(qty),
        "TotalLength_m": (length_mm * qty) / 1000.0,
    })

def calc_volume_m3(material_type, dims_str, total_length_m):
    a, b = parse_dims(dims_str)
    # both boards and beams: cross-section area = a*b in mm^2
    area_m2 = (a / 1000.0) * (b / 1000.0)
    return area_m2 * total_length_m

# -----------------------------
# Core calc
# -----------------------------
def compute_for_crate(crate_row, settings):
    crate_id = str(crate_row["Crate"]).strip()
    L = safe_int(crate_row["L"])
    W = safe_int(crate_row["W"])
    H = safe_int(crate_row["H"])

    walls_dims = str(crate_row["Walls"]).strip()
    reinf_dims = str(crate_row["Reinforcements/diagonals"]).strip()
    runner_long_dims = str(crate_row["Runner longitudal"]).strip()
    runner_tr_dims = str(crate_row["Runner transverse"]).strip()
    lid_dims = str(crate_row["Lid"]).strip()

    gap = settings["gap_mm"]
    spacing = settings["reinf_spacing_mm"]
    n_long_runners = settings["n_long_runners"]

    pieces = []

    # --- Bottom boards ---
    t_b, w_b = parse_dims(walls_dims)  # bottom uses same as walls (simple)
    n_bottom = ceil_div(W, (w_b + gap))
    add_piece(pieces, crate_id, "Bottom", "Bottom board", "Board", walls_dims, L, n_bottom)

    # --- Walls boards ---
    # layers in height (horizontal rows)
    n_layers = ceil_div(H, (w_b + gap))
    # long walls: 2 per layer
    add_piece(pieces, crate_id, "Walls", "Wall board (long)", "Board", walls_dims, L, 2 * n_layers)
    # short walls: 2 per layer
    add_piece(pieces, crate_id, "Walls", "Wall board (short)", "Board", walls_dims, W, 2 * n_layers)

    # --- Lid boards ---
    t_l, w_l = parse_dims(lid_dims)
    n_lid = ceil_div(W, (w_l + gap))
    add_piece(pieces, crate_id, "Lid", "Lid board", "Board", lid_dims, L, n_lid)

    # --- Reinforcements for walls (rails + verticals + diagonals) ---
    # Per wall helper
    def wall_reinf(wall_len, wall_label, multiplier):
        # rails: 2 per wall
        add_piece(pieces, crate_id, "Reinforcements", f"Rail ({wall_label})", "Board", reinf_dims, wall_len, 2 * multiplier)

        n_vertical = (math.floor(wall_len / spacing) + 1) if spacing > 0 else 0
        add_piece(pieces, crate_id, "Reinforcements", f"Vertical ({wall_label})", "Board", reinf_dims, H, n_vertical * multiplier)

        n_bays = max(n_vertical - 1, 0)
        if n_bays > 0:
            bay = wall_len / n_bays
            diag_len = math.sqrt(H**2 + bay**2)
            add_piece(pieces, crate_id, "Reinforcements", f"Diagonal ({wall_label})", "Board", reinf_dims, diag_len, n_bays * multiplier)

    # 2 long walls, 2 short walls
    wall_reinf(L, "long wall", 2)
    wall_reinf(W, "short wall", 2)

    # --- Runners (beams) ---
    # longitudinal: n_long_runners pieces of length L
    add_piece(pieces, crate_id, "Runners", "Runner longitudinal", "Beam", runner_long_dims, L, n_long_runners)

    # transverse: floor(L/1000)+1 pieces of length W
    n_tr = math.floor(L / 1000) + 1
    add_piece(pieces, crate_id, "Runners", "Runner transverse", "Beam", runner_tr_dims, W, n_tr)

    pieces_df = pd.DataFrame(pieces)
    return pieces_df

def material_summary(pieces_df: pd.DataFrame):
    # Aggregate by crate + material type + dims
    g = (pieces_df
         .groupby(["Crate", "MaterialType", "Dimensions"], as_index=False)
         .agg(TotalLength_m=("TotalLength_m", "sum")))
    g["Volume_m3"] = g.apply(lambda r: calc_volume_m3(r["MaterialType"], r["Dimensions"], r["TotalLength_m"]), axis=1)
    return g

def material_summary_all(pieces_df: pd.DataFrame):
    g = (pieces_df
         .groupby(["MaterialType", "Dimensions"], as_index=False)
         .agg(TotalLength_m=("TotalLength_m", "sum")))
    g["Volume_m3"] = g.apply(lambda r: calc_volume_m3(r["MaterialType"], r["Dimensions"], r["TotalLength_m"]), axis=1)
    return g

def pieces_summary_all(pieces_df: pd.DataFrame):
    # Aggregate construction pieces across all crates by (Part, Item, MaterialType, Dimensions, Length_mm)
    g = (pieces_df
         .groupby(["Part", "Item", "MaterialType", "Dimensions", "Length_mm"], as_index=False)
         .agg(Qty=("Qty", "sum"),
              TotalLength_m=("TotalLength_m", "sum")))
    g["Volume_m3"] = g.apply(lambda r: calc_volume_m3(r["MaterialType"], r["Dimensions"], r["TotalLength_m"]), axis=1)
    return g

def export_to_xlsx(crates_df, config_df, pieces_df, mat_all_df):
    crates_export = round_numeric_max_2(crates_df)
    pieces_export = round_numeric_max_2(pieces_df)
    mat_all_export = round_numeric_max_2(mat_all_df)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        crates_export.to_excel(writer, sheet_name="Input_Crates", index=False)
        config_df.to_excel(writer, sheet_name="Input_Configuration", index=False)
        pieces_export.to_excel(writer, sheet_name="Material_Consumption_Cutting", index=False)
        mat_all_export.to_excel(writer, sheet_name="Material_Ordering_Amounts", index=False)
    output.seek(0)
    return output

def round_numeric_max_2(df: pd.DataFrame) -> pd.DataFrame:
    rounded = df.copy()
    for col in rounded.select_dtypes(include=["number"]).columns:
        numeric = pd.to_numeric(rounded[col], errors="coerce")
        rounded[col] = numeric.round(2)
    return rounded

def render_filterable_table(title: str, df: pd.DataFrame, key_prefix: str, sum_columns=None, show_totals=True):
    st.subheader(title)
    if df.empty:
        st.info("No data.")
        return df

    df_display = round_numeric_max_2(df)

    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_default_column(
        filter=True,
        sortable=True,
        resizable=True,
        floatingFilter=True,
    )
    grid_options = gb.build()
    height = min(max(len(df_display) * 28 + 70, 220), 600)
    grid_response = AgGrid(
        df_display,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=False,
        theme="streamlit",
        height=height,
        key=f"{key_prefix}_grid",
    )
    filtered = pd.DataFrame(grid_response["data"])

    if not show_totals:
        return filtered

    if sum_columns is None:
        sum_columns = list(filtered.select_dtypes(include=["number"]).columns)
    totals = {"Rows": int(len(filtered))}
    for c in sum_columns:
        if c in filtered.columns:
            totals[c] = round(float(pd.to_numeric(filtered[c], errors="coerce").fillna(0).sum()), 2)
    st.caption("SUM (filtered)")
    st.dataframe(pd.DataFrame([totals]), use_container_width=True)
    return filtered

def render_material_consumption_table(title: str, df: pd.DataFrame, key_prefix: str):
    st.subheader(title)
    if df.empty:
        st.info("No data.")
        return df

    df_display = round_numeric_max_2(df)
    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_default_column(
        filter=True,
        sortable=True,
        resizable=True,
        floatingFilter=True,
    )

    sum_footer_js = JsCode(
        """
        function(params) {
            const api = params.api;
            const total = {Crate: "", Part: "", Item: "SUM", MaterialType: "", Dimensions: "", Length_mm: null, Qty: null, TotalLength_m: null, m3: null};
            let qty = 0;
            let totalLength = 0;
            let m3 = 0;

            api.forEachNodeAfterFilterAndSort(function(node) {
                qty += Number(node.data.Qty || 0);
                totalLength += Number(node.data.TotalLength_m || 0);
                m3 += Number(node.data.m3 || 0);
            });

            const round2 = (value) => Math.round(value * 100) / 100;
            const filtered = api.isAnyFilterPresent();

            total.m3 = round2(m3);
            if (filtered) {
                total.Qty = round2(qty);
                total.TotalLength_m = round2(totalLength);
            }
            api.setGridOption("pinnedBottomRowData", [total]);
        }
        """
    )

    grid_options = gb.build()
    grid_options["onGridReady"] = sum_footer_js
    grid_options["onFilterChanged"] = sum_footer_js

    height = min(max(len(df_display) * 28 + 100, 260), 700)
    grid_response = AgGrid(
        df_display,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=True,
        theme="streamlit",
        height=height,
        key=f"{key_prefix}_grid",
    )
    return pd.DataFrame(grid_response["data"])

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Calccrate", layout="wide")
st.title("Calccrate")
st.caption("Calculates wooden crate parts and material consumption.")

st.sidebar.header("Global settings")
gap_mm = st.sidebar.number_input("Gap between boards g (mm)", min_value=0, max_value=50, value=3, step=1)
reinf_spacing_mm = st.sidebar.number_input("Reinforcement spacing (mm)", min_value=200, max_value=3000, value=1000, step=50)
n_long_runners = st.sidebar.number_input("Number of longitudinal runners (pcs)", min_value=0, max_value=10, value=2, step=1)

settings = dict(
    gap_mm=int(gap_mm),
    reinf_spacing_mm=int(reinf_spacing_mm),
    n_long_runners=int(n_long_runners),
)

uploaded = st.file_uploader("Upload XLSX with crates + configuration", type=["xlsx"])

if uploaded:
    # Read input
    xls = pd.ExcelFile(uploaded)
    if "Crate list " not in xls.sheet_names or "Configuration" not in xls.sheet_names:
        st.error("Expected sheets: 'Crate list ' and 'Configuration'.")
        st.stop()

    crates_df = pd.read_excel(xls, sheet_name="Crate list ")
    crates_df = normalize_columns(crates_df)

    config_df = pd.read_excel(xls, sheet_name="Configuration")
    config_df.columns = [str(c).strip() for c in config_df.columns]

    # Validate materials: crate selections must exist in Configuration sheet
    available = set((config_df["Material type"].astype(str).str.strip() + "|" +
                     config_df["Dimensions"].astype(str).str.strip()).tolist())

    required_cols = [
        "Crate", "L", "W", "H",
        "Walls", "Reinforcements/diagonals",
        "Runner longitudal", "Runner transverse", "Lid"
    ]
    missing = [c for c in required_cols if c not in crates_df.columns]
    if missing:
        st.error(f"Missing columns in 'Crate list ': {missing}")
        st.stop()

    # Check selections exist
    sel_cols = ["Walls","Reinforcements/diagonals","Runner longitudal","Runner transverse","Lid"]
    problems = []
    for idx,row in crates_df.iterrows():
        for c in sel_cols:
            dims = str(row[c]).strip()
            mtype = "Board" if c in ["Walls","Reinforcements/diagonals","Lid"] else "Beam"
            key = f"{mtype}|{dims}"
            if key not in available:
                problems.append((row["Crate"], c, key))
    if problems:
        st.warning("Some selected materials are not present in Configuration sheet:")
        st.dataframe(pd.DataFrame(problems, columns=["Crate","Column","MissingKey"]))
        st.stop()

    render_filterable_table("Crate overview", crates_df, "crate_overview", show_totals=False)

    # Compute
    all_pieces = []
    for _, row in crates_df.iterrows():
        all_pieces.append(compute_for_crate(row, settings))
    pieces_df = pd.concat(all_pieces, ignore_index=True) if all_pieces else pd.DataFrame()
    if not pieces_df.empty:
        pieces_df["m3"] = pieces_df.apply(
            lambda r: calc_volume_m3(r["MaterialType"], r["Dimensions"], r["TotalLength_m"]),
            axis=1,
        )

    render_material_consumption_table(
        "Material consumption and cutting plan",
        pieces_df,
        "material_consumption_cutting",
    )

    mat_all_df = material_summary_all(pieces_df)
    render_filterable_table(
        "Material needed (ordering amounts)",
        mat_all_df,
        "material_ordering_amounts",
        sum_columns=["TotalLength_m", "Volume_m3"],
        show_totals=False,
    )

    # Export
    out = export_to_xlsx(crates_df, config_df, pieces_df, mat_all_df)
    st.download_button(
        label="Download XLSX (results)",
        data=out,
        file_name="crate_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.info("Upload your input XLSX to start.")
