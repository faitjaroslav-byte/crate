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
    s = str(x).strip()
    if s == "":
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0

def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def resolve_dims(row, col_name: str, fallback_cols=None):
    fallback_cols = fallback_cols or []
    value = clean_text(row.get(col_name, ""))
    if value:
        return value
    for fb in fallback_cols:
        fb_val = clean_text(row.get(fb, ""))
        if fb_val:
            return fb_val
    return ""

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
    for c in df.columns:
        cl = c.lower()
        if cl == "lid reinforcement beams":
            rename[c] = "Lid Reinforcement Beams"
        if cl == "lid reinforcement longitudal boards":
            rename[c] = "Lid reinforcement Longitudal Boards"
        if cl == "lid reinforcement longitudinal boards":
            rename[c] = "Lid reinforcement Longitudal Boards"
        if cl == "square end wall joists":
            rename[c] = "Square End Wall Joists"
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

def derive_crate_geometry(crate_row, settings):
    load_l = safe_int(crate_row["L"])
    load_w = safe_int(crate_row["W"])
    load_h = safe_int(crate_row["H"])

    walls_dims = resolve_dims(crate_row, "Walls")
    reinf_dims = resolve_dims(crate_row, "Reinforcements/diagonals")
    square_joist_dims = resolve_dims(crate_row, "Square End Wall Joists", ["Reinforcements/diagonals"])
    runner_long_dims = resolve_dims(crate_row, "Runner longitudal")
    lid_dims = resolve_dims(crate_row, "Lid")

    wall_t, wall_w = parse_dims(walls_dims)
    reinf_t, _ = parse_dims(reinf_dims)
    sq_joist_t, _ = parse_dims(square_joist_dims)
    runner_a, runner_b = parse_dims(runner_long_dims)
    lid_t, _ = parse_dims(lid_dims)

    top_margin = settings["top_margin_mm"]
    longitudinal_margin = settings["longitudinal_margin_mm"]
    horizontal_margin = settings["horizontal_margin_mm"]

    # Input L/W/H are load inner dimensions; H is between floor boards and top rail.
    inner_l = load_l + longitudinal_margin
    inner_w = load_w + horizontal_margin
    inner_h = load_h + top_margin

    # L is between inner faces of square end wall joints.
    long_wall_len = inner_l + 2 * sq_joist_t
    short_wall_len = inner_w
    wall_reinf_h = inner_h

    runner_h = min(runner_a, runner_b)

    # Approximate outer envelope used for overview and lid sizing.
    outer_l = long_wall_len + 2 * wall_t
    outer_w = short_wall_len + 2 * wall_t
    outer_h = runner_h + wall_t + wall_reinf_h + reinf_t + lid_t

    return {
        "inner_l_mm": inner_l,
        "inner_w_mm": inner_w,
        "inner_h_mm": inner_h,
        "long_wall_len_mm": long_wall_len,
        "short_wall_len_mm": short_wall_len,
        "wall_reinf_h_mm": wall_reinf_h,
        "outer_l_mm": outer_l,
        "outer_w_mm": outer_w,
        "outer_h_mm": outer_h,
        "wall_t_mm": wall_t,
        "wall_w_mm": wall_w,
        "reinf_t_mm": reinf_t,
        "sq_joist_t_mm": sq_joist_t,
    }

# -----------------------------
# Core calc
# -----------------------------
def compute_for_crate(crate_row, settings):
    crate_id = str(crate_row["Crate"]).strip()
    geom = derive_crate_geometry(crate_row, settings)
    inner_l = geom["inner_l_mm"]
    inner_w = geom["inner_w_mm"]
    wall_h = geom["wall_reinf_h_mm"]
    long_wall_len = geom["long_wall_len_mm"]
    short_wall_len = geom["short_wall_len_mm"]
    outer_l = geom["outer_l_mm"]
    outer_w = geom["outer_w_mm"]

    walls_dims = resolve_dims(crate_row, "Walls")
    reinf_dims = resolve_dims(crate_row, "Reinforcements/diagonals")
    lid_reinf_beams_dims = resolve_dims(crate_row, "Lid Reinforcement Beams", ["Reinforcements/diagonals"])
    lid_reinf_long_dims = resolve_dims(crate_row, "Lid reinforcement Longitudal Boards", ["Lid Reinforcement Beams", "Reinforcements/diagonals"])
    square_joist_dims = resolve_dims(crate_row, "Square End Wall Joists", ["Reinforcements/diagonals"])
    runner_long_dims = resolve_dims(crate_row, "Runner longitudal")
    runner_tr_dims = resolve_dims(crate_row, "Runner transverse")
    lid_dims = resolve_dims(crate_row, "Lid")

    gap = settings["gap_mm"]
    spacing = settings["reinf_spacing_mm"]
    n_long_runners = settings["n_long_runners"]

    pieces = []

    # --- Bottom boards ---
    _, w_b = parse_dims(walls_dims)  # bottom uses same as walls
    n_bottom = ceil_div(inner_w, (w_b + gap))
    add_piece(pieces, crate_id, "Bottom", "Bottom board", "Board", walls_dims, inner_l, n_bottom)

    # --- Walls boards ---
    # layers in height (horizontal rows)
    n_layers = ceil_div(wall_h, (w_b + gap))
    # long walls: 2 per layer
    add_piece(pieces, crate_id, "Walls", "Wall board (long)", "Board", walls_dims, long_wall_len, 2 * n_layers)
    # short walls: 2 per layer
    add_piece(pieces, crate_id, "Walls", "Wall board (short)", "Board", walls_dims, short_wall_len, 2 * n_layers)

    # --- Square end wall joints (additional item) ---
    add_piece(pieces, crate_id, "Walls", "Square end wall joist", "Beam", square_joist_dims, wall_h, 4)

    # --- Lid boards ---
    _, w_l = parse_dims(lid_dims)
    # Lid boards are mounted transversely across lid longitudinal reinforcement boards.
    n_lid = ceil_div(outer_l, (w_l + gap))
    add_piece(pieces, crate_id, "Lid", "Lid board", "Board", lid_dims, outer_w, n_lid)

    # --- Reinforcements for walls (rails + verticals + diagonals) ---
    # Per wall helper
    def wall_reinf(wall_len, wall_label, multiplier):
        # rails: 2 per wall
        add_piece(pieces, crate_id, "Reinforcements", f"Rail ({wall_label})", "Board", reinf_dims, wall_len, 2 * multiplier)

        n_vertical = (math.floor(wall_len / spacing) + 1) if spacing > 0 else 0
        add_piece(pieces, crate_id, "Reinforcements", f"Vertical ({wall_label})", "Board", reinf_dims, wall_h, n_vertical * multiplier)

        n_bays = max(n_vertical - 1, 0)
        if n_bays > 0:
            bay = wall_len / n_bays
            diag_len = math.sqrt(wall_h**2 + bay**2)
            add_piece(pieces, crate_id, "Reinforcements", f"Diagonal ({wall_label})", "Board", reinf_dims, diag_len, n_bays * multiplier)

    # 2 long walls, 2 short walls
    wall_reinf(long_wall_len, "long wall", 2)
    wall_reinf(short_wall_len, "short wall", 2)

    # --- Lid reinforcement (additional items) ---
    n_lid_battens = (math.floor(outer_l / spacing) + 1) if spacing > 0 else 0
    add_piece(pieces, crate_id, "Lid reinforcement", "Lid batten (transverse)", "Beam", lid_reinf_beams_dims, outer_w, n_lid_battens)
    add_piece(pieces, crate_id, "Lid reinforcement", "Lid connector (longitudinal)", "Board", lid_reinf_long_dims, outer_l, 2)

    # --- Runners (beams) ---
    # longitudinal: n_long_runners pieces of length inner length
    add_piece(pieces, crate_id, "Runners", "Runner longitudinal", "Beam", runner_long_dims, inner_l, n_long_runners)

    # transverse: floor(inner_l/1000)+1 pieces of length inner width
    n_tr = math.floor(inner_l / 1000) + 1
    add_piece(pieces, crate_id, "Runners", "Runner transverse", "Beam", runner_tr_dims, inner_w, n_tr)

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

def render_filterable_table(
    title: str,
    df: pd.DataFrame,
    key_prefix: str,
    sum_columns=None,
    show_totals=True,
    min_height=220,
):
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
    height = min(max(len(df_display) * 28 + 70, min_height), 600)
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
st.set_page_config(page_title="CalcCrates", layout="wide")
st.title("CalcCrates")
st.caption("Calculates wooden crate parts and material consumption.")

st.sidebar.header("Global settings")
gap_mm = st.sidebar.number_input("Gap between boards g (mm)", min_value=0, max_value=50, value=3, step=1)
reinf_spacing_mm = st.sidebar.number_input("Reinforcement spacing (mm)", min_value=200, max_value=3000, value=1000, step=50)
n_long_runners = st.sidebar.number_input("Number of longitudinal runners (pcs)", min_value=0, max_value=10, value=2, step=1)
longitudinal_margin_mm = st.sidebar.number_input("Longitudinal margin (mm)", min_value=0, max_value=2000, value=0, step=10)
horizontal_margin_mm = st.sidebar.number_input("Horizontal margin (mm)", min_value=0, max_value=2000, value=0, step=10)
top_margin_mm = st.sidebar.number_input("Top margin above load (mm)", min_value=0, max_value=2000, value=0, step=10)
refresh_clicked = st.sidebar.button("Refresh calculations")

settings = dict(
    gap_mm=int(gap_mm),
    reinf_spacing_mm=int(reinf_spacing_mm),
    n_long_runners=int(n_long_runners),
    longitudinal_margin_mm=int(longitudinal_margin_mm),
    horizontal_margin_mm=int(horizontal_margin_mm),
    top_margin_mm=int(top_margin_mm),
)

uploaded = st.file_uploader("Upload XLSX with crates + configuration", type=["xlsx"])

if uploaded:
    if "applied_settings" not in st.session_state:
        st.session_state["applied_settings"] = settings.copy()
    if "last_file_id" not in st.session_state:
        st.session_state["last_file_id"] = None

    current_file_id = (uploaded.name, uploaded.size)
    if st.session_state["last_file_id"] != current_file_id:
        st.session_state["applied_settings"] = settings.copy()
        st.session_state["last_file_id"] = current_file_id
    elif refresh_clicked:
        st.session_state["applied_settings"] = settings.copy()

    applied_settings = st.session_state["applied_settings"]

    # Read input
    xls = pd.ExcelFile(uploaded)
    sheet_by_norm = {str(name).strip().lower(): name for name in xls.sheet_names}
    crate_sheet = sheet_by_norm.get("crate list")
    cfg_sheet = sheet_by_norm.get("configuration")
    if not crate_sheet or not cfg_sheet:
        st.error("Expected sheets: 'Crate list ' and 'Configuration'.")
        st.stop()

    crates_df = pd.read_excel(xls, sheet_name=crate_sheet)
    crates_df = normalize_columns(crates_df)

    config_df = pd.read_excel(xls, sheet_name=cfg_sheet)
    config_df.columns = [str(c).strip() for c in config_df.columns]

    # Validate materials: crate selections must exist in Configuration sheet
    available = set((config_df["Material type"].astype(str).str.strip() + "|" +
                     config_df["Dimensions"].astype(str).str.strip()).tolist())

    required_cols = [
        "Crate", "L", "W", "H",
        "Walls", "Reinforcements/diagonals",
        "Runner longitudal", "Runner transverse", "Lid",
        "Lid Reinforcement Beams", "Lid reinforcement Longitudal Boards", "Square End Wall Joists"
    ]
    missing = [c for c in required_cols if c not in crates_df.columns]
    if missing:
        st.error(f"Missing columns in 'Crate list ': {missing}")
        st.stop()

    crates_df["Crate"] = crates_df["Crate"].apply(clean_text)
    crates_df = crates_df[crates_df["Crate"] != ""].copy()

    # Check selections exist
    sel_specs = {
        "Walls": "Board",
        "Reinforcements/diagonals": "Board",
        "Runner longitudal": "Beam",
        "Runner transverse": "Beam",
        "Lid": "Board",
        "Lid Reinforcement Beams": "Beam",
        "Lid reinforcement Longitudal Boards": "Board",
        "Square End Wall Joists": "Beam",
    }
    fallback_map = {
        "Walls": [],
        "Reinforcements/diagonals": [],
        "Runner longitudal": [],
        "Runner transverse": [],
        "Lid": [],
        "Lid Reinforcement Beams": ["Reinforcements/diagonals"],
        "Lid reinforcement Longitudal Boards": ["Lid Reinforcement Beams", "Reinforcements/diagonals"],
        "Square End Wall Joists": ["Reinforcements/diagonals"],
    }
    problems = []
    for idx,row in crates_df.iterrows():
        for c, mtype in sel_specs.items():
            dims = resolve_dims(row, c, fallback_map.get(c, []))
            if not dims:
                problems.append((row["Crate"], c, "EMPTY"))
                continue
            key = f"{mtype}|{dims}"
            if key not in available:
                problems.append((row["Crate"], c, key))
    if problems:
        st.warning("Some selected materials are not present in Configuration sheet:")
        st.dataframe(pd.DataFrame(problems, columns=["Crate","Column","MissingKey"]))
        st.stop()

    overview_df = crates_df[["Crate"]].copy()
    geom_rows = crates_df.apply(lambda r: derive_crate_geometry(r, applied_settings), axis=1)
    overview_df["Inner_L"] = geom_rows.apply(lambda g: g["inner_l_mm"])
    overview_df["Inner_W"] = geom_rows.apply(lambda g: g["inner_w_mm"])
    overview_df["Inner_H"] = geom_rows.apply(lambda g: g["inner_h_mm"])
    overview_df["Outer_L"] = geom_rows.apply(lambda g: g["outer_l_mm"])
    overview_df["Outer_W"] = geom_rows.apply(lambda g: g["outer_w_mm"])
    overview_df["Outer_H"] = geom_rows.apply(lambda g: g["outer_h_mm"])
    st.subheader("Crate overview")
    st.dataframe(round_numeric_max_2(overview_df), use_container_width=True)

    # Compute
    all_pieces = []
    for _, row in crates_df.iterrows():
        all_pieces.append(compute_for_crate(row, applied_settings))
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
        min_height=140,
    )

    # Export
    out = export_to_xlsx(overview_df, config_df, pieces_df, mat_all_df)
    st.download_button(
        label="Download XLSX (results)",
        data=out,
        file_name="crate_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.info("Upload your input XLSX to start.")
