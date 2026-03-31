import streamlit as st
import pandas as pd
import json
import folium
from streamlit_folium import st_folium

ROUTE_COLS = [
    "SOURCE_FILE", "ROUTE_BOOK", "ZONE", "DAY_OF_WEEK", "WEEKS_OF_MONTH",
    "STEP_NUMBER", "STREET_NAME", "SIDE", "FROM_STREET", "TO_STREET",
    "TIME_START", "TIME_END", "FROM_CNN", "TO_CNN"
]
SEG_COLS = [
    "SOURCE_FILE", "ROUTE_BOOK", "ZONE", "DAY_OF_WEEK", "STEP_NUMBER",
    "STREET_NAME", "SIDE", "FROM_STREET", "TO_STREET", "SEGMENT_CNN",
    "LNG", "LAT", "GEOJSON"
]

STEP_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#a9a9a9", "#e6beff", "#ffe119",
]

@st.cache_data
def load_routes():
    df = pd.read_csv("data/routes.csv.gz", header=None, names=ROUTE_COLS, compression="gzip")
    df["STEP_NUMBER"] = pd.to_numeric(df["STEP_NUMBER"], errors="coerce")
    df["FROM_CNN"] = pd.to_numeric(df["FROM_CNN"], errors="coerce")
    df["TO_CNN"] = pd.to_numeric(df["TO_CNN"], errors="coerce")
    return df

@st.cache_data
def load_segments():
    df = pd.read_csv("data/segments.csv.gz", header=None, names=SEG_COLS, compression="gzip")
    df["STEP_NUMBER"] = pd.to_numeric(df["STEP_NUMBER"], errors="coerce")
    df["LNG"] = pd.to_numeric(df["LNG"], errors="coerce")
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["SEGMENT_CNN"] = pd.to_numeric(df["SEGMENT_CNN"], errors="coerce")
    return df

def get_source_files(routes_df):
    grouped = routes_df.groupby("SOURCE_FILE")["ROUTE_BOOK"].apply(
        lambda x: " | ".join(sorted(x.dropna().unique()))
    ).reset_index()
    grouped.columns = ["SOURCE_FILE", "ROUTE_BOOKS"]
    return grouped.sort_values("SOURCE_FILE")

def get_route_books(routes_df, source_file):
    rb = routes_df.loc[
        (routes_df["SOURCE_FILE"] == source_file) & routes_df["ROUTE_BOOK"].notna(),
        "ROUTE_BOOK"
    ].unique()
    return sorted(rb)

def get_schedule(routes_df, source_file, route_book):
    mask = (routes_df["SOURCE_FILE"] == source_file) & (routes_df["ROUTE_BOOK"] == route_book)
    df = routes_df[mask].groupby(
        ["DAY_OF_WEEK", "WEEKS_OF_MONTH", "TIME_START", "TIME_END"]
    ).size().reset_index(name="Steps")
    df.columns = ["Day", "Weeks", "Start", "End", "Steps"]
    day_order = {"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
                 "Friday": 5, "Saturday": 6, "Sunday": 7}
    df["_sort"] = df["Day"].map(day_order).fillna(8)
    return df.sort_values(["_sort", "Start"]).drop(columns=["_sort"])

def get_route_steps(routes_df, source_file, route_book, day_filter=None):
    mask = (routes_df["SOURCE_FILE"] == source_file) & (routes_df["ROUTE_BOOK"] == route_book)
    if day_filter:
        mask &= routes_df["DAY_OF_WEEK"] == day_filter
    df = routes_df[mask].copy()
    df = df.rename(columns={
        "DAY_OF_WEEK": "Day", "STEP_NUMBER": "Step", "STREET_NAME": "Street",
        "SIDE": "Side", "FROM_STREET": "From", "TO_STREET": "To",
        "FROM_CNN": "From CNN", "TO_CNN": "To CNN"
    })
    day_order = {"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
                 "Friday": 5, "Saturday": 6, "Sunday": 7}
    df["_sort"] = df["Day"].map(day_order).fillna(8)
    return df.sort_values(["_sort", "Step"]).drop(columns=["_sort"])

def get_route_info(routes_df, source_file, route_book):
    mask = (routes_df["SOURCE_FILE"] == source_file) & (routes_df["ROUTE_BOOK"] == route_book)
    df = routes_df[mask]
    if len(df) == 0:
        return None
    return {
        "ZONE": df["ZONE"].dropna().unique()[0] if len(df["ZONE"].dropna().unique()) > 0 else "-",
        "DAYS": df["DAY_OF_WEEK"].dropna().nunique(),
        "TOTAL_STEPS": len(df),
    }

def get_segments(seg_df, source_file, route_book, day_filter=None):
    mask = (seg_df["SOURCE_FILE"] == source_file) & (seg_df["ROUTE_BOOK"] == route_book)
    if day_filter:
        mask &= seg_df["DAY_OF_WEEK"] == day_filter
    return seg_df[mask].copy()

def build_leaflet_map(seg_df):
    valid = seg_df.dropna(subset=["LAT", "LNG"])
    if len(valid) == 0:
        return None

    center_lat = valid["LAT"].mean()
    center_lng = valid["LNG"].mean()
    m = folium.Map(location=[center_lat, center_lng], zoom_start=14, tiles="cartodbpositron")

    steps = sorted(valid["STEP_NUMBER"].dropna().unique())
    step_color_map = {step: STEP_COLORS[i % len(STEP_COLORS)] for i, step in enumerate(steps)}

    for _, row in valid.iterrows():
        geojson_str = row.get("GEOJSON")
        step = row["STEP_NUMBER"]
        color = step_color_map.get(step, "#333333")
        popup_text = f"Step {int(step)}: {row['STREET_NAME']}<br>{row['SIDE']} side<br>{row['FROM_STREET']} → {row['TO_STREET']}<br>CNN: {int(row['SEGMENT_CNN'])}"

        if geojson_str and pd.notna(geojson_str):
            try:
                geojson_obj = json.loads(geojson_str)
                folium.GeoJson(
                    geojson_obj,
                    style_function=lambda x, c=color: {
                        "color": c, "weight": 4, "opacity": 0.8
                    },
                    popup=folium.Popup(popup_text, max_width=300),
                ).add_to(m)
            except (json.JSONDecodeError, TypeError):
                folium.CircleMarker(
                    location=[row["LAT"], row["LNG"]],
                    radius=4, color=color, fill=True, popup=popup_text
                ).add_to(m)
        else:
            folium.CircleMarker(
                location=[row["LAT"], row["LNG"]],
                radius=4, color=color, fill=True, popup=popup_text
            ).add_to(m)

    return m

st.set_page_config(page_title="SF Street Sweeping Route Books", layout="wide")
st.title("SF Street Sweeping Route Books")
st.caption("Browse extracted route books from 55 DPW PDFs")

routes_df = load_routes()
seg_df = load_segments()

files_df = get_source_files(routes_df)

col_pdf, col_rb = st.columns([1, 1])
with col_pdf:
    file_options = files_df["SOURCE_FILE"].tolist()
    selected_file = st.selectbox("Source PDF", options=file_options, index=0)

if not selected_file:
    st.info("Select a source PDF above to view route books.")
    st.dataframe(files_df)
    st.stop()

rb_list = get_route_books(routes_df, selected_file)
with col_rb:
    selected_rb = st.selectbox("Route Book", options=rb_list, index=0)

if not selected_rb:
    st.stop()

info = get_route_info(routes_df, selected_file, selected_rb)
header_cols = st.columns(4)
header_cols[0].metric("Route Book", selected_rb)
header_cols[1].metric("Zone", info["ZONE"] if info else "-")
header_cols[2].metric("Days/Week", int(info["DAYS"]) if info else 0)
header_cols[3].metric("Total Steps", int(info["TOTAL_STEPS"]) if info else 0)

st.markdown("---")

schedule_df = get_schedule(routes_df, selected_file, selected_rb)
days_available = [d for d in schedule_df["Day"].dropna().unique().tolist() if d]

day_filter = None
if len(days_available) > 1:
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_options = ["All Days"] + sorted(days_available, key=lambda d: day_order.index(d) if d in day_order else 8)
    selected_day = st.radio("Filter by day", options=day_options, index=0, horizontal=True)
    if selected_day and selected_day != "All Days":
        day_filter = selected_day

tab_schedule, tab_map, tab_segments = st.tabs(["Schedule & Route Steps", "Route Map", "CNN Street Segments"])

with tab_schedule:
    st.subheader("Schedule")
    st.dataframe(schedule_df)
    st.subheader("Route Steps")
    steps_df = get_route_steps(routes_df, selected_file, selected_rb, day_filter)
    display_cols = ["Step", "Street", "Side", "From", "To"]
    if day_filter is None and len(days_available) > 1:
        display_cols = ["Day"] + display_cols
    display_cols += ["From CNN", "To CNN"]
    st.dataframe(steps_df[display_cols])
    matched = steps_df["From CNN"].notna().sum()
    total = len(steps_df)
    if total > 0:
        st.caption(f"CNN match rate: {matched}/{total} steps ({matched/total*100:.0f}%)")

with tab_map:
    st.subheader("Route Map")
    map_seg_df = get_segments(seg_df, selected_file, selected_rb, day_filter)
    if len(map_seg_df) == 0:
        st.warning("No mapped segments available for this route book.")
    else:
        m = build_leaflet_map(map_seg_df)
        if m:
            legend_html = "<b>Step Legend:</b> "
            steps = sorted(map_seg_df["STEP_NUMBER"].dropna().unique())
            for i, step in enumerate(steps[:15]):
                color = STEP_COLORS[i % len(STEP_COLORS)]
                legend_html += f'<span style="color:{color}; font-weight:bold;">■ {int(step)}</span> '
            if len(steps) > 15:
                legend_html += f"... +{len(steps)-15} more"
            st.markdown(legend_html, unsafe_allow_html=True)
            st_folium(m, width=None, height=500, use_container_width=True)
            st.caption(f"{map_seg_df['SEGMENT_CNN'].nunique()} distinct CNN blocks mapped as line segments")
        else:
            st.warning("Could not parse geometry for this route.")

with tab_segments:
    st.subheader("CNN Street Segments")
    seg_display = get_segments(seg_df, selected_file, selected_rb, day_filter)
    if len(seg_display) == 0:
        st.info("No CNN segments resolved for this route book.")
    else:
        display_seg = seg_display[["STEP_NUMBER", "STREET_NAME", "SIDE", "FROM_STREET",
                                    "TO_STREET", "DAY_OF_WEEK", "SEGMENT_CNN", "LAT", "LNG"]].copy()
        display_seg.columns = ["Step", "Street", "Side", "From", "To", "Day", "CNN", "Lat", "Lng"]
        st.dataframe(display_seg)
        st.caption(f"{len(seg_display)} segment rows | {seg_display['SEGMENT_CNN'].nunique()} distinct CNNs")
