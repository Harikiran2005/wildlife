
import json
from pathlib import Path

import pandas as pd
import streamlit as st

REPORT_DIR = Path("risk_reports")
SUMMARY_FILE = REPORT_DIR / "risk_summary.json"
CSV_FILE = REPORT_DIR / "image_level_results.csv"

st.set_page_config(
    page_title="Wildlife Corridor Monitor",
    page_icon="🐾",
    layout="wide"
)

st.title("🐾 Wildlife Corridor Monitoring System")
st.caption("AI-powered camera-trap analytics using MegaDetector")

if not SUMMARY_FILE.exists() or not CSV_FILE.exists():
    st.error("Reports not found. Run python risk_engine.py first.")
    st.stop()

with SUMMARY_FILE.open("r", encoding="utf-8") as f:
    summary = json.load(f)

df = pd.read_csv(CSV_FILE)

st.subheader("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Images Processed", f"{summary['images_processed']:,}")
c2.metric("Animal Detections", f"{summary['total_animals']:,}")
c3.metric("People Detected", f"{summary['total_people']:,}")
c4.metric("Vehicles Detected", f"{summary['total_vehicles']:,}")

st.subheader("Corridor Risk Assessment")

risk = summary["risk_score"]
level = summary["risk_level"]

c1, c2 = st.columns(2)
c1.metric("Heuristic Risk Score", f"{risk:.2f}/100")
c2.metric("Risk Category", level)

st.progress(min(max(risk / 100, 0.0), 1.0))

st.info(
    "The risk score is a heuristic based on detection activity "
    "and category co-occurrence. It is not a validated collision probability."
)

st.subheader("Detection Activity")

activity = pd.DataFrame({
    "Category": ["Wildlife", "People", "Vehicles"],
    "Images (%)": [
        summary["wildlife_activity_percent"],
        summary["human_activity_percent"],
        summary["vehicle_activity_percent"],
    ],
})
st.bar_chart(activity, x="Category", y="Images (%)")

st.subheader("Potential Co-occurrence")

c1, c2 = st.columns(2)
c1.metric(
    "Animal + Person Images",
    summary["animal_person_events"]
)
c2.metric(
    "Animal + Vehicle Images",
    summary["animal_vehicle_events"]
)

st.subheader("Image-Level Results")

search = st.text_input("Search image filename")
filtered = df.copy()

if search.strip():
    filtered = filtered[
        filtered["image"].astype(str).str.contains(
            search.strip(), case=False, na=False
        )
    ]

interaction_filter = st.selectbox(
    "Filter by co-occurrence",
    [
        "All images",
        "Animal and person",
        "Animal and vehicle",
        "Both types",
    ],
)

if interaction_filter == "Animal and person":
    filtered = filtered[
        filtered["animal_person_cooccurrence"].astype(bool)
    ]
elif interaction_filter == "Animal and vehicle":
    filtered = filtered[
        filtered["animal_vehicle_cooccurrence"].astype(bool)
    ]
elif interaction_filter == "Both types":
    filtered = filtered[
        filtered["animal_person_cooccurrence"].astype(bool)
        & filtered["animal_vehicle_cooccurrence"].astype(bool)
    ]

st.dataframe(filtered, use_container_width=True, hide_index=True)
st.caption(f"Showing {len(filtered):,} of {len(df):,} image records.")

st.download_button(
    "Download filtered image results (CSV)",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="wildlife_image_results.csv",
    mime="text/csv",
)

st.download_button(
    "Download risk summary (JSON)",
    data=json.dumps(summary, indent=4),
    file_name="risk_summary.json",
    mime="application/json",
)
