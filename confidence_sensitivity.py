
import json
import csv
from pathlib import Path

from risk_engine import analyze_images

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "ena24_results" / "ena24_full_results.json"
OUTPUT_FILE = BASE_DIR / "risk_reports" / "confidence_sensitivity.csv"

THRESHOLDS = [0.20, 0.25, 0.30, 0.40, 0.50, 0.75]

with INPUT_FILE.open("r", encoding="utf-8-sig") as f:
    original_data = json.load(f)

if not isinstance(original_data, list) or not original_data:
    raise ValueError("Input JSON must contain a non-empty list of images.")

results = []

print("\n=== Confidence Threshold Sensitivity Analysis ===")
print(f"Images in dataset: {len(original_data)}")

for threshold in THRESHOLDS:
    filtered_data = []

    for image in original_data:
        if not isinstance(image, dict):
            filtered_data.append({"detections": []})
            continue

        detections = image.get("detections", [])
        if not isinstance(detections, list):
            detections = []

        filtered_detections = [
            detection
            for detection in detections
            if isinstance(detection, dict)
            and isinstance(detection.get("confidence"), (int, float))
            and threshold <= detection["confidence"] <= 1
            and detection["confidence"] >= 0
        ]

        filtered_data.append({
            **image,
            "detections": filtered_detections
        })

    summary, _ = analyze_images(filtered_data)

    row = {
        "threshold": threshold,
        "images_processed": summary["images_processed"],
        "animals": summary["total_animals"],
        "people": summary["total_people"],
        "vehicles": summary["total_vehicles"],
        "animal_person_events": summary["animal_person_events"],
        "animal_vehicle_events": summary["animal_vehicle_events"],
        "wildlife_activity_percent": summary["wildlife_activity_percent"],
        "human_activity_percent": summary["human_activity_percent"],
        "vehicle_activity_percent": summary["vehicle_activity_percent"],
        "risk_score": summary["risk_score"],
        "risk_level": summary["risk_level"],
    }

    results.append(row)

    print(
        f"Threshold {threshold:.2f} | "
        f"Animals: {row['animals']} | "
        f"People: {row['people']} | "
        f"Vehicles: {row['vehicles']} | "
        f"Animal-person events: {row['animal_person_events']} | "
        f"Animal-vehicle events: {row['animal_vehicle_events']} | "
        f"Risk: {row['risk_score']:.2f}/100 ({row['risk_level']})"
    )

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to: {OUTPUT_FILE}")
print("Original detection JSON and risk reports were not modified.")
