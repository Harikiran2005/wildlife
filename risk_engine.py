
import json
import csv
import math
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "ena24_results" / "ena24_full_results.json"
REPORT_DIR = BASE_DIR / "risk_reports"

# Configurable parameters
CONFIDENCE_THRESHOLD = 0.25
DUPLICATE_IOU_THRESHOLD = 0.70
PROXIMITY_THRESHOLD = 0.10  # Fraction of the image diagonal

CATEGORIES = ("animal", "person", "vehicle")


def get_bbox(detection):
    """
    Read bounding boxes in [x1, y1, x2, y2] pixel format.
    Returns None if coordinates are missing or invalid.
    """
    bbox = detection.get("bbox")

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = map(float, bbox)
    except (TypeError, ValueError):
        return None

    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def box_iou(box_a, box_b):
    """Calculate intersection over union for two bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def suppress_duplicates(detections):
    """
    Greedy non-maximum suppression within each category.
    Keeps the highest-confidence box when same-category boxes
    overlap substantially. Detections without boxes are retained.
    """
    kept = []

    for category in CATEGORIES:
        category_detections = [
            d for d in detections if d["category"] == category
        ]

        with_boxes = [d for d in category_detections if d["bbox"] is not None]
        without_boxes = [d for d in category_detections if d["bbox"] is None]

        with_boxes.sort(key=lambda d: d["confidence"], reverse=True)
        selected = []

        for candidate in with_boxes:
            duplicate = any(
                box_iou(candidate["bbox"], previous["bbox"])
                >= DUPLICATE_IOU_THRESHOLD
                for previous in selected
            )

            if not duplicate:
                selected.append(candidate)

        kept.extend(selected)
        kept.extend(without_boxes)

    return kept


def prepare_detections(image):
    """Validate, confidence-filter, and deduplicate detections."""
    raw = image.get("detections", []) if isinstance(image, dict) else []

    if not isinstance(raw, list):
        return []

    valid = []

    for detection in raw:
        if not isinstance(detection, dict):
            continue

        category = detection.get("category")
        if category not in CATEGORIES:
            continue

        try:
            confidence = float(detection.get("confidence", 1.0))
        except (TypeError, ValueError):
            continue

        if not math.isfinite(confidence):
            continue

        if not 0.0 <= confidence <= 1.0:
            continue

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        valid.append({
            "category": category,
            "confidence": confidence,
            "bbox": get_bbox(detection),
        })

    return suppress_duplicates(valid)


def get_image_dimensions(image):
    """
    Obtain the actual image dimensions from its file.

    Returns (width, height) if the image can be opened.
    Returns (None, None) if the image path is missing or unavailable.
    """
    if not isinstance(image, dict):
        return None, None

    filename = image.get("file") or image.get("image")

    if not isinstance(filename, str) or not filename.strip():
        return None, None

    image_path = Path(filename)

    if not image_path.is_absolute():
        image_path = BASE_DIR / image_path

    try:
        with Image.open(image_path) as img:
            width, height = img.size

        if width > 0 and height > 0:
            return width, height

    except (OSError, ValueError):
        pass

    return None, None


def boxes_are_near(box_a, box_b, image_diagonal):
    """
    Estimate image-space proximity using the minimum distance
    between two bounding rectangles.

    This is not a real-world distance measurement.
    """
    if box_a is None or box_b is None or image_diagonal <= 0:
        return False

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    gap_x = max(0.0, bx1 - ax2, ax1 - bx2)
    gap_y = max(0.0, by1 - ay2, ay1 - by2)

    distance = math.hypot(gap_x, gap_y)

    return distance / image_diagonal <= PROXIMITY_THRESHOLD


def get_image_metrics(image, image_width=None, image_height=None):
    """
    Return filtered detections, category counts, and interaction flags.

    Uses actual image dimensions when available. If dimensions are
    unavailable, uses the previous bounding-box-based fallback to
    preserve compatibility with existing synthetic test inputs.
    """
    detections = prepare_detections(image)

    counts = {
        category: sum(d["category"] == category for d in detections)
        for category in CATEGORIES
    }

    if (
        image_width is not None
        and image_height is not None
        and image_width > 0
        and image_height > 0
    ):
        diagonal = math.hypot(image_width, image_height)
    else:
        boxes = [
            d["bbox"] for d in detections
            if d["bbox"] is not None
        ]

        if boxes:
            max_x = max(box[2] for box in boxes)
            max_y = max(box[3] for box in boxes)
            diagonal = math.hypot(max_x, max_y)
        else:
            diagonal = 0.0

    animals = [d for d in detections if d["category"] == "animal"]
    people = [d for d in detections if d["category"] == "person"]
    vehicles = [d for d in detections if d["category"] == "vehicle"]

    animal_person = any(
        boxes_are_near(a["bbox"], p["bbox"], diagonal)
        for a in animals
        for p in people
    )

    animal_vehicle = any(
        boxes_are_near(a["bbox"], v["bbox"], diagonal)
        for a in animals
        for v in vehicles
    )

    return counts, animal_person, animal_vehicle


def analyze_images(data):
    """Analyze detections and return summary metrics and image rows."""
    if not isinstance(data, list):
        raise ValueError("Expected the detection results to be a list.")

    if not data:
        raise ValueError("The detection results contain zero images.")

    totals = {category: 0 for category in CATEGORIES}
    images_with = {category: 0 for category in CATEGORIES}

    animal_person_events = 0
    animal_vehicle_events = 0
    image_rows = []

    for image in data:
        # Use actual dimensions whenever the corresponding image is available.
        width, height = get_image_dimensions(image)

        counts, animal_person, animal_vehicle = get_image_metrics(
            image,
            image_width=width,
            image_height=height,
        )

        for category in CATEGORIES:
            totals[category] += counts[category]
            images_with[category] += counts[category] > 0

        animal_person_events += animal_person
        animal_vehicle_events += animal_vehicle

        filename = "unknown"
        if isinstance(image, dict):
            filename = image.get("file") or image.get("image") or "unknown"

        image_rows.append({
            "image": filename,
            "animals": counts["animal"],
            "people": counts["person"],
            "vehicles": counts["vehicle"],
            "animal_person_cooccurrence": animal_person,
            "animal_vehicle_cooccurrence": animal_vehicle,
        })

    num_images = len(data)

    wildlife_activity = images_with["animal"] / num_images * 100
    human_activity = images_with["person"] / num_images * 100
    vehicle_activity = images_with["vehicle"] / num_images * 100

    human_wildlife_rate = animal_person_events / num_images * 100
    vehicle_wildlife_rate = animal_vehicle_events / num_images * 100

    # Existing heuristic formula retained for comparability.
    risk_score = round(
        0.35 * wildlife_activity
        + 0.25 * human_activity
        + 0.15 * vehicle_activity
        + 0.15 * human_wildlife_rate
        + 0.10 * vehicle_wildlife_rate,
        2,
    )

    if risk_score <= 30:
        risk_level = "LOW"
    elif risk_score <= 60:
        risk_level = "MODERATE"
    elif risk_score <= 80:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    summary = {
        "images_processed": num_images,
        "total_animals": totals["animal"],
        "total_people": totals["person"],
        "total_vehicles": totals["vehicle"],
        "wildlife_activity_percent": round(wildlife_activity, 2),
        "human_activity_percent": round(human_activity, 2),
        "vehicle_activity_percent": round(vehicle_activity, 2),
        "animal_person_events": animal_person_events,
        "animal_vehicle_events": animal_vehicle_events,
        "human_wildlife_rate_percent": round(human_wildlife_rate, 2),
        "vehicle_wildlife_rate_percent": round(vehicle_wildlife_rate, 2),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "duplicate_iou_threshold": DUPLICATE_IOU_THRESHOLD,
        "proximity_threshold": PROXIMITY_THRESHOLD,
        "risk_score_type": (
            "Heuristic based on confidence-filtered image detections "
            "and estimated image-space proximity. Not a validated "
            "collision probability."
        ),
    }

    return summary, image_rows


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Detection results not found: {INPUT_FILE}"
        )

    with INPUT_FILE.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    summary, image_rows = analyze_images(data)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Wildlife Corridor Risk Assessment ===")
    print(f"Images processed          : {summary['images_processed']}")
    print(f"Total animals detected    : {summary['total_animals']}")
    print(f"Total people detected     : {summary['total_people']}")
    print(f"Total vehicles detected   : {summary['total_vehicles']}")

    print("\n--- Activity ---")
    print(
        f"Wildlife activity         : "
        f"{summary['wildlife_activity_percent']:.2f}/100"
    )
    print(
        f"Human activity            : "
        f"{summary['human_activity_percent']:.2f}/100"
    )
    print(
        f"Vehicle activity          : "
        f"{summary['vehicle_activity_percent']:.2f}/100"
    )

    print("\n--- Spatial interaction estimates ---")
    print(f"Animal-person events      : {summary['animal_person_events']}")
    print(f"Animal-vehicle events     : {summary['animal_vehicle_events']}")
    print(
        f"Human-wildlife rate       : "
        f"{summary['human_wildlife_rate_percent']:.2f}%"
    )
    print(
        f"Vehicle-wildlife rate     : "
        f"{summary['vehicle_wildlife_rate_percent']:.2f}%"
    )

    print("\n--- Risk Assessment ---")
    print(f"Risk Score                : {summary['risk_score']:.2f}/100")
    print(f"Risk Level                : {summary['risk_level']}")
    print(f"Confidence threshold      : {CONFIDENCE_THRESHOLD:.2f}")

    with (REPORT_DIR / "risk_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=4)

    with (REPORT_DIR / "image_level_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = [
            "image",
            "animals",
            "people",
            "vehicles",
            "animal_person_cooccurrence",
            "animal_vehicle_cooccurrence",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(image_rows)

    print(f"\nReports saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
