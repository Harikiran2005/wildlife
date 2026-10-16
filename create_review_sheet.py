
import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ===================== PATHS =====================

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "ena24_images"
RESULTS_FILE = ROOT / "ena24_results" / "ena24_full_results.json"
CANDIDATES_FILE = ROOT / "risk_reports" / "manual_review_candidates.csv"
OUTPUT_FILE = ROOT / "risk_reports" / "manual_review_contact_sheet.jpg"

# ===================== SETTINGS =====================

MAX_IMAGES = 7
THUMBNAIL_WIDTH = 480
THUMBNAIL_HEIGHT = 320
LABEL_HEIGHT = 48
COLUMNS = 2

COLORS = {
    "animal": (30, 180, 70),
    "person": (255, 70, 70),
    "vehicle": (50, 130, 255),
}

# ===================== MAIN FUNCTION =====================

def main():
    # Check required files
    for path in (RESULTS_FILE, CANDIDATES_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load MegaDetector results
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Index results by image filename
    results_by_name = {
        Path(item["file"].replace("\\", "/")).name: item
        for item in results
    }

    # Load candidate images
    candidates = pd.read_csv(CANDIDATES_FILE)

    # Select images with both co-occurrence flags
    both_flags = (
        candidates["animal_person_cooccurrence"]
        .astype(str).str.lower().eq("true")
        &
        candidates["animal_vehicle_cooccurrence"]
        .astype(str).str.lower().eq("true")
    )

    candidates = candidates.loc[both_flags].head(MAX_IMAGES)

    if candidates.empty:
        print("No images with both co-occurrence flags were found.")
        return

    font = ImageFont.load_default()
    cards = []

    for _, row in candidates.iterrows():
        filename = Path(
            str(row["image"]).replace("\\", "/")
        ).name

        image_path = IMAGE_DIR / filename

        if not image_path.exists():
            print(f"Image missing: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        result = results_by_name.get(filename)

        if result is None:
            print(f"No prediction result found for {filename}")
            continue

        original_width, original_height = image.size

        for det in result.get("detections", []):
            category = str(
                det.get("category", "unknown")
            ).lower()

            bbox = det.get("bbox")

            if not bbox or len(bbox) != 4:
                continue

            # The supplied examples appear to use
            # [x1, y1, x2, y2] pixel coordinates.
            x1, y1, x2, y2 = map(float, bbox)

            # Convert normalized coordinates if applicable.
            # MegaDetector outputs commonly use normalized
            # [x, y, width, height], so verify the schema
            # before interpreting unusual coordinates.
            if (
                min(x1, y1, x2, y2) >= 0
                and max(x1, y1, x2, y2) <= 1.0
            ):
                x1 *= original_width
                x2 *= original_width
                y1 *= original_height
                y2 *= original_height

            # Keep coordinates inside the image
            x1 = max(0, min(original_width - 1, x1))
            y1 = max(0, min(original_height - 1, y1))
            x2 = max(0, min(original_width - 1, x2))
            y2 = max(0, min(original_height - 1, y2))

            # Skip invalid or zero-area boxes
            if x2 <= x1 or y2 <= y1:
                continue

            color = COLORS.get(category, (255, 220, 0))

            draw.rectangle(
                (x1, y1, x2, y2),
                outline=color,
                width=4,
            )

            confidence = det.get("confidence")
            label = category

            if confidence is not None:
                label += f" {float(confidence):.2f}"

            label_y = max(0, int(y1) - 15)
            draw.text(
                (int(x1) + 2, label_y),
                label,
                fill=color,
                font=font,
            )

        # Resize image for contact sheet
        image.thumbnail((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT))

        card = Image.new(
            "RGB",
            (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT + LABEL_HEIGHT),
            "white",
        )

        x_offset = (THUMBNAIL_WIDTH - image.width) // 2
        y_offset = (THUMBNAIL_HEIGHT - image.height) // 2
        card.paste(image, (x_offset, y_offset))

        card_draw = ImageDraw.Draw(card)

        summary = (
            f"{filename} | Animals: {int(row['animals'])} "
            f"People: {int(row['people'])} "
            f"Vehicles: {int(row['vehicles'])}"
        )

        card_draw.text(
            (8, THUMBNAIL_HEIGHT + 5),
            summary,
            fill="black",
            font=font,
        )

        card_draw.text(
            (8, THUMBNAIL_HEIGHT + 23),
            "Animal-person + animal-vehicle co-occurrence",
            fill="black",
            font=font,
        )

        cards.append(card)
        print(f"Prepared: {filename}")

    if not cards:
        print("No candidate images could be processed.")
        return

    # Arrange cards in a grid
    rows = math.ceil(len(cards) / COLUMNS)

    sheet = Image.new(
        "RGB",
        (
            COLUMNS * THUMBNAIL_WIDTH,
            rows * (THUMBNAIL_HEIGHT + LABEL_HEIGHT),
        ),
        (235, 235, 235),
    )

    for index, card in enumerate(cards):
        x = (index % COLUMNS) * THUMBNAIL_WIDTH
        y = (index // COLUMNS) * (
            THUMBNAIL_HEIGHT + LABEL_HEIGHT
        )
        sheet.paste(card, (x, y))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUTPUT_FILE, quality=90)

    print(f"\nReview sheet saved to:\n{OUTPUT_FILE}")
    print(f"Images included: {len(cards)}")


if __name__ == "__main__":
    main()
