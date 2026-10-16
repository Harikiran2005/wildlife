
import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "ena24_results" / "ena24_full_results.json"

with INPUT_FILE.open("r", encoding="utf-8-sig") as f:
    data = json.load(f)

confidence_scores = []
category_counts = Counter()
missing_confidence = 0
invalid_confidence = 0

for image in data:
    for detection in image.get("detections", []):
        if not isinstance(detection, dict):
            continue

        category_counts[detection.get("category", "unknown")] += 1

        confidence = detection.get("confidence")

        if confidence is None:
            missing_confidence += 1
            continue

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            invalid_confidence += 1
            continue

        if not 0 <= confidence <= 1:
            invalid_confidence += 1
            continue

        confidence_scores.append(
            (confidence, detection.get("category", "unknown"))
        )

print("\n=== Detection Confidence Analysis ===")
print(f"Images analysed          : {len(data)}")
print(f"Detections by category   : {dict(category_counts)}")
print(f"Valid confidence scores  : {len(confidence_scores)}")
print(f"Missing confidence       : {missing_confidence}")
print(f"Invalid confidence       : {invalid_confidence}")

if confidence_scores:
    scores = sorted(score for score, _ in confidence_scores)

    def percentile(values, p):
        position = (len(values) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        fraction = position - lower
        return values[lower] * (1 - fraction) + values[upper] * fraction

    print("\n--- Confidence distribution ---")
    print(f"Minimum                  : {scores[0]:.4f}")
    print(f"25th percentile          : {percentile(scores, 0.25):.4f}")
    print(f"Median                   : {percentile(scores, 0.50):.4f}")
    print(f"75th percentile          : {percentile(scores, 0.75):.4f}")
    print(f"Maximum                  : {scores[-1]:.4f}")

    print("\n--- Threshold comparison ---")
    for threshold in (0.10, 0.25, 0.50, 0.75, 0.90):
        kept = sum(score >= threshold for score in scores)
        removed = len(scores) - kept
        print(
            f"Threshold {threshold:.2f}: "
            f"{kept} retained, {removed} excluded"
        )

    print("\n--- Category-level confidence ---")
    for category in sorted({cat for _, cat in confidence_scores}):
        category_scores = [
            score for score, cat in confidence_scores
            if cat == category
        ]
        avg = sum(category_scores) / len(category_scores)
        print(
            f"{category:10s}: count={len(category_scores)}, "
            f"mean={avg:.4f}, min={min(category_scores):.4f}"
        )
