
import unittest
from risk_engine import analyze_images


class TestRiskEngine(unittest.TestCase):

    def test_animal_only(self):
        summary, _ = analyze_images([
            {"detections": [
                {
                    "category": "animal",
                    "confidence": 0.95,
                    "bbox": [10, 10, 50, 50],
                }
            ]}
        ])
        self.assertEqual(summary["total_animals"], 1)

    def test_animal_person_cooccurrence(self):
        summary, _ = analyze_images([
            {"detections": [
                {
                    "category": "animal",
                    "confidence": 0.95,
                    "bbox": [10, 10, 50, 50],
                },
                {
                    "category": "person",
                    "confidence": 0.90,
                    "bbox": [52, 10, 90, 50],
                },
            ]}
        ])
        self.assertEqual(summary["animal_person_events"], 1)

    def test_animal_vehicle_cooccurrence(self):
        summary, _ = analyze_images([
            {"detections": [
                {
                    "category": "animal",
                    "confidence": 0.95,
                    "bbox": [10, 10, 50, 50],
                },
                {
                    "category": "vehicle",
                    "confidence": 0.90,
                    "bbox": [52, 10, 90, 50],
                },
            ]}
        ])
        self.assertEqual(summary["animal_vehicle_events"], 1)

    def test_empty_dataset(self):
        with self.assertRaises(ValueError):
            analyze_images([])

    def test_empty_detections(self):
        summary, _ = analyze_images([
            {"detections": []}
        ])
        self.assertEqual(summary["total_animals"], 0)
        self.assertEqual(summary["total_people"], 0)
        self.assertEqual(summary["total_vehicles"], 0)

    def test_unknown_category(self):
        summary, _ = analyze_images([
            {"detections": [
                {
                    "category": "airplane",
                    "confidence": 0.95,
                    "bbox": [10, 10, 50, 50],
                }
            ]}
        ])
        self.assertEqual(summary["total_animals"], 0)
        self.assertEqual(summary["total_people"], 0)
        self.assertEqual(summary["total_vehicles"], 0)


if __name__ == "__main__":
    unittest.main()
