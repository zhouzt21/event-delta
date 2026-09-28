import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.audit_sevd import audit_recording, build_manifest_rows, create_visualization


class AuditSevdTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.recording = self.root / "extracted" / "fixed-3-archive"
        for name in ("rgb", "event", "flow", "annotation", "semantic", "instance"):
            (self.recording / name).mkdir(parents=True, exist_ok=True)
        (self.recording / "metadata.json").write_text(
            json.dumps({"timestamp_unit": "microseconds", "source": "synthetic-test"}), encoding="utf-8"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def add_frame(self, identifier: int, timestamps=None):
        stem = f"frame_{identifier:06d}"
        image = Image.new("RGB", (8, 6), (10, 20, 30))
        image.save(self.recording / "rgb" / f"{stem}.png")
        image.save(self.recording / "event" / f"{stem}_preview.png")
        image.save(self.recording / "semantic" / f"{stem}.png")
        image.save(self.recording / "instance" / f"{stem}.png")
        t = np.asarray(timestamps if timestamps is not None else [identifier * 10, identifier * 10 + 1])
        np.savez(
            self.recording / "event" / f"{stem}.npz",
            x=np.array([1, 2]),
            y=np.array([2, 3]),
            p=np.array([0, 1]),
            t=t,
        )
        np.savez(self.recording / "flow" / f"{stem}.npz", flow=np.zeros((6, 8, 2), dtype=np.float32))
        annotation = {
            "objects": [
                {
                    "class_id": 2,
                    "object_id": 99,
                    "bbox": {"x_min": 1, "y_min": 1, "x_max": 5, "y_max": 4},
                }
            ]
        }
        (self.recording / "annotation" / f"{stem}.json").write_text(json.dumps(annotation), encoding="utf-8")

    def audit(self):
        report, _mapping, _recording_root = audit_recording(self.root, "fixed-3")
        return report

    def test_normal_pairing(self):
        for identifier in range(1, 4):
            self.add_frame(identifier)
        report = self.audit()
        self.assertEqual(report["frame_id_alignment"]["intersection"], [1, 2, 3])
        self.assertEqual(report["file_counts"]["rgb"], 3)
        self.assertFalse(report["frame_id_alignment"]["duplicates_by_modality"]["rgb"])
        sample = report["annotation_samples"][0]
        self.assertEqual(sample["class_id_fields"], ["class_id"])
        self.assertEqual(sample["object_id_fields"], ["object_id"])
        self.assertIn("bbox", sample["bbox_fields"])

    def test_missing_one_rgb_frame(self):
        self.add_frame(1)
        self.add_frame(2)
        (self.recording / "rgb" / "frame_000002.png").unlink()
        report = self.audit()
        self.assertEqual(report["frame_id_alignment"]["missing_by_modality"]["rgb"], [2])
        self.assertEqual(report["frame_id_alignment"]["intersection"], [1])

    def test_non_monotonic_event_timestamp(self):
        self.add_frame(1, timestamps=[10, 9, 11])
        report = self.audit()
        codes = {item["code"] for item in report["warnings"]}
        self.assertIn("non_monotonic_event_timestamp", codes)
        self.assertFalse(report["event_npz_samples"][0]["timestamp_monotonic_nondecreasing"])

    def test_unknown_npz_fields(self):
        self.add_frame(1)
        path = self.recording / "event" / "frame_000001.npz"
        np.savez(path, unknown=np.array([1, 2, 3]))
        report = self.audit()
        codes = {item["code"] for item in report["warnings"]}
        self.assertIn("unknown_event_npz_fields", codes)
        self.assertIsNone(report["event_npz_samples"][0]["recognized_key_mapping"])

    def test_duplicate_frame_id(self):
        self.add_frame(1)
        image = Image.new("RGB", (8, 6))
        image.save(self.recording / "rgb" / "rgb_copy_000001.png")
        report = self.audit()
        duplicates = report["frame_id_alignment"]["duplicates_by_modality"]["rgb"]
        self.assertIn("1", duplicates)
        self.assertIn("duplicate_frame_id", {item["code"] for item in report["warnings"]})

    def test_twenty_row_manifest_and_visualization(self):
        for identifier in range(1, 21):
            self.add_frame(identifier)
        report, mapping, _recording_root = audit_recording(self.root, "fixed-3")
        self.assertTrue(report["mapping_preview_eligible"])
        rows = build_manifest_rows("fixed-3", mapping, self.root)
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(not Path(row["rgb_path"]).is_absolute() for row in rows))
        destination = self.root / "alignment.png"
        created, reason = create_visualization(rows, self.root, destination)
        self.assertTrue(created, reason)
        self.assertTrue(destination.is_file())


if __name__ == "__main__":
    unittest.main()
