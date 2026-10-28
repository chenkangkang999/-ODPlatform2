import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase

from fishweb import catalog


class ProvidedValidationTests(SimpleTestCase):
    def record(self):
        return {"provenance": "provided_validation_record", "overall": "PASS",
                "checks_passed": 16, "checks_total": 16, "images": 4560,
                "checks": [{"name": str(index), "severity": "PASS"} for index in range(16)],
                "private_path": "must-not-be-exposed"}

    def load(self, value):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs/fish_data_validation.json").write_text(json.dumps(value), encoding="utf-8")
            with patch.object(catalog, "ROOT", root):
                return catalog.validation_record()

    def test_portable_sixteen_checks_are_not_claimed_as_current_server_measurements(self):
        result = self.load(self.record())
        self.assertEqual(result["checks_passed"], 16)
        self.assertEqual(result["images"], 4560)
        self.assertFalse(result["measured_locally"])
        self.assertNotIn("private_path", result)

    def test_failed_or_inconsistent_records_are_not_displayed_as_pass(self):
        record = self.record()
        record["checks"][0]["severity"] = "FAIL"
        self.assertIsNone(self.load(record))
        record = self.record()
        record["checks_total"] = 15
        self.assertIsNone(self.load(record))
