import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml
from django.test import SimpleTestCase

from fishweb import catalog, worker


class LocalDatasetTests(SimpleTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root_patch = patch.object(catalog, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        catalog._scan_dataset.cache_clear()
        self.addCleanup(catalog._scan_dataset.cache_clear)
        self.private = self.root / "runs/web/dataset/fish.yaml"
        self.private.parent.mkdir(parents=True)
        self.config = {"path": "data/processed/fish", "names": ["Mola"],
                       "odp_meta": {"split": {"counts": {"train": 3, "val": 1, "test": 1}},
                                    "contract_fingerprint": "original-fingerprint"}}
        self.private.write_text(yaml.safe_dump(self.config), encoding="utf-8")

    def integration(self, **overrides):
        doc = {"identity_ready": True, "dataset_yaml": "runs/web/dataset/fish.yaml",
               "validation_report": "runs/data_validation/local/report.json"}
        doc.update(overrides)
        (self.private.parent / "integration.json").write_text(json.dumps(doc), encoding="utf-8")

    def test_private_configuration_requires_successful_identity_marker(self):
        self.assertEqual(catalog.dataset_yaml_path(), catalog.DATASET_YAML)
        self.integration(identity_ready=False)
        self.assertEqual(catalog.dataset_yaml_path(), catalog.DATASET_YAML)
        self.integration()
        self.assertEqual(catalog.dataset_yaml_path(), self.private)
        self.private.unlink()
        self.assertEqual(catalog.dataset_yaml_path(), catalog.DATASET_YAML)

    @patch("fishweb.catalog.time.monotonic", return_value=123)
    def test_counts_samples_and_validation_come_from_actual_files_and_report(self, _clock):
        self.integration()
        report = self.root / "runs/data_validation/local/report.json"
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps({"yaml_path": str(self.private), "overall_severity": "PASS",
                                      "results": [{"severity": "PASS"} for _ in range(15)]}), encoding="utf-8")
        for split in ("train", "val", "test"):
            images = self.root / "data/processed/fish/images" / split
            labels = self.root / "data/processed/fish/labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (images / "sample.jpg").write_bytes(b"fixture")
            (labels / "sample.txt").write_text("0 .5 .5 .2 .2\n0 .3 .3 .1 .1\n", encoding="utf-8")
        data = catalog.dataset_info()
        self.assertTrue(data["local"]["available"])
        self.assertEqual(data["local"]["counts"], {"train": 1, "val": 1, "test": 1})
        self.assertEqual(data["local"]["instances"], 6)
        self.assertEqual(data["classes"][0]["instances"], 6)
        self.assertEqual(data["counts"]["total"], 5)  # documented counts stay distinct
        self.assertTrue(data["validation"]["measured_locally"])
        self.assertEqual(data["validation"]["checks_passed"], 15)
        self.assertEqual(len(catalog.sample_paths()), 3)
        with patch.object(catalog, "_direct_files", side_effect=AssertionError("scan was not cached")):
            # Metadata stays fresh while expensive image/label reads are cached.
            self.assertEqual(catalog._local_scan(self.config, 1)[1], [6])

    def test_outside_report_is_never_exposed_as_local_validation(self):
        self.integration(validation_report="../external/report.json")
        self.assertFalse(catalog.local_validation()["measured_locally"])

    def test_training_passes_private_yaml_to_existing_platform_command(self):
        self.integration()
        config = {"model": "pretrained/yolo11n.pt", "epochs": 1, "batch": 8, "imgsz": 640, "device": "cpu"}
        with patch.object(catalog, "resolve_model", return_value=self.root / "models/pretrained/yolo11n.pt"), \
                patch.object(worker.subprocess, "run") as run:
            worker.train(config, self.root / "fake-job")
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--data") + 1], str(self.private))
        self.assertIn("od_platform.cli.train_model", command)

    def test_weight_configuration_is_distinct_from_an_existing_binary(self):
        config = self.root / "apps/platform/configs/runtime/demo.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("weight: fish-trained-best\n", encoding="utf-8")
        result = catalog.model_configuration([{"name": "yolo11n.pt", "kind": "other"}])
        self.assertEqual(result["configured_weight"], "fish-trained-best")
        self.assertEqual(result["expected_path"], "models/trained/fish-trained-best.pt")
        self.assertFalse(result["file_available"])
        self.assertEqual(result["fish_candidates"], 0)
        self.assertIn("fish-trained-best.pt", result["message"])
        self.assertTrue(catalog.model_configuration([{"name": "fish-trained-best.pt", "kind": "fish"}])["file_available"])
