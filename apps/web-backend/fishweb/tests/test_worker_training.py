import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import skipUnless
from unittest.mock import patch

import yaml
from django.test import SimpleTestCase

from fishweb import catalog, worker


PLATFORM_CLI_AVAILABLE = all(importlib.util.find_spec(name) is not None
                             for name in ("pydantic", "colorlog"))


class TrainingAdapterTests(SimpleTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.output = self.root / "runs/web/test-job"
        self.baseline = self.root / "apps/platform/configs/runtime/train.yaml"
        self.baseline.parent.mkdir(parents=True)
        # Values intentionally differ from Pydantic defaults: a batch-only
        # private file would silently lose these existing training settings.
        self.baseline_values = {"batch": 16, "workers": 0, "seed": 42,
                                "lr0": 0.002, "patience": 17, "amp": False}
        self.baseline.write_text(yaml.safe_dump(self.baseline_values), encoding="utf-8")
        self.original_baseline = self.baseline.read_bytes()
        self.dataset = self.root / "runs/web/dataset/fish.yaml"
        self.weight = self.root / "models/pretrained/yolo11n.pt"
        self.params = {"model": "pretrained/yolo11n.pt", "epochs": 15,
                       "batch": 8, "imgsz": 640, "device": "cpu"}

    def worker_command(self):
        with patch.object(worker, "ROOT", self.root), \
                patch.object(catalog, "dataset_yaml_path", return_value=self.dataset), \
                patch.object(catalog, "resolve_model", return_value=self.weight), \
                patch.object(worker.subprocess, "run") as process:
            worker.train(self.params, self.output)
        process.assert_called_once()
        self.assertEqual(process.call_args.kwargs, {"cwd": self.root, "check": True})
        return process.call_args.args[0]

    def test_private_yaml_preserves_baseline_and_integer_batch_without_engine_dependencies(self):
        command = self.worker_command()
        self.assertNotIn("--batch", command)
        self.assertNotIn("--skip-validate", command)
        self.assertEqual(command[:3], [sys.executable, "-m", "od_platform.cli.train_model"])
        config_path = Path(command[command.index("--config") + 1])
        self.assertEqual(config_path.parent, self.output.resolve())
        self.assertTrue(config_path.is_absolute())
        private = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(private, {**self.baseline_values, "batch": 8})
        self.assertIs(type(private["batch"]), int)
        self.assertEqual(command[command.index("--data") + 1], str(self.dataset))
        self.assertEqual(command[command.index("--model") + 1], str(self.weight))
        self.assertEqual(self.baseline.read_bytes(), self.original_baseline)

    @skipUnless(PLATFORM_CLI_AVAILABLE, "Optional platform CLI dependencies are not installed")
    def test_integer_batch_survives_actual_platform_parser_and_merge(self):
        from od_platform.cli import train_model
        from od_platform.runtime_config import build_train_config

        command = self.worker_command()
        args = train_model._build_parser().parse_args(command[3:])
        merged, _ = build_train_config(yaml_path=args.config,
                                       cli_args=train_model._collect_cli_overrides(args))
        self.assertIs(type(merged.batch), int)
        self.assertEqual(merged.batch, 8)
        self.assertEqual((merged.epochs, merged.imgsz, merged.device), (15, 640, "cpu"))
        self.assertEqual((merged.workers, merged.seed, merged.lr0, merged.patience, merged.amp),
                         (0, 42, 0.002, 17, False))
        self.assertEqual(self.baseline.read_bytes(), self.original_baseline)

    @skipUnless(PLATFORM_CLI_AVAILABLE, "Optional platform CLI dependencies are not installed")
    def test_platform_cli_still_rejects_unvalidated_identity_before_training(self):
        from od_platform.cli import train_model
        from od_platform.data_validation.identity import DataIdentity
        from od_platform.training import service

        command = self.worker_command()
        run = SimpleNamespace(run_id="test-web-training", run_dir=self.output)
        # Run the real CLI and training service, stopping at the real identity
        # gate so this regression test cannot launch a training run.
        with patch.object(sys, "argv", [command[2], *command[3:]]), \
                patch.object(train_model, "RunContext") as context, \
                patch.object(train_model, "get_logger"), \
                patch.object(service, "reconcile_data_identity",
                             return_value=DataIdentity(ok=False, reason="test: validation missing")) as identity, \
                patch.object(service, "_run_training") as training:
            context.return_value.__enter__.return_value = run
            self.assertEqual(train_model.main(), 1)
        identity.assert_called_once_with(self.dataset, require_validated=True)
        training.assert_not_called()
        self.assertEqual(self.baseline.read_bytes(), self.original_baseline)
