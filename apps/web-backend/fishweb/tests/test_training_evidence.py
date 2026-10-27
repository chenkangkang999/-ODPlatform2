import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from django.test import SimpleTestCase

from fishweb import assistant, catalog


class TrainingEvidenceTests(SimpleTestCase):
    def setUp(self):
        self.summary = json.loads((catalog.ROOT / "docs/fish_training_run.json").read_text(encoding="utf-8"))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "docs").mkdir()
        (self.root / "docs/fish_training_run.json").write_text(json.dumps(self.summary), encoding="utf-8")

    def test_new_clone_has_real_curves_and_distinct_final_validation_metrics(self):
        # No ignored runs, weights or original archive are needed to display this evidence.
        with patch.object(catalog, "ROOT", self.root):
            response = self.client.get("/api/experiments/")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        run = body["training_summary"]
        self.assertEqual(run["provenance"], "provided_training_archive")
        self.assertFalse(run["measured_locally"])
        self.assertEqual([row["epoch"] for row in run["curves"]], list(range(1, 11)))
        self.assertEqual(run["curves"][0]["map50"], .32572)
        self.assertEqual(run["curves"][-1]["box_loss"], .65472)
        self.assertEqual(run["training_seconds"], 410.985)
        self.assertEqual(run["metrics"]["map50"], .9095257756720391)
        self.assertEqual(run["last_epoch_metrics"]["map50"], .90969)
        self.assertEqual(run["split"], {"train": 3652, "val": 458, "test": 450})
        self.assertEqual(len(run["per_class"]), 33)
        self.assertEqual(run["weight_sha256"], "172f8f54747f7fe8bbf810a29ae514a58c4b4db2067fb8ed4b96e9307fcb76ab")
        self.assertEqual(body["documented_metrics"]["map50"], .9073)
        self.assertEqual(len(body["experiments"]), 1)
        self.assertEqual(body["source"], "provided_training_archive")
        serialized = json.dumps(run)
        for private in ("D:/", "D:\\", "xwechat_files", "run_dir", "save_dir"):
            self.assertNotIn(private, serialized)

    def test_imported_local_files_keep_external_origin_and_native_runs_remain_local(self):
        for run_id in (self.summary["id"], "new-local-run"):
            run = self.root / "runs/training" / run_id
            run.mkdir(parents=True)
            (run / "results.csv").write_text("epoch,metrics/mAP50(B)\n10,0.90969\n", encoding="utf-8")
            (run / "odp_audit.json").write_text(json.dumps({"data_yaml": "configs/datasets/fish.yaml"}), encoding="utf-8")
        with patch.object(catalog, "ROOT", self.root):
            records = catalog.experiments()["experiments"]
        self.assertEqual(len(records), 2)  # importing the archive never duplicates its snapshot
        imported = next(row for row in records if row["id"] == self.summary["id"])
        native = next(row for row in records if row["id"] == "new-local-run")
        self.assertFalse(imported["measured_locally"])
        self.assertEqual(imported["provenance"], "provided_training_archive")
        self.assertEqual(imported["metrics"]["map50"], self.summary["metrics"]["map50"])
        self.assertTrue(native["measured_locally"])
        self.assertEqual(native["provenance"], "local_training")

    def local_data(self):
        return {"local": {"available": True, "counts": self.summary["split"], "instances": 6034,
                          "dataset_yaml": "C:/private-local-data/fish.yaml"},
                "validation": {"measured_locally": True, "overall": "PASS", "checks_passed": 15,
                               "checks_total": 15, "source": "C:/private-report/report.json"}}

    def test_offline_training_answer_acknowledges_integrated_data_and_supplied_run(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch("fishweb.assistant.dataset_info", return_value=self.local_data()), \
                patch("fishweb.assistant.training_summary", return_value=self.summary):
            result = assistant.answer("现在的数据和训练情况是什么？")
        self.assertEqual(result["mode"], "local")
        for text in ("本机已接入", "15/15", "10轮", "410.985秒", "90.95%", "不是本机新训练", "未调用大模型"):
            self.assertIn(text, result["answer"])
        self.assertNotIn("先接入本地数据", result["answer"])

    def test_model_payload_uses_sanitized_training_summary_and_distinguishes_baseline(self):
        remote = Mock()
        remote.json.return_value = {"choices": [{"message": {"content": "依据提供训练记录回答。"}}]}
        with patch.dict(os.environ, {"FISH_LLM_BASE_URL": "https://provider.example/v1", "FISH_LLM_API_KEY": "secret",
                                    "FISH_LLM_MODEL": "configured-model"}, clear=True), \
                patch("fishweb.assistant.dataset_info", return_value=self.local_data()), \
                patch("fishweb.assistant.training_summary", return_value=self.summary), \
                patch("fishweb.assistant.requests.post", return_value=remote) as post:
            result = assistant.answer("训练指标是多少？")
        self.assertEqual(result["mode"], "llm")
        messages = post.call_args.kwargs["json"]["messages"]
        sent = json.dumps(messages, ensure_ascii=False)
        self.assertIn("provided_training_archive", sent)
        self.assertIn(str(self.summary["metrics"]["map50"]), sent)
        self.assertIn("0.9073", sent)
        for private in ("private-local-data", "private-report", "source_archive", ".zip", "weight_sha256", ".pt"):
            self.assertNotIn(private, sent)

    def test_offline_metrics_keep_training_validation_and_independent_baseline_separate(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch("fishweb.assistant.dataset_info", return_value=self.local_data()), \
                patch("fishweb.assistant.training_summary", return_value=self.summary):
            result = assistant.answer("mAP是多少？")
        self.assertIn("90.95%", result["answer"])
        self.assertIn("90.73%", result["answer"])
        self.assertIn("不是独立测试集成绩", result["answer"])
