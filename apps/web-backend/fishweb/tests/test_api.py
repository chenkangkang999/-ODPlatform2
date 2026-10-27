import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import RequestFactory, SimpleTestCase

from fishweb import assistant, catalog, jobs, views
from fishweb.worker import aggregate_frames, validate_fish_labels


class ApiTests(SimpleTestCase):
    def test_evaluation_reads_original_dataset_identity_from_materialized_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs/detect_eval/20261027-150000"
            run.mkdir(parents=True)
            (run / "report.json").write_text(json.dumps({
                "data_yaml": "runs/detect_eval/20261027-150000/_dataset_view/data.yaml",
                "dataset": {"source_yaml": "apps/platform/configs/datasets/fish.yaml"},
                "overall": {"map50": 0.5},
            }), encoding="utf-8")
            with patch.object(catalog, "ROOT", root):
                records = catalog.experiments()["experiments"]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["metrics"]["map50"], 0.5)
            self.assertTrue(records[0]["measured_locally"])

    def test_documented_dataset_never_claims_local_measurements(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(catalog, "ROOT", Path(directory)):
            response = self.client.get("/api/dataset/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["classes"]), 33)
        self.assertEqual(data["counts"]["total"], 4560)
        self.assertEqual(data["provenance"], "repository_documentation")
        self.assertFalse(data["validation"]["measured_locally"])

    def test_external_machine_and_cross_origin_mutations_rejected(self):
        self.assertEqual(self.client.get("/api/status/", REMOTE_ADDR="192.168.1.23").status_code, 403)
        response = self.client.post("/api/chat/", data=json.dumps({"message": "你好"}), content_type="application/json",
                                    HTTP_ORIGIN="https://evil.example")
        self.assertEqual(response.status_code, 403)

    def test_missing_upload_and_missing_weights(self):
        self.assertEqual(self.client.post("/api/inference/").status_code, 400)
        response = self.client.post("/api/inference/", {"file": SimpleUploadedFile("fish.mp4", b"test")})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "missing_fish_weights")

    def test_external_model_path_rejected_before_job_creation(self):
        response = self.client.post("/api/inference/", {"model": "../../secret.pt",
                                    "file": SimpleUploadedFile("test.jpg", b"image")})
        self.assertEqual(response.status_code, 400)
        self.assertIn("models", response.json()["error"])

    def test_upload_weights_rejected(self):
        with patch("fishweb.jobs.resolve_model", return_value=Path("models/fish.pt")):
            response = self.client.post("/api/inference/", {"model": "fish.pt", "file": SimpleUploadedFile("unsafe.pt", b"pickle")})
        self.assertEqual(response.status_code, 400)

    def test_invalid_numeric_parameters_rejected(self):
        with patch("fishweb.jobs.resolve_model", return_value=Path("models/fish.pt")):
            for key, value in (("conf", "NaN"), ("imgsz", "641"), ("stride", "0"), ("device", "../../evil")):
                with self.subTest(key=key):
                    with self.assertRaises(ValueError):
                        jobs.inference_params({"model": "fish.pt", key: value})

    def test_training_requires_local_dataset(self):
        with patch("fishweb.jobs.dataset_info", return_value={"local": {"available": False}}):
            response = self.client.post("/api/jobs/", data=json.dumps({"type": "train", "model": "fish.pt"}), content_type="application/json")
        self.assertEqual(response.status_code, 400)
        self.assertIn("数据", response.json()["error"])

    def test_invalid_job_and_private_files_not_exposed(self):
        self.assertEqual(self.client.get("/api/jobs/" + "a" * 32 + "/").status_code, 404)
        self.assertEqual(self.client.get("/api/jobs/" + "a" * 32 + "/files/worker.json").status_code, 404)

    def test_chat_missing_config_explicit_offline_mode(self):
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.post("/api/chat/", data=json.dumps({"message": "帮我解释视频"}), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "local")
        self.assertFalse(body["evidence_used"])
        self.assertIn("不能判断", body["answer"])

    def test_chat_invalid_json_and_unknown_job(self):
        self.assertEqual(self.client.post("/api/chat/", data="broken", content_type="application/json").status_code, 400)
        self.assertEqual(self.client.post("/api/chat/", data=json.dumps({"message": "解释", "job_id": "nope"}), content_type="application/json").status_code, 400)

    def test_model_payload_contains_stats_and_no_raw_media(self):
        mocked = Mock()
        mocked.json.return_value = {"choices": [{"message": {"content": "依据检测统计，发现Mola。"}}]}
        job = {"type": "inference", "status": "completed", "result": {"model": "fish.pt", "media_type": "video",
               "total_detections": 3, "species_count": 1, "frames_analyzed": 2, "annotated_url": "/private-video.mp4",
               "source": "input.mp4", "class_counts": [{"label": "Mola", "count": 3, "mean_confidence": .9}]}}
        with patch.dict(os.environ, {"FISH_LLM_BASE_URL": "https://provider.example/v1", "FISH_LLM_API_KEY": "test-secret",
                                    "FISH_LLM_MODEL": "configured-model"}, clear=True), \
             patch("fishweb.assistant.get_job", return_value=job), \
             patch("fishweb.assistant.requests.post", return_value=mocked) as post:
            result = assistant.answer("解释视频", "a" * 32)
        self.assertEqual(result["mode"], "llm")
        sent = json.dumps(post.call_args.kwargs["json"], ensure_ascii=False)
        self.assertIn("Mola", sent)
        self.assertNotIn("input.mp4", sent)
        self.assertNotIn("private-video", sent)
        self.assertNotIn("test-secret", sent)

    def test_api_failure_falls_back_with_honest_label(self):
        import requests
        with patch.dict(os.environ, {"FISH_LLM_BASE_URL": "https://provider.example/v1", "FISH_LLM_API_KEY": "test-secret",
                                    "FISH_LLM_MODEL": "configured-model"}, clear=True), \
             patch("fishweb.assistant.requests.post", side_effect=requests.Timeout("no connection")):
            result = assistant.answer("mAP是什么")
        self.assertEqual(result["mode"], "local")
        self.assertIn("切换", result["notice"])

    def test_video_box_counts_not_unique_fish(self):
        frames = [
            {"detections": [{"label": "Mola", "confidence": .8}, {"label": "Mola", "confidence": 1.0}]},
            {"detections": [{"label": "Mola", "confidence": .9}, {"label": "Rui", "confidence": .6}]},
        ]
        data = aggregate_frames(frames)
        self.assertEqual(data["total_detections"], 4)
        self.assertEqual(data["species_count"], 2)
        self.assertEqual(data["class_counts"][0]["count"], 3)
        self.assertEqual(data["class_counts"][0]["peak_per_frame"], 2)
        self.assertEqual(data["class_counts"][0]["mean_confidence"], .9)
        self.assertIn("重复", data["counting_note"])

    def test_coco_weights_cannot_claim_fish_detection(self):
        with self.assertRaisesRegex(ValueError, "COCO"):
            validate_fish_labels({0: "person", 1: "bicycle"}, catalog.class_names())
        validate_fish_labels(dict(enumerate(catalog.class_names())), catalog.class_names())

    def test_video_range_playback_and_invalid_ranges(self):
        factory = RequestFactory()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "video.mp4"
            path.write_bytes(b"0123456789")
            response = views.ranged_file(factory.get("/video.mp4", HTTP_RANGE="bytes=2-5"), path)
            self.assertEqual(response.status_code, 206)
            self.assertEqual(b"".join(response.streaming_content), b"2345")
            self.assertEqual(response["Content-Range"], "bytes 2-5/10")
            response = views.ranged_file(factory.get("/video.mp4", HTTP_RANGE="bytes=-3"), path)
            self.assertEqual(b"".join(response.streaming_content), b"789")
            self.assertEqual(views.ranged_file(factory.get("/video.mp4", HTTP_RANGE="bytes=20-30"), path).status_code, 416)
