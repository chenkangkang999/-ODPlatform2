"""Integration checks use an explicit fake predictor, never measured fish model metrics."""
import importlib.util
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import skipUnless
from unittest.mock import Mock, patch

from django.test import SimpleTestCase

from fishweb import catalog, jobs, worker

ENGINE_TOOLS = all(importlib.util.find_spec(name) for name in ("cv2", "numpy", "imageio_ffmpeg"))


@skipUnless(ENGINE_TOOLS, "optional OpenCV/ffmpeg dependencies are not installed")
class MediaPipelineTests(SimpleTestCase):
    def fake_yolo(self):
        import numpy as np

        def predict(frame, **kwargs):
            box = SimpleNamespace(cls=np.array(11), conf=np.array(.88), xyxy=np.array([[2, 2, 18, 18]]))
            return [SimpleNamespace(boxes=[box], names=dict(enumerate(catalog.class_names())), plot=lambda: frame.copy())]

        return SimpleNamespace(names=dict(enumerate(catalog.class_names())), predict=predict)

    def test_video_transcode_and_structured_evidence_with_fake_predictor(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            source = output / "fixture.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 6, (64, 48))
            self.assertTrue(writer.isOpened())
            for index in range(6):
                writer.write(np.full((48, 64, 3), 30 + index * 20, dtype=np.uint8))
            writer.release()
            params = {"model": "fixture-fish.pt", "source": str(source), "conf": .4,
                      "iou": .7, "imgsz": 640, "device": "cpu", "stride": 2}
            with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=lambda _: self.fake_yolo())}), \
                 patch("fishweb.catalog.resolve_model", return_value=output / "fixture-fish.pt"):
                result = worker.infer(params, output)
            self.assertEqual(result["frames_analyzed"], 3)
            self.assertEqual(result["total_detections"], 3)
            self.assertEqual(result["class_counts"][0]["label"], "Mola")
            self.assertEqual(result["class_counts"][0]["peak_per_frame"], 1)
            self.assertEqual(result["duration_seconds"], 1.0)
            data = json.loads((output / "detections.json").read_text(encoding="utf-8"))
            self.assertEqual([f["frame"] for f in data["frames"]], [0, 2, 4])
            playback = cv2.VideoCapture(str(output / result["annotated"]))
            self.assertTrue(playback.isOpened())
            self.assertEqual(int(playback.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            self.assertAlmostEqual(playback.get(cv2.CAP_PROP_FPS), 3)
            playback.release()
            self.assertFalse((output / "annotated_raw.avi").exists())
            self.assertIn(b"ftyp", (output / "annotated.mp4").read_bytes()[:32])

    def test_image_pipeline_with_fake_predictor(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            source = output / "fixture.png"
            cv2.imencode(".png", np.full((64, 64, 3), 120, dtype=np.uint8))[1].tofile(source)
            params = {"model": "fixture-fish.pt", "source": str(source), "conf": .4,
                      "iou": .7, "imgsz": 640, "device": "cpu", "stride": 1}
            with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=lambda _: self.fake_yolo())}), \
                 patch("fishweb.catalog.resolve_model", return_value=output / "fixture-fish.pt"):
                result = worker.infer(params, output)
            self.assertEqual(result["total_detections"], 1)
            self.assertEqual(result["media_type"], "image")
            self.assertIn("单帧", result["counting_note"])
            self.assertTrue((output / "annotated.jpg").is_file())


class SupervisorTests(SimpleTestCase):
    def test_completed_process_attaches_public_artifact_urls(self):
        with tempfile.TemporaryDirectory() as temp, patch("fishweb.jobs.STATE", Path(temp)), \
             patch.dict(jobs.JOBS, {}, clear=True), patch.dict(jobs.PROCESSES, {}, clear=True):
            job = jobs.new_job("inference", {"model": "fish.pt"})
            directory = jobs.job_directory(job["id"])
            (directory / "result.json").write_text(json.dumps({"annotated": "annotated.mp4", "thumbnail": "thumbnail.jpg"}), encoding="utf-8")
            process = Mock(stdout=io.StringIO("FISH_PROGRESS 50\nfixture predictor executed\n"))
            process.wait.return_value = 0
            with patch("fishweb.jobs.subprocess.Popen", return_value=process) as launch:
                jobs.run_worker(job["id"], "inference", {"model": "fish.pt"})
            completed = jobs.get_job(job["id"])
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["progress"], 100)
            self.assertTrue(completed["result"]["annotated_url"].endswith("/files/annotated.mp4"))
            self.assertIsInstance(launch.call_args.args[0], list)
            self.assertNotIn("shell", launch.call_args.kwargs)
            self.assertNotIn(job["id"], jobs.PROCESSES)
