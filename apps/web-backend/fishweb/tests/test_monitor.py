import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import SimpleTestCase

from fishweb import jobs, monitor


class MonitorTests(SimpleTestCase):
    def setUp(self):
        super().setUp()
        patches = [
            patch.object(monitor, "_CACHE", None),
            patch.object(monitor, "_CACHE_TIME", None),
            patch.object(monitor, "_gpu_state", return_value={"available": False, "count": 0,
                                                             "devices": [], "message": "CPU环境"}),
            patch.object(monitor.psutil, "cpu_percent", return_value=24.5),
            patch.object(monitor.psutil, "cpu_count", return_value=8),
            patch.object(monitor.psutil, "virtual_memory", return_value=SimpleNamespace(used=400, total=1000, percent=40)),
            patch.object(monitor.psutil, "disk_usage", return_value=SimpleNamespace(used=300, total=1000, free=700, percent=30)),
            patch.object(monitor.psutil, "Process", return_value=Mock(pid=123, create_time=Mock(return_value=monitor.time.time()-120),
                                                                     memory_info=Mock(return_value=SimpleNamespace(rss=200)))),
            patch.object(monitor.catalog, "dataset_info", return_value={"local": {"available": True, "total": 4560}}),
            patch.object(monitor.catalog, "engine_available", return_value=True),
            patch.object(monitor.catalog, "model_catalog", return_value=[{}, {}]),
            patch.object(monitor.catalog, "llm_configured", return_value=False),
            patch.object(jobs, "JOBS", {}),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def test_api_returns_real_sampler_schema_and_is_read_only(self):
        response = self.client.get("/api/monitor/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["source"], "local_system")
        self.assertEqual(data["refresh_seconds"], 5)
        self.assertIn("+00:00", data["sampled_at"])
        self.assertEqual(data["cpu"], {"percent": 24.5, "logical_count": 8})
        self.assertEqual(data["memory"]["used_bytes"], 400)
        self.assertEqual(data["disk"]["free_bytes"], 700)
        self.assertEqual(data["process"]["pid"], 123)
        self.assertGreaterEqual(data["process"]["uptime_seconds"], 119)
        self.assertFalse(data["gpu"]["available"])
        self.assertTrue(data["runtime"]["dataset_available"])
        self.assertEqual(data["runtime"]["dataset_images"], 4560)
        self.assertEqual(data["runtime"]["model_candidates"], 2)
        self.assertEqual(self.client.post("/api/monitor/").status_code, 405)

    def test_unavailable_resources_are_null_and_never_fabricated_zero(self):
        with patch.object(monitor.psutil, "cpu_percent", return_value=-1), \
                patch.object(monitor.psutil, "virtual_memory", side_effect=OSError()), \
                patch.object(monitor.psutil, "disk_usage", side_effect=OSError()), \
                patch.object(monitor.psutil, "Process", side_effect=monitor.psutil.NoSuchProcess(1)):
            data = monitor.snapshot()
        self.assertIsNone(data["cpu"]["percent"])
        self.assertIsNone(data["memory"]["used_bytes"])
        self.assertIsNone(data["disk"]["percent"])
        self.assertIsNone(data["process"]["rss_bytes"])
        self.assertEqual(len(data["warnings"]), 4)

    def test_sampling_is_cached_and_return_value_cannot_mutate_cache(self):
        first = monitor.snapshot()
        first["cpu"]["percent"] = 99
        with patch.object(monitor, "_sample", side_effect=AssertionError("sampled too often")):
            second = monitor.snapshot()
        self.assertEqual(second["cpu"]["percent"], 24.5)

    def test_task_totals_use_all_jobs_and_recent_logs_are_bounded_and_redacted(self):
        records = {}
        for index in range(9):
            records[str(index)] = {"id": str(index), "type": "train", "status": "completed" if index < 8 else "failed",
                                   "updated_at": f"2026-10-27T10:00:0{index}", "progress": 100,
                                   "params": {"FISH_LLM_API_KEY": "internal configuration"},
                                   "result": {"private": "unneeded details"},
                                   "error": "Authorization: Bearer local-example-value",
                                   "logs": ["old line"]*5 + ["FISH_LLM_API_KEY=local-example-value"]*5}
        with patch.object(jobs, "JOBS", records):
            data = monitor.snapshot()
        self.assertEqual(data["jobs"]["total"], 9)
        self.assertEqual(data["jobs"]["completed"], 8)
        self.assertEqual(data["jobs"]["failed"], 1)
        self.assertEqual(len(data["recent_jobs"]), 5)
        self.assertEqual(data["recent_jobs"][0]["id"], "8")
        self.assertEqual(len(data["recent_jobs"][0]["latest_logs"]), 5)
        encoded = json.dumps(data)
        self.assertNotIn("local-example-value", encoded)
        self.assertNotIn("internal configuration", encoded)
        self.assertNotIn("unneeded details", encoded)
        self.assertIn("redacted", encoded)

    def test_cpu_gpu_probe_and_probe_failure_remain_distinct(self):
        response = SimpleNamespace(returncode=0, stdout=json.dumps({"available": False, "count": 0, "devices": [], "message": "CPU环境"}))
        with patch.object(monitor.subprocess, "run", return_value=response):
            self.assertFalse(monitor._read_gpu_state()["available"])
        with patch.object(monitor.subprocess, "run", side_effect=OSError()):
            self.assertIsNone(monitor._read_gpu_state()["available"])

    def test_expired_cache_samples_again(self):
        first = monitor.snapshot()
        with patch.object(monitor, "_CACHE_TIME", monitor.time.monotonic()-6), \
                patch.object(monitor.psutil, "cpu_percent", return_value=67.1):
            second = monitor.snapshot()
        self.assertEqual(first["cpu"]["percent"], 24.5)
        self.assertEqual(second["cpu"]["percent"], 67.1)
