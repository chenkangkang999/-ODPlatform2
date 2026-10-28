"""Deployment preparation and credential boundaries, with no remote requests."""
import base64
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "sync_modelscope.py"
SPEC = importlib.util.spec_from_file_location("sync_modelscope", SCRIPT)
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def commit(self, repo):
        sync.git(repo, "add", "--all")
        sync.git(repo, "-c", "user.name=Unit Test", "-c", "user.email=unit-test@example.invalid",
                 "commit", "-m", "test fixture")

    def test_env_file_is_literal_and_never_expands_shell_syntax(self):
        file = self.root / ".env.modelscope"
        file.write_text("\ufeff# local\nMODELSCOPE_API_KEY='fake value=one' # note\n"
                        'OTHER="$(do-not-run) ${literal}"\n', encoding="utf-8")
        values = sync.read_env_file(file)
        self.assertEqual(values["MODELSCOPE_API_KEY"], "fake value=one")
        self.assertEqual(values["OTHER"], "$(do-not-run) ${literal}")

    def test_git_auth_exists_only_in_temporary_child_environment(self):
        with patch.dict(os.environ, {"GIT_TRACE": "1", "GIT_CONFIG_COUNT": "7", "GIT_ASKPASS": "unexpected"}):
            env = sync.git_environment("fake-token-for-test")
        self.assertNotIn("GIT_TRACE", env)
        self.assertNotIn("GIT_ASKPASS", env)
        self.assertEqual(env["GIT_CONFIG_COUNT"], "3")
        self.assertEqual(env["GIT_CONFIG_VALUE_0"], "")
        auth = base64.b64encode(b"oauth2:fake-token-for-test").decode()
        self.assertEqual(env["GIT_CONFIG_VALUE_2"], "Authorization: Basic " + auth)
        self.assertNotIn("fake-token-for-test", env["GIT_CONFIG_VALUE_2"])

    def test_readme_preserves_studio_frontmatter_and_custom_notes_and_is_idempotent(self):
        old = "---\nsdk: docker\napp_port: 7860\ncustom: keep-me\n---\n\n# 团队说明\n保留这个备注。左侧导航进入。\n"
        merged = sync.merge_readme(old, "a" * 40)
        self.assertIn("custom: keep-me", merged)
        self.assertIn("保留这个备注", merged)
        self.assertIn("顶部导航进入", merged)
        self.assertEqual(sync.merge_readme(merged, "a" * 40), merged)
        self.assertEqual(merged.count(sync.START), 1)

    def test_path_traversal_and_git_metadata_are_rejected(self):
        for path in ("../private", ".git/config", "/absolute", "folder\\private", "drive:private"):
            with self.subTest(path=path), self.assertRaises(sync.SyncError):
                sync.safe_child(self.root, path)

    def test_overlay_preserves_custom_files_and_second_identical_run_is_noop(self):
        source, studio = self.root / "source", self.root / "studio"
        source.mkdir()
        studio.mkdir()
        sync.git(studio, "init")
        (studio / "README.md").write_text("---\nsdk: docker\n---\n\n自定义说明。\n", encoding="utf-8")
        (studio / "custom.txt").write_text("team-owned", encoding="utf-8")
        self.commit(studio)
        (source / "README.md").write_text("source README", encoding="utf-8")
        (source / "app.txt").write_text("new-code", encoding="utf-8")
        changed = sync.overlay_source(source, studio, "a" * 40, ["README.md", "app.txt"])
        self.assertIn("app.txt", changed)
        self.assertEqual((studio / "custom.txt").read_text(), "team-owned")
        self.assertIn("自定义说明", (studio / "README.md").read_text(encoding="utf-8"))
        self.commit(studio)
        self.assertEqual(sync.overlay_source(source, studio, "a" * 40, ["README.md", "app.txt"]), [])

    def test_archive_refuses_tracked_private_env_file(self):
        source, out = self.root / "source", self.root / "out"
        source.mkdir()
        out.mkdir()
        sync.git(source, "init")
        (source / ".env").write_text("VALUE=fake", encoding="utf-8")
        self.commit(source)
        with self.assertRaisesRegex(sync.SyncError, "私有环境配置"):
            sync.export_head(source, out)

    def test_deepseek_only_updates_its_known_secret(self):
        api = Mock()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fake-deepseek-key"}, clear=True):
            changed = sync.sync_deepseek_secret(api, "/studios/example/fish", self.root,
                                                {"UNRELATED_SECRET"}, {"UNRELATED_VARIABLE"})
        self.assertTrue(changed)
        api.call.assert_called_once_with("POST", "/studios/example/fish/secrets",
                                         {"key": "DEEPSEEK_API_KEY", "value": "fake-deepseek-key"})

    def test_existing_compatible_service_is_preserved_without_mutation(self):
        api = Mock()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fake-deepseek-key"}, clear=True):
            with self.assertRaisesRegex(sync.SyncError, "保留现有设置"):
                sync.sync_deepseek_secret(api, "/studios/example/fish", self.root,
                                          {"FISH_LLM_API_KEY"}, set())
        api.call.assert_not_called()

    def test_api_error_contains_no_raw_response(self):
        api = sync.ModelScopeAPI("fake-modelscope-token")
        response = Mock(status_code=403, text="fake-private-response")
        api.session.request = Mock(return_value=response)
        with self.assertRaises(sync.SyncError) as error:
            api.call("GET", "/users/me")
        self.assertIn("403", str(error.exception))
        self.assertNotIn("fake-private-response", str(error.exception))
        self.assertNotIn("fake-modelscope-token", str(error.exception))

    def test_variable_list_returns_names_without_values(self):
        result = sync.variable_keys({"secrets": [{"key": "KNOWN", "value": "not-for-output"}]}, "secrets")
        self.assertEqual(result, {"KNOWN"})
        self.assertNotIn("not-for-output", json.dumps(sorted(result)))

    def test_release_only_updates_the_known_plain_variable(self):
        for existing, method in ((set(), "POST"), ({"FISH_RELEASE", "CUSTOM_SETTING"}, "PUT")):
            with self.subTest(method=method):
                api = Mock()
                sync.sync_release(api, "/studios/example/fish", "a" * 40, existing)
                api.call.assert_called_once_with(method, "/studios/example/fish/variables",
                                                 {"key": "FISH_RELEASE", "value": "a" * 40})


if __name__ == "__main__":
    unittest.main()
