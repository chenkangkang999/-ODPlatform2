import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from fishweb import env


class LocalEnvironmentTests(TestCase):
    def load_text(self, text, existing=None):
        destination = {} if existing is None else dict(existing)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(text, encoding="utf-8")
            loaded = env.load_local_env(path, destination)
        return destination, loaded

    def test_quotes_equals_comments_and_bom_are_literal(self):
        destination, loaded = self.load_text(
            "\ufeff# local configuration\n\n"
            " FISH_LLM_MODEL = 'model=one' # inline comment\n"
            'FISH_LLM_API_KEY="local value=two#literal"\n'
            "FISH_LLM_BASE_URL=https://provider.invalid/v1 # note\n"
        )
        self.assertEqual(destination["FISH_LLM_MODEL"], "model=one")
        self.assertEqual(destination["FISH_LLM_API_KEY"], "local value=two#literal")
        self.assertEqual(destination["FISH_LLM_BASE_URL"], "https://provider.invalid/v1")
        self.assertEqual(set(loaded), set(destination))

    def test_existing_environment_including_empty_value_wins(self):
        destination, loaded = self.load_text(
            "FISH_LLM_MODEL=file-model\nFISH_LLM_API_KEY=file-value",
            {"FISH_LLM_MODEL": "shell-model", "FISH_LLM_API_KEY": ""},
        )
        self.assertEqual(destination, {"FISH_LLM_MODEL": "shell-model", "FISH_LLM_API_KEY": ""})
        self.assertEqual(loaded, ())

    def test_unrelated_variables_and_malformed_lines_are_ignored(self):
        destination, loaded = self.load_text(
            "PATH=unexpected\nPYTHONPATH=unexpected\nDJANGO_SETTINGS_MODULE=unexpected\n"
            "FISH_LLM_MODEL\nFISH_LLM_API_KEY='unfinished\nUNKNOWN=value\n"
        )
        self.assertEqual(destination, {})
        self.assertEqual(loaded, ())

    def test_shell_and_variable_syntax_is_never_expanded(self):
        destination, _ = self.load_text(
            'FISH_LLM_MODEL="$(literal-command) ${literal-variable} `literal-data`"\n'
        )
        self.assertEqual(
            destination["FISH_LLM_MODEL"],
            "$(literal-command) ${literal-variable} `literal-data`",
        )

    def test_missing_file_is_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = {}
            self.assertEqual(env.load_local_env(Path(directory) / "missing", destination), ())
            self.assertEqual(destination, {})

    def test_default_does_not_search_parent_env_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = root / "backend"
            backend.mkdir()
            (root / ".env").write_text("FISH_LLM_MODEL=parent-model", encoding="utf-8")
            with patch.object(env, "DEFAULT_ENV_FILE", backend / ".env"):
                destination = {}
                self.assertEqual(env.load_local_env(environ=destination), ())
                self.assertEqual(destination, {})

    def test_engine_and_django_keys_are_supported(self):
        destination, _ = self.load_text(
            "FISH_ENGINE_PYTHON=engine-python\nFISH_DJANGO_SECRET_KEY=local-example-value\n"
        )
        self.assertEqual(destination, {
            "FISH_ENGINE_PYTHON": "engine-python",
            "FISH_DJANGO_SECRET_KEY": "local-example-value",
        })
