import json
import os
from unittest.mock import Mock, patch

import requests
from django.test import SimpleTestCase

from fishweb import assistant
from fishweb.llm_config import resolve_llm_config


class ModelConfigurationTests(SimpleTestCase):
    def test_deepseek_key_alone_resolves_official_defaults(self):
        config = resolve_llm_config({"DEEPSEEK_API_KEY": "unit-test-key"})
        self.assertTrue(config.configured)
        self.assertEqual(config.endpoint, "https://api.deepseek.com/chat/completions")
        self.assertEqual(config.model, "deepseek-flash")
        self.assertEqual(config.provider, "deepseek")

    def test_deepseek_endpoint_and_model_can_be_overridden(self):
        config = resolve_llm_config({"DEEPSEEK_API_KEY": "unit-test-key",
                                    "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1/",
                                    "DEEPSEEK_MODEL": "chosen-model"})
        self.assertEqual(config.endpoint, "https://api.deepseek.com/v1/chat/completions")
        self.assertEqual(config.model, "chosen-model")

    def test_existing_compatible_settings_remain_independent_and_take_priority(self):
        config = resolve_llm_config({"FISH_LLM_API_KEY": "compatible-key",
                                    "FISH_LLM_BASE_URL": "https://provider.invalid/v1/chat/completions/",
                                    "FISH_LLM_MODEL": "existing-model", "DEEPSEEK_API_KEY": "other-key"})
        self.assertEqual(config.api_key, "compatible-key")
        self.assertEqual(config.model, "existing-model")
        self.assertEqual(config.endpoint, "https://provider.invalid/v1/chat/completions")
        self.assertFalse(config.is_deepseek)

    def test_partial_compatible_config_never_reuses_deepseek_key(self):
        config = resolve_llm_config({"FISH_LLM_BASE_URL": "https://provider.invalid/v1",
                                    "FISH_LLM_MODEL": "existing-model", "DEEPSEEK_API_KEY": "other-key"})
        self.assertFalse(config.configured)
        self.assertEqual(config.api_key, "")

    def test_blank_template_settings_allow_deepseek_environment_key(self):
        config = resolve_llm_config({"FISH_LLM_BASE_URL": " ", "FISH_LLM_API_KEY": "",
                                    "FISH_LLM_MODEL": "", "DEEPSEEK_API_KEY": "unit-test-key"})
        self.assertTrue(config.configured)
        self.assertEqual(config.provider, "deepseek")

    def test_missing_key_is_offline_and_public_status_never_contains_credential(self):
        self.assertFalse(resolve_llm_config({}).configured)
        config = resolve_llm_config({"DEEPSEEK_API_KEY": "unit-test-private-value"})
        public = json.dumps(config.public_status())
        self.assertNotIn("unit-test-private-value", public)
        self.assertNotIn("unit-test-private-value", repr(config))
        self.assertNotIn("api_key", public)


class PlainLanguageAssistantTests(SimpleTestCase):
    def facts(self):
        return {"current_detection": None, "local_dataset": {"available": False}, "training_run": None}

    def test_deepseek_request_uses_plain_language_and_quick_text_mode(self):
        remote = Mock()
        remote.json.return_value = {"choices": [{"message": {"content": "  模型在画面里找到了鱼，请回看画框位置。  "}}]}
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "unit-test-key"}, clear=True), \
                patch("fishweb.assistant.evidence_for", return_value=self.facts()), \
                patch("fishweb.assistant.requests.post", return_value=remote) as post:
            result = assistant.answer("用普通人能听懂的话解释。")
        self.assertEqual(result["mode"], "llm")
        self.assertEqual(result["provider"], "deepseek")
        self.assertEqual(result["model"], "deepseek-flash")
        self.assertEqual(result["answer"], "模型在画面里找到了鱼，请回看画框位置。")
        self.assertEqual(post.call_args.args[0], "https://api.deepseek.com/chat/completions")
        sent = post.call_args.kwargs["json"]
        self.assertEqual(sent["thinking"], {"type": "disabled"})
        self.assertEqual(sent["messages"][0]["role"], "system")
        self.assertIn("通俗易懂的日常中文", sent["messages"][0]["content"])
        self.assertIn("不能等同", assistant.local_answer("这些分数能说明识别有多准吗？", self.facts()))
        self.assertNotIn("unit-test-key", json.dumps(sent))

    def test_compatible_request_does_not_receive_deepseek_specific_options(self):
        remote = Mock()
        remote.json.return_value = {"choices": [{"message": {"content": "普通中文说明。"}}]}
        with patch.dict(os.environ, {"FISH_LLM_BASE_URL": "https://provider.invalid/v1",
                                    "FISH_LLM_API_KEY": "unit-test-key", "FISH_LLM_MODEL": "existing-model"}, clear=True), \
                patch("fishweb.assistant.evidence_for", return_value=self.facts()), \
                patch("fishweb.assistant.requests.post", return_value=remote) as post:
            assistant.answer("解释一下")
        self.assertNotIn("thinking", post.call_args.kwargs["json"])

    def test_history_is_bounded_and_client_system_roles_are_ignored(self):
        remote = Mock()
        remote.json.return_value = {"choices": [{"message": {"content": "简单说明。"}}]}
        history = [{"role": "user", "content": "discard-old-" + str(i)} for i in range(3)]
        history += [{"role": "user", "content": "x" * 3000}, {"role": "system", "content": "untrusted-rule"}]
        history += [{"role": "assistant", "content": "previous"}] * 6
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "unit-test-key"}, clear=True), \
                patch("fishweb.assistant.evidence_for", return_value=self.facts()), \
                patch("fishweb.assistant.requests.post", return_value=remote) as post:
            assistant.answer("继续解释", history=history)
        sent = post.call_args.kwargs["json"]["messages"]
        self.assertEqual(sum(item["role"] == "system" for item in sent), 1)
        self.assertEqual(len(sent[1]["content"]), 2000)
        serialized = json.dumps(sent)
        self.assertNotIn("discard-old", serialized)
        self.assertNotIn("untrusted-rule", serialized)

    def test_timeout_or_empty_answer_falls_back_without_exposing_provider_error(self):
        empty = Mock()
        empty.json.return_value = {"choices": [{"message": {"content": " "}}]}
        for effect in (requests.Timeout("private-server-detail"), empty):
            with self.subTest(effect=type(effect).__name__), \
                    patch.dict(os.environ, {"DEEPSEEK_API_KEY": "unit-test-key"}, clear=True), \
                    patch("fishweb.assistant.evidence_for", return_value=self.facts()), \
                    patch("fishweb.assistant.requests.post", side_effect=effect if isinstance(effect, Exception) else None,
                          return_value=effect):
                result = assistant.answer("这些分数说明什么？")
            self.assertEqual(result["mode"], "local")
            self.assertIn("DeepSeek", result["notice"])
            self.assertIn("未调用大模型", result["answer"])
            self.assertNotIn("private-server-detail", json.dumps(result))

    def test_local_video_explanation_uses_everyday_words_and_preserves_counting_limit(self):
        facts = {"current_detection": {"media_type": "video", "frames_analyzed": 3,
                                      "total_detections": 3, "species_count": 1,
                                      "class_counts": [{"label": "Mola", "count": 3, "mean_confidence": .9}]}}
        result = assistant.local_answer("检测到了几次，就代表有几条鱼吗？", facts)
        self.assertIn("累计画出鱼框3次", result)
        self.assertIn("不能当成不同鱼的条数", result)
        self.assertNotIn("置信度", result)
        self.assertNotIn("框次", result)

    def test_incomplete_detection_task_never_calls_provider(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "unit-test-key"}, clear=True), \
                patch("fishweb.assistant.get_job", return_value={"type": "inference", "status": "running"}), \
                patch("fishweb.assistant.requests.post") as post:
            with self.assertRaisesRegex(ValueError, "尚无已完成"):
                assistant.answer("解释视频", "a" * 32)
        post.assert_not_called()
