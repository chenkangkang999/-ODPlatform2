import json
import os
from unittest.mock import patch

from django.test import RequestFactory, SimpleTestCase, override_settings

from fishweb.views import mutation_origin_allowed


class DeploymentOriginTests(SimpleTestCase):
    studio = "https://cx2474556379-fish.ms.show"

    def request(self, **headers):
        return RequestFactory().post("/api/inference/", HTTP_HOST="testserver", **headers)

    def test_local_same_origin_is_allowed_and_foreign_origin_is_rejected(self):
        with patch.dict(os.environ, {"FISH_TRUSTED_ORIGINS": ""}):
            self.assertTrue(mutation_origin_allowed(self.request(HTTP_ORIGIN="http://testserver")))
            self.assertFalse(mutation_origin_allowed(self.request(HTTP_ORIGIN="https://foreign.invalid")))
            self.assertFalse(mutation_origin_allowed(self.request(HTTP_ORIGIN="https://testserver")))

    def test_exact_studio_origin_is_accepted_behind_internal_host(self):
        with patch.dict(os.environ, {"FISH_TRUSTED_ORIGINS": self.studio}):
            self.assertTrue(mutation_origin_allowed(self.request(HTTP_ORIGIN=self.studio,
                                                                 HTTP_SEC_FETCH_SITE="same-origin")))
            self.assertTrue(mutation_origin_allowed(self.request(HTTP_ORIGIN=self.studio,
                                                                 HTTP_SEC_FETCH_SITE="cross-site")))

    def test_other_studios_suffix_attacks_and_null_origins_stay_rejected(self):
        with patch.dict(os.environ, {"FISH_TRUSTED_ORIGINS": self.studio}):
            for origin in ("https://other-fish.ms.show", self.studio + ".foreign.invalid", "null",
                           "http://cx2474556379-fish.ms.show", "https://user:password@cx2474556379-fish.ms.show"):
                with self.subTest(origin=origin):
                    self.assertFalse(mutation_origin_allowed(self.request(HTTP_ORIGIN=origin)))

    def test_referer_fallback_and_default_https_port(self):
        with patch.dict(os.environ, {"FISH_TRUSTED_ORIGINS": self.studio}):
            self.assertTrue(mutation_origin_allowed(self.request(HTTP_REFERER=self.studio + "/#training")))
            self.assertTrue(mutation_origin_allowed(self.request(HTTP_ORIGIN=self.studio + ":443")))
            self.assertFalse(mutation_origin_allowed(self.request(HTTP_ORIGIN=self.studio + ":8443")))

    @override_settings(USE_X_FORWARDED_HOST=True, SECURE_PROXY_SSL_HEADER=("HTTP_X_FORWARDED_PROTO", "https"),
                       ALLOWED_HOSTS=["cx2474556379-fish.ms.show", "testserver"])
    def test_configured_proxy_headers_preserve_the_browser_origin(self):
        with patch.dict(os.environ, {"FISH_TRUSTED_ORIGINS": ""}):
            request = self.request(HTTP_ORIGIN=self.studio, HTTP_X_FORWARDED_HOST="cx2474556379-fish.ms.show",
                                   HTTP_X_FORWARDED_PROTO="https")
            self.assertTrue(mutation_origin_allowed(request))

    def test_cross_site_without_an_explicit_trusted_origin_is_rejected(self):
        with patch.dict(os.environ, {"FISH_TRUSTED_ORIGINS": ""}):
            self.assertFalse(mutation_origin_allowed(self.request(HTTP_SEC_FETCH_SITE="cross-site")))
            self.assertFalse(mutation_origin_allowed(self.request(HTTP_ORIGIN="http://testserver",
                                                                  HTTP_SEC_FETCH_SITE="cross-site")))

    def test_training_request_reaches_validation_through_studio_proxy(self):
        with patch.dict(os.environ, {"FISH_PUBLIC": "1", "FISH_TRUSTED_ORIGINS": self.studio}), \
                patch("fishweb.jobs.create_training", side_effect=ValueError("当前环境缺少训练数据")):
            response = self.client.post("/api/jobs/", data=json.dumps({"type": "train"}),
                                        content_type="application/json", REMOTE_ADDR="10.0.0.2",
                                        HTTP_ORIGIN=self.studio, HTTP_SEC_FETCH_SITE="same-origin")
        self.assertEqual(response.status_code, 400)
        self.assertIn("训练数据", response.json()["error"])

    def test_upload_request_reaches_file_validation_instead_of_origin_error(self):
        with patch.dict(os.environ, {"FISH_PUBLIC": "1", "FISH_TRUSTED_ORIGINS": self.studio}):
            response = self.client.post("/api/inference/", REMOTE_ADDR="10.0.0.2", HTTP_ORIGIN=self.studio,
                                        HTTP_SEC_FETCH_SITE="same-origin")
        self.assertEqual(response.status_code, 400)
        self.assertNotEqual(response.json()["code"], "untrusted_origin")
