import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from escna_v3.providers import LoggedClient, parse_contract, request_payload, response_fields, schema_for, validate_envelope

# Synthetic profiles keep the offline tests independent of local endpoint files.
PROFILES = json.loads((ROOT / "tests/fixtures/provider_profiles.json").read_text())["profiles"]


class Response:
    def __init__(self, status, data):
        self.status_code = status
        self.text = json.dumps(data)


class Session:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


def good(model="qwen3.7-flash-2026-07-15", content='{"message":"Please share relevant evidence."}', finish="stop", tokens=32):
    return {"id": "test-response", "model": model,
            "choices": [{"finish_reason": finish, "message": {"content": content}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": tokens, "total_tokens": 50 + tokens,
                      "completion_tokens_details": {"reasoning_tokens": 0}}}


class V3ProviderTests(unittest.TestCase):
    def test_qwen_wire_parameters(self):
        payload = request_payload(PROFILES["qwen"], "s", "u", schema_for("facilitator"))
        self.assertFalse(payload["enable_thinking"])
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 512)
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertNotIn("extra_body", payload)

    def test_deepseek_strict_textual_contract(self):
        payload = request_payload(PROFILES["deepseek"], "s", "u", schema_for("facilitator"))
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertIn("JSON schema", payload["messages"][0]["content"])

    def test_gemini_omits_deprecated_sampling(self):
        config = request_payload(PROFILES["gemini"], "s", "u", schema_for("facilitator"))["generationConfig"]
        self.assertFalse(set(config) & {"temperature", "seed", "topP", "topK"})
        self.assertEqual(config["thinkingConfig"], {"thinkingLevel": "low"})
        self.assertEqual(config["maxOutputTokens"], 512)

    def test_strict_json(self):
        for text in ('```json\n{"message":"ok"}\n```', '{"message":"ok","extra":1}', '{"message":"ok","message":"two"}', '{"message":" "}', '{"message":NaN}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_contract(text, "facilitator")

    def test_private_fact_ownership(self):
        with self.assertRaises(ValueError):
            parse_contract('{"message":"x","recommendation":"A","disclose_fact_ids":["OTHER"]}', "participant", ("A", "B", "C"), ("OWN",))

    def test_ballot(self):
        self.assertEqual(parse_contract('{"recommendation":"B","acceptable_options":["B","C"]}', "ballot", ("A", "B", "C"))["recommendation"], "B")
        with self.assertRaises(ValueError):
            parse_contract('{"recommendation":"A","acceptable_options":["B"]}', "ballot", ("A", "B", "C"))
        with self.assertRaises(ValueError):
            parse_contract('{"recommendation":"ABSTAIN","acceptable_options":["B"]}', "ballot", ("A", "B", "C"))

    def test_envelope_rejects_wrong_model_and_truncation(self):
        for data in (good(model="other"), good(finish="length"), good(tokens=513)):
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_envelope(PROFILES["qwen"], response_fields(PROFILES["qwen"], data))

    def test_thinking_cannot_be_hidden_in_budget(self):
        data = {"modelVersion": "gemini-3.7-flash", "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "secret thought", "thought": True}, {"text": '{"message":"ok"}'}]}}],
                "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 50, "thoughtsTokenCount": 480, "totalTokenCount": 630}}
        fields = response_fields(PROFILES["gemini"], data)
        self.assertNotIn("secret", fields["text"])
        self.assertEqual(fields["completion_including_reasoning"], 530)
        with self.assertRaises(ValueError):
            validate_envelope(PROFILES["gemini"], fields)

    def test_missing_usage_not_zero(self):
        data = good()
        del data["usage"]
        fields = response_fields(PROFILES["qwen"], data)
        self.assertIsNone(fields["input_tokens"])
        self.assertIsNone(fields["reported_monetary_cost"])

    def test_blocked_candidate_still_retains_usage(self):
        fields = response_fields(PROFILES["gemini"], {"candidates": [], "usageMetadata": {"promptTokenCount": 123, "totalTokenCount": 123}})
        self.assertEqual(fields["input_tokens"], 123)
        self.assertEqual(fields["total_tokens"], 123)
        with self.assertRaises(ValueError):
            validate_envelope(PROFILES["gemini"], fields)

    def test_content_failure_preserved_never_regenerated(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-test-token"}):
            session = Session(Response(200, good(content="not JSON")))
            client = LoggedClient(PROFILES["qwen"], Path(temp) / "log.jsonl", retries=1, session=session)
            result = client.generate("r1", "s", "u", kind="facilitator")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["response"]["output_tokens"], 32)
            self.assertEqual(len(session.calls), 1)
            self.assertEqual(client.generate("r1", "s", "u", kind="facilitator"), result)
            self.assertEqual(len(session.calls), 1)

    def test_service_retry_is_identical_and_logged(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-test-token"}), patch("escna_v3.providers.time.sleep"):
            session = Session(Response(503, {"error": "unavailable"}), Response(200, good()))
            path = Path(temp) / "log.jsonl"
            client = LoggedClient(PROFILES["qwen"], path, retries=1, session=session)
            result = client.generate("r1", "s", "u", kind="facilitator")
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["attempt"], 2)
            self.assertEqual(session.calls[0][1]["json"], session.calls[1][1]["json"])
            self.assertEqual(len(path.read_text().splitlines()), 4)
            self.assertNotIn("fake-test-token", path.read_text())
            self.assertFalse(session.calls[0][1]["allow_redirects"])

    def test_unresolved_request_not_automatically_reissued(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-test-token"}):
            session = Session(KeyboardInterrupt())
            client = LoggedClient(PROFILES["qwen"], Path(temp) / "log.jsonl", session=session)
            with self.assertRaises(KeyboardInterrupt):
                client.generate("r1", "s", "u", kind="facilitator")
            with self.assertRaises(RuntimeError):
                client.generate("r1", "s", "u", kind="facilitator")
            self.assertEqual(len(session.calls), 1)


if __name__ == "__main__":
    unittest.main()
