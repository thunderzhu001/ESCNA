from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from escna_v3.study_tasks import entropy_task, constraint_task
from escna_v3.study_prompt_fix import explicit_evidence_view, strict_participant_contract
from escna_v3.providers import validate_envelope, schema_for, request_payload


class DisclosureFixTests(unittest.TestCase):
    def test_visible_budget_does_not_rewrite_legacy_rule(self):
        profile = {"provider": "gemini", "accepted_response_models": ["gemini-3.7-flash"]}
        fields = {"observed_model": "gemini-3.7-flash", "finish_reason": "STOP",
                  "input_tokens": 1000, "output_tokens": 100, "reasoning_tokens": 479,
                  "completion_including_reasoning": 579, "reasoning_text_present": False}
        with self.assertRaises(ValueError):
            validate_envelope(profile, fields)
        revised = {**profile, "completion_budget_basis": "visible_output"}
        validate_envelope(revised, fields)
        with self.assertRaises(ValueError):
            validate_envelope(revised, {**fields, "output_tokens": 513, "completion_including_reasoning": 992})
        validate_envelope(revised, {**fields, "reasoning_tokens": None})
        with self.assertRaises(ValueError):
            validate_envelope(revised, {**fields, "reasoning_tokens": -1})

    def test_strict_schema_is_shared_without_changing_legacy(self):
        old = schema_for("participant", ("a", "b"), ("R04",))
        new = schema_for("participant", ("a", "b"), ("R04",), strict_disclosure=True)
        self.assertNotIn("enum", old["properties"]["disclose_fact_ids"]["items"])
        self.assertEqual(new["properties"]["disclose_fact_ids"]["items"]["enum"], ["R04"])
        for provider, api in (("qwen", "chat_completions"), ("deepseek", "chat_completions"), ("gemini", "gemini_generate_content")):
            p = request_payload({"provider": provider, "api": api, "model": "test"}, "test", "{}", new)
            instruction = p["systemInstruction"]["parts"][0]["text"] if provider == "gemini" else p["messages"][0]["content"]
            self.assertIn('"enum": ["R04"]', instruction)

    def test_public_and_private_split(self):
        task = entropy_task()
        v = explicit_evidence_view(task, "P4")
        self.assertEqual(v["allowed_new_disclosure_ids"], ["R04"])
        self.assertEqual(len(v["initially_public_records"]), 10)
        self.assertEqual(len(v["your_unpublished_private_records"]), 1)
        self.assertNotIn("R12", str(v))
        schema = strict_participant_contract(task.scenario.options, v["allowed_new_disclosure_ids"])
        self.assertEqual(schema["properties"]["disclose_fact_ids"]["items"]["enum"], ["R04"])

    def test_repeat_publication_and_empty_contract(self):
        task = entropy_task()
        v = explicit_evidence_view(task, "P4", [{"published_records": [{"record_id": "R04"}]}])
        self.assertEqual(v["allowed_new_disclosure_ids"], [])
        s = strict_participant_contract(task.scenario.options, [])
        self.assertEqual(s["properties"]["disclose_fact_ids"]["maxItems"], 0)
        c = explicit_evidence_view(constraint_task(), "P1")
        self.assertEqual(c["allowed_new_disclosure_ids"], [])

if __name__ == "__main__":
    unittest.main()
