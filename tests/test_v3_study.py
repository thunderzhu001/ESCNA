import json
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from escna_v3.study_tasks import acquisition_task, entropy_task, constraint_task, strict_winner, DOMAINS
from escna_v3.study_controller import decide_control
from escna_v3.study_runner import PROTOCOL, run_study, host_view, participant_view


class FakeClient:
    profile = {"provider": "qwen", "model": "offline", "api": "chat_completions",
               "completion_budget_basis": "visible_output", "strict_disclosure": True}
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = [], fail_at
    def generate(self, request_id, system, user, **kw):
        view = json.loads(user)
        self.calls.append({"view": view, "kind": kw["kind"], "system": system})
        if len(self.calls) == self.fail_at:
            return {"request_id": request_id, "status": "failed", "failure_kind": "response_contract"}
        if kw["kind"] == "facilitator":
            parsed = {"message": "Please review and publish your available records."}
        elif kw["kind"] == "participant":
            parsed = {"message": "Here are my records.", "disclose_fact_ids": list(kw["fact_ids"]), "recommendation": kw["options"][0]}
        else:
            parsed = {"recommendation": kw["options"][0], "acceptable_options": list(kw["options"])}
        return {"request_id": request_id, "status": "completed", "parsed": parsed}


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = acquisition_task()

    def test_text_numeric_and_custody(self):
        for domain in DOMAINS:
            for pattern in range(3):
                t = acquisition_task(domain, pattern, pattern+1)
                s = t.scenario
                self.assertEqual(len(s.facts), 12)
                self.assertEqual(strict_winner(s, tuple(s.fact_map())), s.optimal_decision)
                self.assertNotIn(s.optimal_decision, t.metadata["certificate"]["local_winners"].values())
                for f in s.facts:
                    person, values = next(iter(f.utility_effects.items()))
                    option, delta = next(iter(values.items()))
                    self.assertIn(f"{delta:+g}", f.text)
                    self.assertIn(person, f.text)
                    self.assertIn(option, f.text)
                    self.assertNotEqual(person, f.owner)

    def test_b_custody_changes_not_utilities(self):
        k, n = entropy_task(), entropy_task(control="irrelevant")
        self.assertEqual(k.metadata["exposure_counts"], n.metadata["exposure_counts"])
        for f, g in zip(k.scenario.facts, n.scenario.facts):
            self.assertEqual(f.utility_effects, g.utility_effects)
            self.assertEqual(f.is_public, g.is_public)
        self.assertEqual(k.scenario.fact_map()["R04"].owner, "P4")
        self.assertEqual(n.scenario.fact_map()["R04"].owner, "P1")
        for task in (k, n):
            s = task.scenario
            public = s.initial_public_fact_ids()
            self.assertNotEqual(strict_winner(s, public), s.optimal_decision)
            self.assertEqual(strict_winner(s, public | {"R04"}), s.optimal_decision)

    def test_shapley_and_privacy(self):
        s = self.a.scenario
        with self.assertRaises(PermissionError):
            decide_control(s, method="cluster_shapley")
        d = decide_control(s, method="cluster_shapley", complete_brief_authorized=True)
        self.assertEqual(d["private_audit"]["permutations"], 128)
        self.assertIn("approximation_max_abs_error", d["private_audit"])
        payload = json.dumps(host_view(d["safe_action"]))
        for f in s.facts:
            self.assertNotIn(f.fact_id, payload)
            self.assertNotIn(f.text, payload)
        self.assertNotIn("scores", payload)
        with self.assertRaises(ValueError):
            host_view({**d["safe_action"], "private_audit": d["private_audit"]})
        v = participant_view(self.a, "P1", [])
        self.assertEqual(len(v["available_records"]), 3)
        self.assertNotIn("optimal_decision", v)

    def test_formal_numeric_diversity_and_constraint_strata(self):
        gaps = set()
        for di, domain in enumerate(DOMAINS):
            for pattern in range(3):
                task = acquisition_task(domain, pattern, (di+pattern+1) % 3, "formal")
                gaps.add(task.metadata["public_welfare_gap"])
                self.assertEqual(task.metadata["certificate"]["full_information_winner"], task.scenario.optimal_decision)
                self.assertNotIn(task.scenario.optimal_decision, task.metadata["certificate"]["local_winners"].values())
            for control in ("key", "irrelevant"):
                task = entropy_task(domain, 0, control, "formal")
                s = task.scenario
                public = s.initial_public_fact_ids()
                self.assertNotEqual(strict_winner(s, public), s.optimal_decision)
                self.assertEqual(strict_winner(s, public | {"R04"}), s.optimal_decision)
            families = set()
            for variant, control, expected in ((1,"binding",2), (2,"binding",1), (3,"nonbinding",4)):
                task = constraint_task(domain, variant, control, "formal")
                families.add(task.family_id)
                s = task.scenario
                feasible = [o for o in s.options if all(s.base_utilities[p][o] >= s.batna[p] for p in s.participants)]
                self.assertEqual(len(feasible), expected)
            self.assertEqual(len(families), 1)
        self.assertEqual(gaps, {18.,20.,22.})

    def test_entropy_and_constraints(self):
        t = entropy_task()
        d = decide_control(t.scenario, counts=t.metadata["exposure_counts"], method="entropy")
        self.assertEqual(d["safe_action"]["target_owner"], "P4")
        self.assertTrue(0 <= d["private_audit"]["normalized_entropy"] <= 1)
        s = constraint_task().scenario
        advice = decide_control(s, method="constraint")["safe_action"]["public_advice"]
        o = advice["suggested_option"]
        self.assertTrue(all(s.base_utilities[p][o] >= s.batna[p] for p in s.participants))
        impossible = replace(s, batna={p: 100 for p in s.participants})
        self.assertIsNone(decide_control(impossible, method="constraint")["safe_action"]["public_advice"]["suggested_option"])

    def run_fixture(self, task, method, fail=None):
        spec = {"protocol": PROTOCOL, "task_id": task.task_id, "phase": "pilot", "method": method, "seed": 410,
                "run_id": "offline_test", "provider": "qwen"}
        client = FakeClient(fail)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"result.json"
            result = run_study(task, spec, client, path)
            self.assertEqual(run_study(task, spec, client, path), result)
        return client, result

    def test_acquisition_counts_and_sealed_votes(self):
        for method in ("cluster_shapley", "greedy", "structured_exchange"):
            client, r = self.run_fixture(self.a, method)
            self.assertEqual(r["status"], "completed")
            self.assertEqual(len(client.calls), 20)
            self.assertEqual(len(set(r["used_owners"])), 4)
            self.assertEqual([c["owner_budget"] for c in r["checkpoints"]], [2,3,4])
            self.assertEqual(r["checkpoints"][-1]["metrics"]["structured_fact_publication_rate"], 1)
            for call in client.calls:
                self.assertNotIn("ballots", json.dumps(call["view"]))
            self.assertEqual(sum(x["kind"] == "ballot" for x in client.calls), 12)

    def test_content_failure_not_retried_or_erased(self):
        c, r = self.run_fixture(self.a, "structured_exchange", fail=5)
        self.assertEqual(len(c.calls), 20)
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["checkpoints"][0]["metrics"]["accuracy_itt"], 0)
        self.assertEqual(r["checkpoints"][-1]["status"], "completed")
        c, r = self.run_fixture(self.a, "structured_exchange", fail=2)
        self.assertEqual(len(c.calls), 2)
        self.assertTrue(all(c["metrics"]["accuracy_itt"] == 0 for c in r["checkpoints"]))

    def test_formal_private_explanation_never_broadcast(self):
        class ExplanationClient(FakeClient):
            def generate(self, *args, **kwargs):
                row = super().generate(*args, **kwargs)
                if kwargs["kind"] == "participant":
                    row["parsed"]["message"] = "UNREGISTERED_SECRET_991: an unregistered private explanation."
                    row["parsed"]["disclose_fact_ids"] = []
                return row
        task = acquisition_task(phase="formal")
        spec = {"protocol": PROTOCOL, "task_id": task.task_id, "phase": "formal", "method": "structured_exchange",
                "seed": 410, "run_id": "offline_formal", "provider": "qwen", "model": "offline"}
        client = ExplanationClient()
        with tempfile.TemporaryDirectory() as tmp:
            r = run_study(task, spec, client, Path(tmp)/"result.json")
        self.assertEqual(len(r["private_explanations"]), 4)
        self.assertTrue(r["record_channel_only"])
        self.assertNotIn("UNREGISTERED_SECRET_991", json.dumps(r["transcript"]))
        self.assertTrue(all("UNREGISTERED_SECRET_991" not in json.dumps(c["view"]) for c in client.calls))
        self.assertEqual(r["checkpoints"][-1]["metrics"]["structured_fact_publication_rate"], 0)

    def test_provider_model_mismatch_rejected_before_api(self):
        spec = {"protocol": PROTOCOL, "task_id": self.a.task_id, "phase": "pilot", "method": "structured_exchange",
                "seed": 1, "run_id": "mismatch", "provider": "deepseek"}
        client = FakeClient()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_study(self.a, spec, client, Path(tmp)/"result.json")
            with self.assertRaises(ValueError):
                run_study(self.a, {**spec, "provider": "qwen", "model": "wrong"}, client, Path(tmp)/"result.json")
        self.assertEqual(len(client.calls), 0)

    def test_b_c_budgets_and_no_projection(self):
        c, r = self.run_fixture(entropy_task(), "entropy")
        self.assertEqual(len(c.calls), 6)
        c, r = self.run_fixture(constraint_task(), "constraint")
        self.assertEqual(len(c.calls), 9)
        # Fake actors vote for an unsafe alternative; the solver must not
        # replace that outcome with its own feasible suggestion.
        self.assertEqual(r["checkpoints"][0]["evaluation"]["decision"], constraint_task().scenario.options[0])
        self.assertEqual(r["checkpoints"][0]["metrics"]["feasible_agreement_itt"], 0)
        self.assertIn("Do not automatically maximise", c.calls[1]["system"])

if __name__ == "__main__":
    unittest.main()
