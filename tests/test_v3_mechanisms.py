import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from escna.models import PrivateFact, Scenario
from escna_v3.mechanisms import (
    AuthorizationError,
    evaluate_sealed_ballots,
    full_truth_decision,
    nondominated_options,
    normalized_entropy,
    revised_information_value,
    robust_utility,
    sealed_ballot,
    choose_information_action,
    surrogate_decision,
    validate_shapley_cost,
)


def information_scenario() -> Scenario:
    participants = ("P1", "P2", "P3")
    options = ("A", "B", "C")
    base = {person: {"A": 10.0, "B": 9.0, "C": 0.0} for person in participants}
    facts = (
        PrivateFact(
            "bad_news",
            "P1",
            "SECRET BAD NEWS SHOULD NEVER ENTER LANGUAGE PAYLOAD",
            "feasibility",
            {"P1": {"A": -8.0}},
        ),
        PrivateFact("irrelevant", "P2", "irrelevant", "noise", {"P2": {"C": 0.0}}),
        PrivateFact("positive", "P3", "positive", "benefit", {"P3": {"B": 1.0}}),
    )
    return Scenario(
        "info",
        "test",
        1,
        "hidden",
        participants,
        options,
        base,
        {person: 5.0 for person in participants},
        facts,
        "B",
    )


def constraint_scenario() -> Scenario:
    return Scenario(
        "constraint",
        "test",
        1,
        "symmetric",
        ("P1", "P2"),
        ("A", "B", "C"),
        {
            "P1": {"A": 20.0, "B": 8.0, "C": 6.0},
            "P2": {"A": 1.0, "B": 8.0, "C": 6.0},
        },
        {"P1": 5.0, "P2": 5.0},
        (),
        "A",
    )


class V3MechanismTests(unittest.TestCase):
    def test_entropy_is_normalized_and_bounded(self):
        self.assertEqual(normalized_entropy([]), 0.0)
        self.assertEqual(normalized_entropy([1, 0, 0, 0]), 0.0)
        self.assertEqual(normalized_entropy([1, 1, 1, 1]), 1.0)
        self.assertTrue(0.0 <= normalized_entropy([10, 10, 10, 1]) <= 1.0)
        with self.assertRaises(ValueError):
            normalized_entropy([1, -1])

    def test_surrogate_tie_rule_is_deterministic_but_final_tie_is_no_decision(self):
        scenario = information_scenario()
        self.assertEqual(surrogate_decision(scenario, ()), "A")
        self.assertIsNone(
            sealed_ballot(
                {
                    "P1": {"recommendation": "A", "acceptable_options": ["A"]},
                    "P2": {"recommendation": "B", "acceptable_options": ["B"]},
                },
                scenario.options,
            )
        )
        self.assertIsNone(
            sealed_ballot(
                {
                    "P1": {"recommendation": "ABSTAIN", "acceptable_options": []},
                    "P2": {"recommendation": "ABSTAIN", "acceptable_options": []},
                },
                scenario.options,
            )
        )

    def test_revised_value_can_select_negative_bad_news(self):
        scenario = information_scenario()
        self.assertGreater(revised_information_value(scenario, (), ("bad_news",)), 0.0)
        self.assertEqual(revised_information_value(scenario, (), ("irrelevant",)), 0.0)
        selection = choose_information_action(scenario, complete_brief_authorized=True)
        self.assertEqual(selection.target_owner, "P1")
        self.assertEqual(selection.selected_fact_ids, ("bad_news",))

    def test_private_brief_requires_consent_and_language_payload_is_text_free(self):
        scenario = information_scenario()
        with self.assertRaises(AuthorizationError):
            choose_information_action(scenario)
        selection = choose_information_action(
            scenario,
            method="greedy",
            complete_brief_authorized=True,
            candidate_fact_ids=("bad_news",),
            timing=3,
        )
        payload = selection.language_payload()
        self.assertEqual(payload, {"action": "invite_owner", "target_owner": "P1", "timing": 3})
        self.assertNotIn("bad_news", payload)
        self.assertNotIn("SECRET", str(payload))
        self.assertNotIn("text", payload)

    def test_scheduled_owner_and_round_robin_are_deterministic(self):
        scenario = information_scenario()
        selection = choose_information_action(
            scenario,
            method="round_robin",
            complete_brief_authorized=True,
            scheduled_owner="P2",
            timing="before_vote",
        )
        self.assertEqual(selection.target_owner, "P2")
        self.assertEqual(selection.selected_fact_ids, ("irrelevant",))
        self.assertEqual(selection.metadata["timing"], "before_vote")

    def test_utility_uses_full_truth_when_fact_ids_omitted(self):
        scenario = information_scenario()
        partial = robust_utility(scenario, "A", ())
        full = robust_utility(scenario, "A")
        self.assertEqual(partial["P1"], 10.0)
        self.assertEqual(full["P1"], 2.0)
        self.assertEqual(robust_utility(scenario, "A", participant="P1"), 2.0)

    def test_ballot_resolution_counts_only_valid_recommendations(self):
        ballots = {
            "P1": {"recommendation": "B", "acceptable_options": ["B"]},
            "P2": {"recommendation": "B", "acceptable_options": ["A", "B"]},
            "P3": {"recommendation": "A", "acceptable_options": ["A"]},
            "P4": {"recommendation": "ABSTAIN", "acceptable_options": []},
        }
        self.assertEqual(sealed_ballot(ballots, ("A", "B", "C")), "B")

    def test_metrics_are_separate_and_no_decision_gets_batna_payoff(self):
        scenario = constraint_scenario()
        accepted_b = {
            "P1": {"recommendation": "B", "acceptable_options": ["B"]},
            "P2": {"recommendation": "B", "acceptable_options": ["B"]},
        }
        evaluation = evaluate_sealed_ballots(scenario, accepted_b)
        self.assertEqual(evaluation.decision, "B")
        self.assertFalse(evaluation.full_truth_accuracy)
        self.assertTrue(evaluation.unanimity_accept)
        self.assertTrue(evaluation.batna_satisfied)
        self.assertTrue(evaluation.nondominated)
        # C meets BATNA but is strictly dominated by B; these metrics differ.
        dominated_c = {
            "P1": {"recommendation": "C", "acceptable_options": ["C"]},
            "P2": {"recommendation": "C", "acceptable_options": ["C"]},
        }
        dominated = evaluate_sealed_ballots(scenario, dominated_c)
        self.assertTrue(dominated.batna_satisfied)
        self.assertFalse(dominated.nondominated)
        self.assertEqual(nondominated_options(scenario), ("A", "B"))
        tied = evaluate_sealed_ballots(
            scenario,
            {
                "P1": {"recommendation": "A", "acceptable_options": ["A"]},
                "P2": {"recommendation": "B", "acceptable_options": ["B"]},
            },
        )
        self.assertTrue(tied.no_decision)
        self.assertEqual(tied.payoffs, scenario.batna)
        self.assertFalse(tied.full_truth_accuracy)
        self.assertFalse(tied.unanimity_accept)
        self.assertFalse(tied.batna_satisfied)

        not_unanimous = {
            "P1": {"recommendation": "B", "acceptable_options": ["B"]},
            "P2": {"recommendation": "B", "acceptable_options": ["B"]},
            "P3": {"recommendation": "A", "acceptable_options": ["A"]},
        }
        failed_agreement = evaluate_sealed_ballots(information_scenario(), not_unanimous)
        self.assertEqual(failed_agreement.decision, "B")
        self.assertFalse(failed_agreement.unanimity_accept)
        self.assertEqual(failed_agreement.payoffs, information_scenario().batna)
        self.assertEqual(failed_agreement.decision_utilities, {"P1": 9.0, "P2": 9.0, "P3": 10.0})

    def test_full_truth_tie_does_not_get_an_accuracy_label(self):
        scenario = Scenario(
            "truth_tie",
            "test",
            1,
            "symmetric",
            ("P1", "P2"),
            ("A", "B"),
            {"P1": {"A": 1.0, "B": 1.0}, "P2": {"A": 1.0, "B": 1.0}},
            {"P1": 0.0, "P2": 0.0},
            (),
            "A",
        )
        ballots = {
            "P1": {"recommendation": "A", "acceptable_options": ["A"]},
            "P2": {"recommendation": "A", "acceptable_options": ["A"]},
        }
        result = evaluate_sealed_ballots(scenario, ballots)
        self.assertIsNone(result.full_truth_decision)
        self.assertIsNone(result.full_truth_accuracy)

    def test_abstain_with_acceptance_set_is_rejected(self):
        with self.assertRaises(ValueError):
            sealed_ballot(
                {"P1": {"recommendation": "ABSTAIN", "acceptable_options": ["A"]}},
                ("A", "B"),
            )

    def test_exact_permutation_and_cluster_costs_report_na_for_constant_rank(self):
        scenario = Scenario(
            "zero",
            "test",
            1,
            "hidden",
            ("P1", "P2"),
            ("A", "B"),
            {"P1": {"A": 1.0, "B": 1.0}, "P2": {"A": 1.0, "B": 1.0}},
            {"P1": 0.0, "P2": 0.0},
            (
                PrivateFact("z1", "P1", "zero one", "z", {"P1": {"A": 0.0}}),
                PrivateFact("z2", "P2", "zero two", "z", {"P2": {"B": 0.0}}),
            ),
            "A",
        )
        result = validate_shapley_cost(scenario, samples=4)
        self.assertEqual(result["exact_scores"], {"z1": 0.0, "z2": 0.0})
        self.assertEqual(result["permutation_scores"], {"z1": 0.0, "z2": 0.0})
        self.assertIsNone(result["rank_exact_vs_permutation"])
        self.assertIsNone(result["rank_exact_vs_cluster"])
        self.assertGreater(result["exact_value_calls"], 0)
        self.assertGreater(result["permutation_value_calls"], 0)
        self.assertGreaterEqual(result["cluster_value_calls"], 0)


if __name__ == "__main__":
    unittest.main()
