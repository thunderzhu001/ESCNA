"""Select an evidence custodian in a synthetic scenario without API calls."""

import json

from escna_v3.study_controller import decide_control
from escna_v3.study_tasks import acquisition_task


task = acquisition_task()
decision = decide_control(
    task.scenario,
    method="cluster_shapley",
    complete_brief_authorized=True,  # This example uses synthetic data only.
    seed=81031,
)

print(json.dumps({
    "task_id": task.task_id,
    "safe_action": decision["safe_action"],
    "sampled_scores": decision["private_audit"]["scores"],
    "exact_scores": decision["private_audit"]["exact_scores"],
}, indent=2))
