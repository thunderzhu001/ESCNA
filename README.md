# ESCNA: Event-Sourced Collaborative Negotiation Architecture

This repository contains the core mechanism-study implementation for **“Guiding Multi-Agent Deliberation: An Event-Driven Framework for Balanced Information Sharing and Feasible Consensus.”** ESCNA coordinates multi-agent deliberation using explicit control rules alongside language-model participants.

The implementation includes three study mechanisms:

1. **Information acquisition:** rank evidence custodians using sampled Shapley values, with exact calculations for small-case validation and greedy or fixed-order comparisons.
2. **Participation balance:** measure normalized speaking-turn entropy and invite less active participants when the study rule calls for it.
3. **Feasible agreement:** evaluate reservation constraints, nondominance and sealed ballots, and provide public constraint-based negotiation advice.

## Interactive research platform

The companion entry point links to the multi-agent scientific research planning workbench and the ESCNA mechanism experience:

**[Open the interactive research platform](https://argdebate-graph.app.workbuddy.host/?sharecode=UZOTEP76TA)**

The platform is hosted separately. This repository contains the core study code, not the platform's website source.

## Repository contents

- `src/escna_v3/study_controller.py`: custodian selection, entropy-based invitations and constraint advice.
- `src/escna_v3/mechanisms.py`: information valuation, participation and ballot evaluation utilities.
- `src/escna_v3/study_runner.py`, `study_tasks.py` and `study_prompt_fix.py`: mechanism-study workflow, synthetic tasks and disclosure views.
- `src/escna_v3/providers.py`: provider contracts and request logging.
- `src/escna/`: shared scenario models and Shapley utilities used by the study.
- `tests/`: offline tests with simulated provider responses; no API calls are required.
- `examples/offline_demo.py`: a small deterministic controller example using a synthetic task.

This is a **core-code release**. The raw experiment ledgers, processed datasets, full replay and analysis pipeline, manuscript files and platform source are not included. The repository alone therefore does not reproduce the paper's complete empirical results.

## Installation and offline checks

Python 3.10 or later is required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/offline_demo.py
```

The example and tests are offline and do not need credentials. The provider adapter reads API credentials from environment variables when used with a live provider; live calls require a separately configured profile and may incur charges. The example grants complete-brief access only to its synthetic scenario. In an actual deployment, private utility-based ranking requires explicit authorization for the underlying information.

## Citation

If this code helps your research, please cite this repository and, when available, the accompanying paper. For a stable reference to a particular code snapshot, use its GitHub commit URL. GitHub's **Cite this repository** button provides software citation metadata.
