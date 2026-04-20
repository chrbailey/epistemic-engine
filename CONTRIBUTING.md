# Contributing

Thanks for looking.

## Before opening a PR

1. **Open an issue first** for anything larger than a typo.
2. **Changes need tests.** `pytest tests/ -v` must pass.
3. **Respect the module boundaries.** `verity/`, `ubs/`, `ewm/`, `flow_control/` were separate projects for a reason — cross-module coupling needs explicit justification.
4. **The `belief-math` dependency is external.** Do not inline it or vendor it.

## What this project will not accept

epistemic-engine merges four experimental reasoning projects. Code quality varies across modules — contributions should raise the floor, not complicate the ceiling.

- PRs that cross module boundaries without discussion (e.g., `verity/` importing from `ewm/`).
- PRs that remove the human-in-the-loop review gates in `flow_control/`. Those gates exist because LLM reasoning is not yet trustworthy for the domain.
- PRs that change the belief-propagation math without accompanying mathematical justification and tests.
- PRs that bundle the external `belief-math` dependency into this repo.
- PRs that add domain-specific reasoning modules to `ubs/` without tests against a representative input set.
- PRs that modify the N.D. Cal / Judge Alsup jurisdictional context in `flow_control/` without explicit sourcing — that module's correctness depends on factual accuracy about real judges.

## Reporting security issues

See [SECURITY.md](SECURITY.md). Do not file security issues in the public tracker.

## Author

[Christopher Bailey](https://github.com/chrbailey).
