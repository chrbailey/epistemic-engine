# Testing

This repo is an **experimental research codebase** that depends on several private packages not published to PyPI:

- `belief_math` — belief-propagation primitives
- `guardian` — integrity/hold gatekeeper
- `truth_validator` — human review checklist generator
- `truth_layer` — multi-hop truth propagation

All test files under `tests/` import at least one of these. Because none are publicly installable, **there is no public CI workflow for this repo** — running the tests requires the maintainer's local environment.

## For contributors

If you want to contribute, open an issue first. The maintainer will either:
1. Accept a PR and run the tests locally before merging, OR
2. Work with you to refactor the change so it sits in the narrow slice of logic that does not depend on the private packages.

## For the curious

The only test file without private-package imports is `tests/test_efc_llm_integration.py`, but all of its tests use `pytest.mark.skip(...)` decorators because they require live LLM access that CI cannot provide securely. So the file collects zero runnable tests in a clean public environment.

## Related

For a repo with the same architectural pattern but clean public CI, see [chrbailey/sniperscope](https://github.com/chrbailey/sniperscope) — 144 tests, all local mocks, zero private-dep imports.
