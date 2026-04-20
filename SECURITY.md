# Security

## Responsible Disclosure

If you find a security issue, please do **not** file a public GitHub issue.

Email: chris.bailey@erp-access.com — include "SECURITY: epistemic-engine" in the subject line.

Expect an acknowledgment within 72 hours.

## What this tool does

epistemic-engine is a library and MCP server for epistemic reasoning: belief propagation (Dempster-Shafer, Subjective Logic), a LeCun-style 6-module world model with SQLite persistence, and LLM flow-control utilities with human-in-the-loop review gates. The `ewm/` module exposes 12 MCP tools.

## What this tool does NOT do

- It does not perform autonomous decision-making in production without a review gate. The `flow_control/` module exists to insert human approval before consequential LLM output is acted on.
- It does not send belief networks, world-model contents, or review-gate data to any external service.
- It does not claim the output is fit for any specific domain without domain-specific validation. Judicial analysis in `ubs/judicial_analyzer.py` is experimental.
- It does not store credentials. LLM client configuration goes through environment variables; nothing is persisted to the repo.

## Known Considerations

- The MCP server in `ewm/` uses local SQLite persistence. If you expose it over a network transport, any caller reaching the port can read or modify the world model.
- Belief-propagation math operates on user-supplied evidence. If the evidence is adversarially crafted, downstream belief values will reflect that — validate input provenance.
- LLM output routed through `flow_control/` is still LLM output. Review gates reduce risk but do not make the output factual. Treat all conclusions as hypotheses.
- Domain modules in `ubs/` (especially `judicial_analyzer.py`) contain factual claims about real jurisdictions and judges. Errors in those claims are correctness bugs, not just style issues.
- The external `belief-math` library is a dependency; security issues there propagate here. Track upstream advisories.

If you see evidence of any of the "does NOT do" items, that is a security issue — please report.
