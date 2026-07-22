"""Multi-agent privacy system built on LangGraph.

Layout:
  config.py       the ``PrivacyConfig`` a privacy.yaml is loaded into
  pipeline.py     the LangGraph wiring, with the bounded escalation loop
  runner.py       drives the compiled graph for one RAG query
  gate_ui.py      terminal front-end for the human approval gate
  agents/         one module per graph node
  schemas/        the data records the agents exchange
  detection/      PII detection engines, registered as agent tools
  sanitization/   sanitization strategies, registered as agent tools
  domains.py      per-domain defaults (entities, prompts, engine choice)
"""
