# Conventions — every tool/session must follow this exactly

- Python 3.11, PEP8, format with black, lint with ruff
- Type hints on all function signatures
- Docstrings: one-line summary + Args/Returns (Google style)
- No hardcoded paths or model names — read from config.yaml
- One module = one responsibility; don't merge unrelated logic into one file
- Every retriever implements: rank(query: str, top_k: int) -> list[tuple[str, float]]
- Commit messages: "<phase>: <what>", e.g. "phase3: implement BM25 retriever"
- Do not refactor or rename existing working modules without asking first
- Write at least one test per new module in tests/
