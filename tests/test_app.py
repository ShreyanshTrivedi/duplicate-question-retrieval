import ast


def test_app_syntax():
    """Verify that app.py parses without syntax errors."""
    with open("src/ui/app.py", encoding="utf-8") as f:
        source = f.read()
    ast.parse(source)
