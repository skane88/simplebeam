"""
Nox configuration file.
"""

import nox


@nox.session(python=["3.10", "3.11"])
def all_tests(session):
    session.install(".[dev]")
    session.run("pip", "list")
    session.run("black", "simplebeam", "--check")
    session.run("black", "tests", "--check")
    session.run("isort", ".", "--check-only", "--diff")
    session.run("ruff", "simplebeam")
    session.run("ruff", "tests")
    session.run("pylint", "simplebeam")
    session.run("pylint", "tests")
    session.run("mypy", "simplebeam")
    session.run("coverage", "erase")
    session.run("coverage", "run", "--include=simplebeam/*", "-m", "pytest", "-ra")
    session.run("coverage", "report", "-m")
