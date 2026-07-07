"""The book's Build-it exercises must stay solvable.

Two promises, checked on every test run:

* the shipped **starter** files are healthy: running the graders on them
  yields skips ("your turn"), never errors or failures;
* the shipped **solutions** pass every grader with zero skips -- so as
  the library evolves, an exercise can never quietly become impossible.
"""

import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXERCISES = os.path.join(REPO, "book", "exercises")


def run_graders(solutions):
    env = dict(os.environ, BABYTORCH_DEVICE="cpu")
    env.pop("EXERCISES_SOLUTIONS", None)
    if solutions:
        env["EXERCISES_SOLUTIONS"] = "1"
    # pytest.ini already sets -q; a second -q would silence the summary
    # line these tests read.
    result = subprocess.run(
        [sys.executable, "-m", "pytest", EXERCISES,
         "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=REPO, env=env)
    return result


def test_starters_are_healthy_and_unsolved():
    result = run_graders(solutions=False)
    assert result.returncode == 0, (
        "graders errored on the pristine starter files:\n" + result.stdout)
    assert "failed" not in result.stdout
    assert "skipped" in result.stdout, \
        "starter stubs should register as skips ('your turn')"


def test_solutions_pass_every_grader():
    result = run_graders(solutions=True)
    assert result.returncode == 0, (
        "the shipped solutions no longer pass the graders -- an exercise "
        "has become unsolvable:\n" + result.stdout)
    assert "skipped" not in result.stdout, \
        "solutions must implement every exercise (no skips allowed)"
