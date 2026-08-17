"""The Countdown reward: exact where it must be, closed where it must be.

The reward is the ground truth of E13. A parser hole (eval, unary abuse, reusing a
number) inflates every arm equally in the best case and one arm silently in the worst,
so these tests are about the reward's *edges*, not its happy path.
"""

import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "countdown"
spec = importlib.util.spec_from_file_location("countdown_task", HERE / "task.py")
task = importlib.util.module_from_spec(spec)
sys.modules["countdown_task"] = task
spec.loader.exec_module(task)


P = task.Puzzle((3, 5, 7, 2), 21)


def test_puzzles_are_solvable_deterministic_and_in_range():
    a, b = task.make_puzzles(0, 50), task.make_puzzles(0, 50)
    assert a == b, "puzzles must be a pure function of the seed"
    assert task.make_puzzles(1, 50) != a
    for p in a:
        assert 1 <= p.target <= 999 and len(p.numbers) == 4


def test_correct_expressions_score_one():
    assert task.reward("<answer>3 * 7</answer>", P) == 1.0
    assert task.reward("<answer>7 * (5 - 2)</answer>", P) == 1.0


def test_wrong_but_well_formed_scores_a_tenth():
    assert task.reward("<answer>3 + 5</answer>", P) == 0.1


def test_garbage_and_missing_tags_score_zero():
    assert task.reward("21", P) == 0.0
    assert task.reward("<answer>twenty one</answer>", P) == 0.0
    assert task.reward("<answer>import os</answer>", P) == 0.0
    assert task.reward("<answer>3 * 7", P) == 0.0  # unclosed tag


def test_numbers_outside_the_pool_are_wrong():
    assert task.check("21", P) is False
    assert task.check("3 * 7 * 1", P) is False


def test_each_number_at_most_once():
    assert task.check("3 * 3 + 5 + 7", task.Puzzle((3, 5, 7), 21)) is False
    # but a duplicated number in the pool may be used twice
    assert task.check("3 * 3", task.Puzzle((3, 3, 5), 9)) is True


def test_division_is_exact_not_float():
    assert task.check("22 / 7 * 7", task.Puzzle((22, 7, 7), 22)) is True
    assert task.check("1 / 3 * 3", task.Puzzle((1, 3, 3), 1)) is True  # floats would say 0.9999...


def test_division_by_zero_is_wrong_not_a_crash():
    assert task.check("3 / (5 - 5)", task.Puzzle((3, 5, 5), 1)) is False


def test_the_parser_never_evals():
    for evil in ("__import__('os')", "3 ** 9 ** 9", "a", "3 if 1 else 5"):
        assert task.check(evil, P) is False


def test_first_generation_format_signal_exists():
    """The 0.1 tier exists so N=30 sees a gradient at generation zero. If every tier
    below correct collapses to 0, that design decision has been silently lost."""
    assert 0.0 < task.reward("<answer>3 + 5</answer>", P) < 1.0


def test_eval_puzzles_disjoint_from_train():
    """The held-out set must never contain a training puzzle: numbers come from a
    small pool, so seed separation alone does not give disjointness and the
    generator filters. Identity is the (numbers, target) pair."""
    train = task.make_puzzles(7, 64)
    ev = task.make_eval_puzzles(1007, 128, train)
    assert len(ev) == 128
    assert not set(ev) & set(train)
    assert len(set(ev)) == 128, "eval set has internal duplicates"
    again = task.make_eval_puzzles(1007, 128, train)
    assert ev == again, "eval set is not deterministic"
