"""Countdown: puzzles, prompts, and the verifiable reward. No model in this file.

A puzzle is `numbers -> target` where the target is built by actually combining the
numbers with + - * /, so every generated puzzle is solvable by construction. The reward
parses the completion's `<answer>` expression with its own recursive-descent parser
(never `eval`), checks that it uses only the given numbers, each at most once, and that
it equals the target.

Reward tiers, and the reasoning is worth five lines because reward design is where RL
experiments quietly go wrong. 1.0 for a correct expression. 0.1 for a well-formed
`<answer>` that is wrong, so the format itself is learnable at generation zero, which
at N=30 is the difference between a gradient and a flat zero. 0.0 otherwise. This is
the TinyZero-style shape and it is deliberately blunt: no partial credit for "close"
targets, because closeness is not verifiable correctness, and the whole point of
Countdown as an E13 task is that the reward is exact.

Determinism: puzzles are a pure function of a seed. The E13 config commits the seed, so
every arm sees the same puzzle sequence.
"""

import re
from fractions import Fraction
from typing import NamedTuple

import numpy as np

OPS = "+-*/"

PROMPT = (
    "Using the numbers {numbers}, write an arithmetic expression that equals {target}. "
    "Use each number at most once and only + - * /. "
    "Answer with just the expression inside <answer></answer> tags."
)


class Puzzle(NamedTuple):
    numbers: tuple
    target: int

    def prompt(self) -> str:
        return PROMPT.format(numbers=", ".join(map(str, self.numbers)), target=self.target)


def make_puzzles(seed: int, count: int, n_numbers: int = 4) -> list:
    """Solvable by construction: the target is one random full combination of the numbers.

    Numbers are drawn in 1..25 (small pool twice, like the game). Combinations that leave
    integer targets in 1..999 are kept; others are redrawn, so targets stay in a range
    where a 0.5B model's arithmetic is at least on the board.
    """
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        nums = [int(rng.integers(1, 26)) for _ in range(n_numbers)]
        vals = [Fraction(v) for v in nums]
        rng.shuffle(vals)
        acc = vals[0]
        for v in vals[1:]:
            op = OPS[rng.integers(0, 4)]
            if op == "/" and v == 0:
                op = "+"
            acc = acc + v if op == "+" else acc - v if op == "-" else \
                acc * v if op == "*" else (acc / v if v != 0 else acc)
        if acc.denominator == 1 and 1 <= acc <= 999:
            out.append(Puzzle(tuple(nums), int(acc)))
    return out


# ---------------------------------------------------------------- expression checking


class _Parser:
    """Recursive descent over + - * / and parentheses, exact arithmetic via Fraction.

    Rejects anything else (names, calls, unary tricks beyond a leading minus), which is
    the safety property: model output is untrusted text and never reaches eval.
    """

    def __init__(self, text: str):
        self.toks = re.findall(r"\d+|[()+\-*/]", text)
        if "".join(self.toks).replace(" ", "") != re.sub(r"\s+", "", text):
            raise ValueError("unexpected characters")
        self.i = 0
        self.used: list = []

    def _peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _next(self):
        tok = self._peek()
        self.i += 1
        return tok

    def expr(self) -> Fraction:
        v = self.term()
        while self._peek() in ("+", "-"):
            v = v + self.term() if self._next() == "+" else v - self.term()
        return v

    def term(self) -> Fraction:
        v = self.factor()
        while self._peek() in ("*", "/"):
            if self._next() == "*":
                v = v * self.factor()
            else:
                d = self.factor()
                if d == 0:
                    raise ValueError("division by zero")
                v = v / d
        return v

    def factor(self) -> Fraction:
        tok = self._next()
        if tok == "(":
            v = self.expr()
            if self._next() != ")":
                raise ValueError("unbalanced parens")
            return v
        if tok == "-":
            return -self.factor()
        if tok is None or not tok.isdigit():
            raise ValueError(f"expected a number, got {tok!r}")
        self.used.append(int(tok))
        return Fraction(int(tok))


def check(expression: str, puzzle: Puzzle) -> bool:
    """Correct iff it parses, uses only the puzzle's numbers each at most once, and
    equals the target exactly."""
    try:
        p = _Parser(expression)
        value = p.expr()
        if p._peek() is not None:
            return False
    except (ValueError, ZeroDivisionError):
        return False
    pool = list(puzzle.numbers)
    for n in p.used:
        if n not in pool:
            return False
        pool.remove(n)
    return value == puzzle.target


_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def reward(completion: str, puzzle: Puzzle) -> float:
    """1.0 correct, 0.1 well-formed but wrong, 0.0 otherwise. See the module docstring."""
    m = _ANSWER.search(completion)
    if not m:
        return 0.0
    body = m.group(1).strip()
    try:
        _Parser(body)  # well-formed?
    except ValueError:
        return 0.0
    return 1.0 if check(body, puzzle) else 0.1
