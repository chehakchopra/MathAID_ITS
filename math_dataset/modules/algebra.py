from random import randint, choice, shuffle

def make_mcq(correct, wrong_range=10):
    """Generate 3 wrong answers and return shuffled MCQ list."""
    wrong = set()
    while len(wrong) < 3:
        val = correct + randint(-wrong_range, wrong_range)
        if val != correct:
            wrong.add(val)
    choices = list(wrong) + [correct]
    shuffle(choices)
    return [str(c) for c in choices]

# ---------------------------------------------
# LINEAR EQUATIONS (difficulty 1)
# ---------------------------------------------
def linear_1d():
    while True:
        a = randint(1, 5)
        x = randint(-10, 10)
        b = randint(-10, 10)
        c = a * x + b

        yield {
            "question": f"Solve: {a}x + {b} = {c}",
            "answer": str(x),
            "choices": make_mcq(x),
            "hint": f"Move {b} to the other side → {a}x = {c - b}. Then divide by {a}.",
            "difficulty": 1
        }

# ---------------------------------------------
# QUADRATIC EQUATIONS (difficulty 2)
# ---------------------------------------------
def quadratic():
    while True:
        r1 = randint(-6, 6)
        r2 = randint(-6, 6)
        b = -(r1 + r2)
        c = r1 * r2

        correct = max(r1, r2)

        yield {
            "question": f"Solve: x² + ({b})x + ({c}) = 0 (larger root)",
            "answer": str(correct),
            "choices": make_mcq(correct, wrong_range=5),
            "hint": f"Factor into (x - {r1})(x - {r2}). Pick the larger root.",
            "difficulty": 2
        }

# ---------------------------------------------
# POLYNOMIAL ROOTS (difficulty 3)
# ---------------------------------------------
def polynomial_roots():
    while True:
        r1 = randint(-5, 5)
        r2 = randint(-5, 5)
        r3 = randint(-5, 5)

        a2 = -(r1 + r2)
        b2 = r1 * r2
        a = a2 - r3
        b = b2 - a2 * r3
        c = -b2 * r3

        correct = choice([r1, r2, r3])

        yield {
            "question":
                f"Solve: x³ + ({a})x² + ({b})x + ({c}) = 0 (give one real root)",
            "answer": str(correct),
            "choices": make_mcq(correct, wrong_range=7),
            "hint": f"Roots are {r1}, {r2}, and {r3}.",
            "difficulty": 3
        }
