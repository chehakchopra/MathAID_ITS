from random import randint, shuffle

def make_mcq(correct):
    wrong = set()
    while len(wrong) < 3:
        val = correct + randint(-5, 5)
        if val != correct:
            wrong.add(val)
    choices = list(wrong) + [correct]
    shuffle(choices)
    return [str(c) for c in choices]

# Difficulty 2
def differentiate():
    while True:
        a = randint(1, 6)
        x = randint(1, 10)
        correct = 2 * a * x

        yield {
            "question": f"d/dx ({a}x²) at x = {x}",
            "answer": str(correct),
            "choices": make_mcq(correct),
            "hint": "Derivative of ax² is 2ax. Plug in x.",
            "difficulty": 2
        }

# Difficulty 3
def integrate():
    while True:
        a = randint(1, 6)
        b = randint(1, 10)
        correct = a * b * b // 2

        yield {
            "question": f"∫ {a}x dx from 0 to {b}",
            "answer": str(correct),
            "choices": make_mcq(correct),
            "hint": "Integral of ax is (a/2)x². Evaluate at limits.",
            "difficulty": 3
        }
