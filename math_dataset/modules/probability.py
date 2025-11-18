from random import randint, shuffle

def make_mcq(correct):
    wrong = set()
    while len(wrong) < 3:
        val = correct + randint(-3, 3)
        if val != correct:
            wrong.add(val)
    choices = list(wrong) + [correct]
    shuffle(choices)
    return [str(c) for c in choices]

# Difficulty 1
def basic():
    while True:
        a = randint(1, 10)
        b = randint(1, 10)
        c = randint(1, 10)
        mean = (a + b + c) / 3
        mean = round(mean, 2)

        yield {
            "question": f"Mean of {a}, {b}, {c}?",
            "answer": str(mean),
            "choices": make_mcq(mean),
            "hint": "Mean = (sum of values) ÷ count.",
            "difficulty": 1
        }
