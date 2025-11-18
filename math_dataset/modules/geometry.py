from random import randint, shuffle
import math

def make_mcq(correct):
    wrong = set()
    while len(wrong) < 3:
        val = round(correct + randint(-3, 3), 2)
        if val != correct:
            wrong.add(val)
    choices = list(wrong) + [correct]
    shuffle(choices)
    return [str(c) for c in choices]

# Difficulty 1
def area():
    while True:
        w = randint(2, 12)
        h = randint(2, 12)
        correct = w * h

        yield {
            "question": f"Area of rectangle {w} × {h}?",
            "answer": str(correct),
            "choices": make_mcq(correct),
            "hint": "Area = width × height.",
            "difficulty": 1
        }

# Difficulty 2
def triangles():
    while True:
        a = randint(3, 10)
        b = randint(3, 10)
        c = round((a*a + b*b)**0.5, 2)

        yield {
            "question": f"Right triangle legs {a}, {b} → hypotenuse?",
            "answer": str(c),
            "choices": make_mcq(c),
            "hint": "Use Pythagoras: √(a² + b²).",
            "difficulty": 2
        }
