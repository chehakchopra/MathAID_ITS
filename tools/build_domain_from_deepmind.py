import sys, os, json, random, re
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from math_dataset import generate

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
DOMAIN_PATH = os.path.join(DATA_DIR, "domain.json")


# ----------------------------------------------------
# DeepMind task → skill mapping
# ----------------------------------------------------
TASK_TO_SKILL = {
    "algebra__linear_1d": "linear-equations",
    "algebra__quadratic": "quadratic-equations",
    "algebra__polynomial_roots": "polynomials",
    "calculus__differentiate": "calculus-basics",
    "calculus__integrate": "calculus-basics",
    "geometry__area": "geometry-trigonometry",
    "geometry__triangles": "geometry-trigonometry",
    "probability__basic": "statistics"
}

GRADES = ["grade9", "grade10", "grade11", "grade12"]


# ----------------------------------------------------
# Sample DeepMind questions
# ----------------------------------------------------
def sample_examples(task_name, n=400):
    gen = generate.dataset(task_name)
    out = []
    for i, ex in enumerate(gen):
        if i >= n:
            break
        out.append((ex["question"], ex["answer"]))
    return out


# ----------------------------------------------------
# Hint generator
# ----------------------------------------------------
def generate_hint(question, answer):
    q = question.lower()

    if "solve" in q and "x" in q:
        return "Isolate x by reversing operations."
    if "derivative" in q:
        return "Use derivative rule: d/dx[x^n] = n*x^(n-1)."
    if "integral" in q:
        return "Use antiderivative rules."
    if "area" in q:
        return "Use geometry area formulas."
    if "triangle" in q:
        return "Recall triangle angle sum/side rules."
    if "probability" in q:
        return "Probability = favorable / total."
    if "root" in q:
        return "Try factoring or substitution."
    return "Think step-by-step about the mathematical operations."


# ----------------------------------------------------
# MCQ distractor generation
# ----------------------------------------------------
def generate_mcq_choices(answer):
    choices = set()
    correct = str(answer).strip()

    # Numeric answers
    try:
        val = float(correct)
        choices.add(correct)
        choices.add(str(round(val + random.uniform(1, 3), 2)))
        choices.add(str(round(val - random.uniform(1, 3), 2)))
        choices.add(str(round(val * random.uniform(0.4, 1.6), 2)))
        return random.sample(list(choices), min(4, len(choices)))
    except:
        pass

    # String answers
    choices.add(correct)
    choices.add("-" + correct if not correct.startswith("-") else correct[1:])
    choices.add(correct + "+1")
    choices.add(correct + "²")

    choices = list(choices)
    random.shuffle(choices)
    return choices[:4]


# ----------------------------------------------------
# Grade classifier: decides grade bucket
# ----------------------------------------------------
def classify_grade(skill, q, a, index):
    q_low = q.lower()

    if skill == "linear-equations":
        return "grade9" if index < 150 else "grade10"

    if skill == "quadratic-equations":
        return "grade11"

    if skill == "polynomials":
        if "root" in q_low:
            return "grade11"
        return "grade10"

    if skill == "calculus-basics":
        if "derivative" in q_low:
            return "grade11"
        return "grade12"

    if skill == "geometry-trigonometry":
        return "grade9" if index < 150 else "grade10"

    if skill == "statistics":
        return "grade9" if index < 200 else "grade10"

    return "grade9"


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
def main():
    domain = {}

    for task, skill in TASK_TO_SKILL.items():
        print(f"[+] Generating {skill} from {task}")
        qas = sample_examples(task, n=300)

        if skill not in domain:
            domain[skill] = {g: [] for g in GRADES}

        for idx, (q, a) in enumerate(qas, start=1):
            grade = classify_grade(skill, q, a, idx)
            hint = generate_hint(q, a)
            choices = generate_mcq_choices(a)

            domain[skill][grade].append({
                "id": f"{skill[:2]}_dm_{idx}",
                "question": q,
                "answer": str(a),
                "choices": choices,
                "hint": hint,
                "difficulty": int(grade[-1])  # grade9 → difficulty 9
            })

    # Merge existing user domain
    if os.path.exists(DOMAIN_PATH):
        with open(DOMAIN_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for skill, block in existing.items():
            domain.setdefault(skill, {g: [] for g in GRADES})
            for g in GRADES:
                domain[skill][g].extend(block.get(g, []))

    with open(DOMAIN_PATH, "w", encoding="utf-8") as f:
        json.dump(domain, f, indent=2)

    print("domain.json Created")


if __name__ == "__main__":
    main()
