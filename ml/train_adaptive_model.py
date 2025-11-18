import os, glob, json, numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
LOG_DIR = os.path.join(DATA_DIR, "logs")
MODEL_PATH = os.path.join(DATA_DIR, "adaptive_model.pkl")

def load_domain_skills():
    with open(os.path.join(DATA_DIR, "domain.json"), "r") as f:
        dom = json.load(f)
    return sorted(dom.keys())

def load_mastery(student_id):
    path = os.path.join(DATA_DIR, f"student_{student_id}.json")
    if not os.path.exists(path):
        return {}
    return json.load(open(path))["mastery"]

def build_dataset(skills):
    X, y = [], []

    for file in glob.glob(os.path.join(LOG_DIR, "*.jsonl")):
        student_id = os.path.basename(file).replace(".jsonl", "")
        mastery = load_mastery(student_id)

        streak = 0
        for line in open(file):
            row = json.loads(line)
            skill = row["skill"]
            correct = row["correct"]
            time_ms = row.get("time_ms", 0)

            # features
            X.append([
                skills.index(skill),
                mastery.get(skill, 0.0),
                streak,
                np.log1p(time_ms)
            ])
            y.append(1 if correct else 0)

            streak = streak + 1 if correct else 0

    return np.array(X), np.array(y)

def main():
    skills = load_domain_skills()
    X, y = build_dataset(skills)

    if len(X) == 0:
        print("No training data yet!")
        return

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)

    joblib.dump({"model": clf, "skills": skills}, MODEL_PATH)
    print(f" Saved ML model : {MODEL_PATH}")

if __name__ == "__main__":
    main()
