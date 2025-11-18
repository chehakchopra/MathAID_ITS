# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import os, json, random, hashlib, time, csv, io, jwt
from ml.hint_engine import generate_hint
import joblib
import numpy as np

app = Flask(__name__, static_folder='static', static_url_path='/')
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
LOGS_DIR = os.path.join(DATA_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

JWT_SECRET = os.environ.get('MATHAID_JWT_SECRET', 'devsecret123')
JWT_ALGO = 'HS256'

# ------------------------------------------------------
# UTILITIES
# ------------------------------------------------------
def _load(name, default):
    try:
        with open(os.path.join(DATA_DIR, name), 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return default

def _save(name, obj):
    with open(os.path.join(DATA_DIR, name), 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)

def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def ensure_student(sid: str):
    """
    Ensure student progress file exists.
    Mastery is tracked per skill (not per grade).
    """
    dom = _load('domain.json', {})
    path = os.path.join(DATA_DIR, f'student_{sid}.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # mastery defaults to 0.0 for every skill in domain
        st = {
            'student_id': sid,
            'mastery': {skill: 0.0 for skill in dom.keys()}
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(st, f, indent=2)
        return st

def student_path(sid: str) -> str:
    return os.path.join(DATA_DIR, f'student_{sid}.json')

def log_path(sid: str) -> str:
    return os.path.join(LOGS_DIR, f'{sid}.jsonl')

def append_log(sid: str, row: dict):
    with open(log_path(sid), 'a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')

def read_logs(sid: str):
    p = log_path(sid)
    if not os.path.exists(p):
        return []
    rows = []
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def token_for(username: str, role: str) -> str:
    payload = {
        'sub': username,
        'role': role,
        'iat': int(time.time()),
        'exp': int(time.time()) + 60 * 60 * 12
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

def require_teacher(req) -> bool:
    try:
        auth = req.headers.get('Authorization', '').replace('Bearer ', '').strip()
        if not auth:
            return False
        data = jwt.decode(auth, JWT_SECRET, algorithms=[JWT_ALGO])
        return data.get('role') == 'teacher'
    except Exception:
        return False

# ------------------------------------------------------
# AUTH
# ------------------------------------------------------
@app.post('/api/auth/register')
def register():
    b = request.get_json(force=True)
    u = b.get('username', '').strip().lower()
    p = b.get('password', '').strip()
    role = b.get('role', 'student')
    if not u or not p:
        return jsonify({'ok': False, 'error': 'missing'}), 400

    users = _load('users.json', {})
    if u in users:
        return jsonify({'ok': False, 'error': 'exists'}), 400

    users[u] = {
        'role': role,
        'password_hash': _hash(p),
        'name': b.get('name', u.title()),
        'classes': []
    }
    _save('users.json', users)

    if role == 'student':
        ensure_student(u)

    return jsonify({'ok': True})

@app.post('/api/auth/login')
def login():
    b = request.get_json(force=True)
    u = b.get('username', '').strip().lower()
    p = b.get('password', '').strip()
    users = _load('users.json', {})
    rec = users.get(u)

    if not rec or rec['password_hash'] != _hash(p):
        return jsonify({'ok': False, 'error': 'bad credentials'}), 401

    token = token_for(u, rec['role'])
    return jsonify({
        'ok': True,
        'token': token,
        'user': {
            'username': u,
            'role': rec['role'],
            'name': rec.get('name', u.title())
        }
    })

# ------------------------------------------------------
# DOMAIN & IMPORT (NESTED GRADE BUCKETS)
# ------------------------------------------------------
def normalize_domain_nested(dom_raw):
    """
    Accept both:
      - new nested format: { skill: { gradeKey: [q...] } }
      - legacy flat list: { skill: [q...] }
    Always return nested dict form.
    """
    out = {}
    for skill, val in dom_raw.items():
        if isinstance(val, dict):
            out[skill] = val
        elif isinstance(val, list):
            # treat legacy list as grade9 by default
            out[skill] = {'grade9': val}
        else:
            out[skill] = {}
    return out

@app.get('/api/domain')
def get_domain():
    dom_raw = _load('domain.json', {})
    return jsonify(dom_raw)

@app.post('/api/domain')
def save_domain():
    if not require_teacher(request):
        return jsonify({'ok': False, 'error': 'forbidden'}), 403
    body = request.get_json(force=True)
    # trust authoring UI; just save raw
    _save('domain.json', body)
    return jsonify({'ok': True})

@app.post('/api/import/csv')
def import_csv():
    """
    CSV columns:
      skill,id,q,a,choices,hint,difficulty,grade
    grade is optional and defaults to 'grade9' if omitted.
    """
    if not require_teacher(request):
        return jsonify({'ok': False, 'error': 'forbidden'}), 403

    text = request.data.decode('utf-8', 'ignore')
    reader = csv.DictReader(io.StringIO(text))
    dom_raw = _load('domain.json', {})
    dom = normalize_domain_nested(dom_raw)

    for row in reader:
        skill = row.get('skill', '').strip()
        if not skill:
            continue

        grade = row.get('grade', '').strip().lower()
        if not grade:
            grade = 'grade9'  # default

        if not grade.startswith('grade'):
            # allow '9' etc.
            if grade.isdigit():
                grade = f'grade{grade}'
            else:
                grade = 'grade9'

        dom.setdefault(skill, {})
        dom[skill].setdefault(grade, [])

        choices = [c.strip() for c in (row.get('choices', '').split('|')) if c.strip()] or None

        dom[skill][grade].append({
            'id'        : row.get('id') or f'{skill[:2]}{len(dom[skill][grade]) + 1}',
            'q'         : row.get('q', ''),
            'a'         : row.get('a', ''),
            'choices'   : choices,
            'hint'      : row.get('hint'),
            'difficulty': int(row.get('difficulty') or 1),
            'grade'     : grade
        })

    _save('domain.json', dom)
    return jsonify({'ok': True, 'skills': list(dom.keys())})

@app.get('/api/skills')
def skills():
    dom_raw = _load('domain.json', {})
    dom = normalize_domain_nested(dom_raw)
    return jsonify({'skills': list(dom.keys())})

@app.get('/api/grades')
def grades():
    """
    Discover all grade keys present in domain.json.
    Returns e.g. ["grade9","grade10","grade11","grade12"].
    """
    dom_raw = _load('domain.json', {})
    dom = normalize_domain_nested(dom_raw)
    grades_set = set()
    for skill, buckets in dom.items():
        if isinstance(buckets, dict):
            for g in buckets.keys():
                grades_set.add(g)
    # sort by numeric part if possible
    def grade_key(g):
        if g.startswith('grade'):
            num = g[5:]
            return int(num) if num.isdigit() else 99
        return 99
    grades_sorted = sorted(grades_set, key=grade_key)
    return jsonify({'grades': grades_sorted})

# ------------------------------------------------------
# CLASSES & ANALYTICS
# ------------------------------------------------------
@app.get('/api/classes')
def classes():
    cls = _load('classes.json', {})
    res = {}
    for cid, c in cls.items():
        avgs = []
        for sid in c.get('students', []):
            st = ensure_student(sid)
            vals = list(st['mastery'].values())
            if vals:
                avgs.append(sum(vals) / len(vals))
        res[cid] = {
            'name': c.get('name', cid),
            'students': c.get('students', []),
            'avg_mastery': round(sum(avgs) / len(avgs), 2) if avgs else 0.0
        }
    return jsonify(res)

@app.get('/api/student/<sid>/progress')
def progress(sid):
    return jsonify(ensure_student(sid))

@app.get('/api/analytics/<sid>')
def analytics(sid):
    logs = read_logs(sid)
    by = {}
    for r in logs:
        s = r['skill']
        d = by.setdefault(s, {
            'attempts': 0,
            'correct': 0,
            'time_ms': [],
            'streak': 0,
            'best_streak': 0
        })
        d['attempts'] += 1
        if r['correct']:
            d['correct'] += 1
            d['streak'] += 1
            d['best_streak'] = max(d['best_streak'], d['streak'])
        else:
            d['streak'] = 0
        if 'time_ms' in r:
            d['time_ms'].append(r['time_ms'])

    for s, d in by.items():
        acc = d['correct'] / d['attempts'] if d['attempts'] else 0
        d['accuracy'] = round(acc, 2)
        d['avg_time_ms'] = int(sum(d['time_ms']) / len(d['time_ms'])) if d['time_ms'] else None
        d.pop('time_ms', None)

    return jsonify(by)

# ------------------------------------------------------
# MODEL (optional adaptive prob; currently unused directly)
# ------------------------------------------------------
MODEL = None
MODEL_SKILLS = []

def load_model():
    global MODEL, MODEL_SKILLS
    path = os.path.join(DATA_DIR, "adaptive_model.pkl")
    if os.path.exists(path):
        blob = joblib.load(path)
        MODEL = blob["model"]
        MODEL_SKILLS = blob["skills"]
    else:
        MODEL = None
        MODEL_SKILLS = []

load_model()

def predict_correct_prob(skill, mastery, streak, avg_time):
    if MODEL is None or skill not in MODEL_SKILLS:
        return 0.7
    idx = MODEL_SKILLS.index(skill)
    feat = np.array([[idx, mastery, streak, np.log1p(avg_time)]])
    return float(MODEL.predict_proba(feat)[0][1])

# ------------------------------------------------------
# ADAPTIVE LOGIC
# ------------------------------------------------------
def prereq_ok(mastery, skill, deps):
    reqs = deps.get(skill, [])
    return all(mastery.get(r, 0.0) >= 0.8 for r in reqs)

def pick_skill(mastery, deps, preferred=None):
    # if user prefers a skill and prereqs OK, honour that
    if preferred and mastery.get(preferred, 0.0) < 1.0 and prereq_ok(mastery, preferred, deps):
        return preferred

    eligible = [k for k, v in mastery.items() if v < 0.95 and prereq_ok(mastery, k, deps)]
    if eligible:
        # lowest mastery first
        return sorted(eligible, key=lambda k: mastery[k])[0]

    # fall back to any not-fully-mastered skill
    for k, v in mastery.items():
        if v < 1.0:
            return k

    return None

def generate_distractors(ans):
    """Simple numeric/text distractor generator."""
    try:
        val = float(ans)
        wrong = set()
        # generate 3 nearby numeric distractors
        while len(wrong) < 3:
            delta = random.randint(-4, 4)
            if delta == 0:
                continue
            candidate = val + delta
            wrong.add(str(int(candidate)) if candidate.is_integer() else str(candidate))
        return list(wrong)
    except Exception:
        # text fallback
        return ["None of these", "Not sure", "Check your work"]

def generate_backend_hint(question_text, explicit_hint=None):
    """Use ML hint_engine if possible, otherwise rule-based text."""
    if explicit_hint:
        return explicit_hint

    try:
        h = generate_hint(question_text, None)
        if h:
            return h
    except Exception:
        pass

    t = (question_text or "").lower()
    if "solve" in t and "x" in t:
        return "Try isolating x step by step. Move constants to one side, x terms to the other."
    if "area" in t:
        return "Recall common area formulas: rectangle = b×h, triangle = ½bh, circle = πr²."
    if "differentiate" in t or "derivative" in t:
        return "Use the power rule: d/dx(xⁿ) = n·xⁿ⁻¹ and handle each term separately."
    if "factor" in t:
        return "Look for common factors or patterns like a²−b² = (a−b)(a+b)."
    return "Break it into smaller steps and write each algebra move clearly."

def select_adaptive_question(domain_nested, skill, grade_key, accuracy):
    """
    domain_nested: { skill: { gradeKey: [ {q,a,choices?,hint?,difficulty?,grade?}, ... ] } }
    skill: string
    grade_key: e.g. 'grade9'
    accuracy: 0..1 (from analytics)
    """
    if skill not in domain_nested:
        return None

    buckets = domain_nested[skill]
    if not isinstance(buckets, dict):
        return None

    # default grade if not present
    if grade_key not in buckets:
        # try downgrade/upgrade gracefully
        for fallback in ['grade9', 'grade10', 'grade11', 'grade12']:
            if fallback in buckets:
                grade_key = fallback
                break
        else:
            # just pick any bucket
            if buckets:
                grade_key = sorted(buckets.keys())[0]
            else:
                return None

    qlist = buckets.get(grade_key, [])
    if not qlist:
        return None

    # difficulty targeting
    if accuracy < 0.4:
        target_diff = 1  # easy
    elif accuracy < 0.7:
        target_diff = 2  # medium
    else:
        target_diff = 3  # hard

    pool = [q for q in qlist if int(q.get('difficulty', 1)) == target_diff]
    if not pool:
        pool = qlist

    q = random.choice(pool)

    # support both {q,a} and {question,answer}
    question_text = q.get('q') or q.get('question') or ""
    correct = q.get('a') or q.get('answer') or ""

    stored_choices = q.get('choices')
    if stored_choices and isinstance(stored_choices, list) and len(stored_choices) >= 2:
        options = list(stored_choices)
        if correct not in options:
            options.append(str(correct))
    else:
        # auto-generate MCQ choices
        distractors = generate_distractors(correct)
        options = distractors + [str(correct)]

    # shuffle options
    random.shuffle(options)

    hint = generate_backend_hint(question_text, q.get('hint'))

    return {
        'id'        : q.get('id'),
        'question'  : question_text,
        'answer'    : str(correct),
        'choices'   : options,
        'difficulty': int(q.get('difficulty', 1)),
        'hint'      : hint,
        'skill'     : skill,
        'grade'     : grade_key
    }

# ------------------------------------------------------
# NEXT QUESTION API
# ------------------------------------------------------
@app.get('/api/next')
def next_q():
    sid = request.args.get('student_id', 'alex')
    preferred_skill = request.args.get('skill')
    grade = request.args.get('grade', 'grade9')  # e.g. 'grade9', 'grade10', ...

    dom_raw = _load('domain.json', {})
    dom = normalize_domain_nested(dom_raw)
    deps = _load('dependencies.json', {})

    st = ensure_student(sid)
    mastery = st['mastery']

    skill = pick_skill(mastery, deps, preferred_skill)
    if not skill:
        return jsonify({
            'done': True,
            'message': 'All skills mastered!'
        })

    # get accuracy from analytics
    stats = json.loads(analytics(sid).get_data())
    accuracy = stats.get(skill, {}).get('accuracy', 0.7)

    q = select_adaptive_question(dom, skill, grade, accuracy)
    if not q:
        return jsonify({'error': f'No questions available for {skill} / {grade}'}), 404

    return jsonify(q)

# ------------------------------------------------------
# ANSWER SUBMISSION
# ------------------------------------------------------
@app.post('/api/answer')
def answer():
    b = request.get_json(force=True)
    sid = b.get('student_id', 'alex')
    skill = b.get('skill')
    correct = bool(b.get('is_correct'))
    tms = int(b.get('time_ms') or 0)

    st = ensure_student(sid)
    m = st['mastery']
    m.setdefault(skill, 0.0)

    # mastery update step (simple)
    m[skill] = max(0.0, min(1.0, m[skill] + (0.12 if correct else -0.06)))

    with open(student_path(sid), 'w', encoding='utf-8') as f:
        json.dump(st, f, indent=2)

    append_log(sid, {
        'ts': int(time.time() * 1000),
        'skill': skill,
        'correct': correct,
        'time_ms': tms,
        'qid': b.get('id'),
        'grade': b.get('grade')
    })

    return jsonify({
        'ok': True,
        'skill': skill,
        'mastery': m[skill]
    })

# ------------------------------------------------------
# FRONTEND ROUTE
# ------------------------------------------------------
@app.get('/')
def root():
    return app.send_static_file('index.html')

# ------------------------------------------------------
# RUN SERVER
# ------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
