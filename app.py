from flask import Flask, jsonify, request
from flask_cors import CORS
import os, json, random, hashlib, time, csv, io, jwt
from ml.hint_engine import generate_hint  # your original hint engine
import joblib
import numpy as np

app = Flask(__name__, static_folder='static', static_url_path='/')
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
LOGS_DIR = os.path.join(DATA_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

JWT_SECRET = os.environ.get('MATHAID_JWT_SECRET', 'devsecret123')
JWT_ALGO = 'HS256'

GRADE_KEYS = ["grade9", "grade10", "grade11", "grade12"]


def _load(name, default):
    try:
        with open(os.path.join(DATA_DIR, name), 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        # Corrupted JSON → reset to default
        return default


def _save(name, obj):
    with open(os.path.join(DATA_DIR, name), 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def student_path(sid: str) -> str:
    return os.path.join(DATA_DIR, f'student_{sid}.json')


def log_path(sid: str) -> str:
    return os.path.join(LOGS_DIR, f'{sid}.jsonl')


def ensure_student(sid: str) -> dict:
    """
    Load / create a student record, auto-healing corrupted JSON
    and ensuring mastery has entries for all current skills.
    """
    dom = _load('domain.json', {})
    path = student_path(sid)

    st = None
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                st = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st = None

    if not st or not isinstance(st, dict):
        st = {'student_id': sid, 'mastery': {}}

    st.setdefault('student_id', sid)
    mastery = st.setdefault('mastery', {})

    # ensure mastery has all current skills
    for skill in dom.keys():
        mastery.setdefault(skill, 0.0)

    # ensure difficulty_state structure exists (for new adaptive difficulty)
    st.setdefault('difficulty_state', {})

    # persist healed / updated record
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(st, f, indent=2)

    return st


def get_difficulty_state(st: dict, skill: str) -> dict:
    """
    Per-student, per-skill difficulty tracking:
      level: 1(easy),2(med),3(hard)
      streak: correct-in-a-row at current level
    """
    ds = st.setdefault('difficulty_state', {})
    state = ds.setdefault(skill, {'level': 1, 'streak': 0})
    try:
        level = int(state.get('level', 1))
    except Exception:
        level = 1
    try:
        streak = int(state.get('streak', 0))
    except Exception:
        streak = 0
    level = max(1, min(3, level))
    streak = max(0, streak)
    state['level'] = level
    state['streak'] = streak
    ds[skill] = state
    return state


def update_difficulty_state(st: dict, skill: str, correct: bool):
    """
    If student answers 2 in a row correctly at a level → bump difficulty.
    If they miss a question → drop difficulty one level (to a minimum of 1).
    """
    state = get_difficulty_state(st, skill)
    level = state['level']
    streak = state['streak']

    if correct:
        streak += 1
        # after 2 correct in a row at this level → go up
        if streak >= 2 and level < 3:
            level += 1
            streak = 0
    else:
        # any mistake knocks difficulty back one step (unless already at easy)
        if level > 1:
            level -= 1
        streak = 0

    state['level'] = level
    state['streak'] = streak
    st.setdefault('difficulty_state', {})[skill] = state


def append_log(sid, row):
    with open(log_path(sid), 'a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')


def read_logs(sid):
    p = log_path(sid)
    if not os.path.exists(p):
        return []
    rows = []
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def token_for(username, role):
    payload = {
        'sub': username,
        'role': role,
        'iat': int(time.time()),
        'exp': int(time.time()) + 60 * 60 * 12
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def require_teacher(req):
    try:
        auth = req.headers.get('Authorization', '').replace('Bearer ', '').strip()
        if not auth:
            return False
        data = jwt.decode(auth, JWT_SECRET, algorithms=[JWT_ALGO])
        return data.get('role') == 'teacher'
    except Exception:
        return False


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
        'classes': [],
        # optional: default grade
        'grade': b.get('grade', 'grade9')
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
            'name': rec.get('name', u.title()),
            'grade': rec.get('grade', 'grade9')
        }
    })

@app.get('/api/domain')
def get_domain():
    return jsonify(_load('domain.json', {}))


@app.post('/api/domain')
def save_domain():
    if not require_teacher(request):
        return jsonify({'ok': False, 'error': 'forbidden'}), 403
    body = request.get_json(force=True)
    _save('domain.json', body)
    return jsonify({'ok': True})


@app.post('/api/import/csv')
def import_csv():
    """
    CSV columns: skill,id,q,a,choices (|),hint,difficulty,grade
    grade optional; if missing → grade9.
    Works with nested grade structure transparently.
    """
    if not require_teacher(request):
        return jsonify({'ok': False, 'error': 'forbidden'}), 403

    text = request.data.decode('utf-8', 'ignore')
    reader = csv.DictReader(io.StringIO(text))
    dom = _load('domain.json', {})

    for row in reader:
        skill = row.get('skill', '').strip()
        if not skill:
            continue

        qtext = row.get('q', '') or row.get('question', '')
        ans = row.get('a', '') or row.get('answer', '')
        grade = row.get('grade', 'grade9').strip().lower()
        if grade not in GRADE_KEYS:
            grade = 'grade9'

        raw_choices = row.get('choices', '')
        choices = [c.strip() for c in raw_choices.split('|') if c.strip()] if raw_choices else None
        hint = row.get('hint') or None
        try:
            difficulty = int(row.get('difficulty') or 1)
        except ValueError:
            difficulty = 1

        dom.setdefault(skill, {})
        if isinstance(dom[skill], list):
            # flatten → convert to nested by putting old list into grade9
            old = dom[skill]
            dom[skill] = {g: [] for g in GRADE_KEYS}
            dom[skill]['grade9'] = old

        dom[skill].setdefault(grade, [])
        dom[skill][grade].append({
            'id': row.get('id') or f'{skill[:2]}{len(dom[skill][grade]) + 1}',
            'q': qtext,
            'a': ans,
            'choices': choices,
            'hint': hint,
            'difficulty': difficulty,
            'grade': grade
        })

    _save('domain.json', dom)
    return jsonify({'ok': True, 'skills': list(dom.keys())})


@app.get('/api/skills')
def skills():
    """
    Optional query param: ?grade=grade9|grade10|grade11|grade12
    For nested domains, returns only skills that have questions for that grade.
    For flat list domains, grade is ignored and all skills are returned.
    """
    grade = request.args.get('grade')
    d = _load('domain.json', {})

    if not grade:
        return jsonify({'skills': list(d.keys())})

    grade = grade.lower()
    out = []
    for skill, payload in d.items():
        if isinstance(payload, dict):
            # nested by grade
            bucket = payload.get(grade)
            if bucket:
                out.append(skill)
        elif isinstance(payload, list):
            if payload:
                out.append(skill)
    return jsonify({'skills': out})


@app.get('/api/dependencies')
def deps():
    return jsonify(_load('dependencies.json', {}))


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


def prereq_ok(mastery, skill, deps):
    reqs = deps.get(skill, [])
    return all(mastery.get(r, 0.0) >= 0.8 for r in reqs)


def pick_skill(mastery, deps, preferred=None):
    # honour preferred skill if reasonable
    if preferred and mastery.get(preferred, 0.0) < 1.0 and prereq_ok(mastery, preferred, deps):
        return preferred

    eligible = [k for k, v in mastery.items() if v < 0.95 and prereq_ok(mastery, k, deps)]
    if eligible:
        return sorted(eligible, key=lambda k: mastery[k])[0]

    for k, v in mastery.items():
        if v < 1.0:
            return k
    return None


def domain_for_skill_by_grade(domain_raw: dict, skill: str) -> dict:
    """
    Normalise domain into: { gradeKey: [questions...] } for a given skill.
    Handles:
      - nested: { "grade9": [...], "grade10": [...] }
      - flat list: [ {...}, {...} ]  (→ everything goes into grade9 unless question has 'grade')
    """
    payload = domain_raw.get(skill)
    if payload is None:
        return {}

    # New nested format
    if isinstance(payload, dict):
        grade_keys_in_payload = [k for k in payload.keys() if k in GRADE_KEYS]
        if grade_keys_in_payload:
            return payload

        # Maybe payload = {"questions": [...]}
        if isinstance(payload.get("questions"), list):
            return {"grade9": payload["questions"]}

        # Fallback: treat all values as a list of questions
        vals = list(payload.values())
        if vals and all(isinstance(v, dict) for v in vals):
            return {"grade9": vals}
        return {}

    # Old format: flat list of questions
    if isinstance(payload, list):
        buckets = {g: [] for g in GRADE_KEYS}
        for q in payload:
            g = (q.get('grade') or 'grade9').lower()
            if g not in GRADE_KEYS:
                g = 'grade9'
            buckets[g].append(q)
        return buckets

    return {}


def numeric_precision(s: str) -> int:
    """How many decimal places does the string representation use?"""
    if '.' in s:
        return len(s.split('.', 1)[1].rstrip('0'))
    return 0


def generate_distractors(ans):
    """
    Stronger distractor generator:
      * For numeric answers, generates close-by values with same scale & precision.
      * For non-numeric, uses generic but not obviously wrong options.
    """
    s = str(ans).strip()

    # Try numeric path first
    try:
        val = float(s)
        decs = numeric_precision(s)
        wrong = set()

        # If answer is 0, use a symmetric range around 0
        if abs(val) < 1e-9:
            while len(wrong) < 3:
                candidate = random.uniform(-5, 5)
                if decs > 0:
                    candidate = round(candidate, decs)
                else:
                    candidate = round(candidate)
                if candidate == val:
                    continue
                wrong.add(str(candidate))
        else:
            # Use relative perturbations so they "look" similar
            while len(wrong) < 3:
                delta_pct = random.uniform(-0.2, 0.2)  # -20% .. +20%
                candidate = val * (1 + delta_pct)
                if decs > 0:
                    candidate = round(candidate, decs)
                else:
                    candidate = round(candidate)
                if candidate == val:
                    continue
                wrong.add(str(candidate))

        return list(wrong)
    except Exception:
        pass

    base = s
    candidates = set()

    if len(base) > 0:
        candidates.add(base.upper())
        candidates.add(base.lower())
        candidates.add(base.title())

    candidates.update({
        "None of these",
        "Cannot be determined",
        "Check your steps again"
    })

    candidates.discard(base)
    out = list(candidates)
    random.shuffle(out)
    return out[:3]


def classify_question_type(text: str) -> str:
    """Rough classification for smarter fallback hints."""
    t = (text or "").lower()

    if "probability" in t or "chance" in t or "odds" in t:
        return "probability"
    if "differentiate" in t or "derivative" in t:
        return "derivative"
    if "integrate" in t or "integral" in t:
        return "integral"
    if "area" in t or "perimeter" in t or "triangle" in t or "rectangle" in t or "circle" in t:
        return "geometry"
    if "quadratic" in t or "x^2" in t or "x²" in t:
        return "quadratic"
    if "solve" in t and "x" in t:
        return "linear"
    return "generic"


def generate_backend_hint(question_text, answer_text):
    """
    1. Try ml.hint_engine.generate_hint(question, answer) – your original engine.
    2. If that fails or returns nothing, give a varied, topic-aware fallback.
    """
    q_str = question_text or ""
    a_str = "" if answer_text is None else str(answer_text)

    try:
        h = generate_hint(q_str, a_str)
        if h:
            return h
    except Exception as e:
        app.logger.warning("hint_engine error for question %r: %s", q_str, e)

    qtype = classify_question_type(q_str)

    if qtype == "linear":
        templates = [
            "Start by getting all the x terms on one side and the numbers on the other, then divide by the coefficient of x.",
            "Undo the operations in reverse order: first subtract/add the constant, then divide or multiply to isolate x.",
            "Think of the equation as a balance: whatever you do to one side, do to the other until x is alone."
        ]
    elif qtype == "quadratic":
        templates = [
            "First put everything on one side so you have 0 on the other, then try factoring or use the quadratic formula.",
            "Check if the quadratic factors nicely; if not, use the quadratic formula x = [-b ± √(b²−4ac)] / (2a).",
            "Make sure the equation is in the form ax² + bx + c = 0 before you decide to factor or apply the formula."
        ]
    elif qtype == "geometry":
        templates = [
            "Identify what shape it is and recall the correct formula (e.g., rectangle: A = b·h, triangle: A = ½·b·h, circle: A = π·r²).",
            "Label the sides/angles you know, then plug them carefully into the appropriate area or perimeter formula.",
            "Draw a quick sketch and mark the given lengths; the formula usually becomes clearer when you see the shape."
        ]
    elif qtype == "derivative":
        templates = [
            "Use the power rule term by term: d/dx(xⁿ) = n·xⁿ⁻¹, then simplify your result.",
            "Differentiate each term separately, apply the power rule, and watch out for any constants or coefficients.",
            "Check for products or quotients; if there are, consider the product or quotient rule instead of just the power rule."
        ]
    elif qtype == "integral":
        templates = [
            "Think of the anti-derivative: reverse the power rule (add 1 to the exponent, then divide by the new exponent).",
            "Look for a simple substitution or a direct power-rule style integral if the function is just xⁿ.",
            "Break the integral into simpler pieces and integrate each term one by one."
        ]
    elif qtype == "probability":
        templates = [
            "Write probability as favourable outcomes ÷ total outcomes, and be sure you are counting each correctly.",
            "List all equally likely outcomes, then count how many satisfy the event described.",
            "Sometimes it’s easier to find the probability of the opposite event and subtract from 1."
        ]
    else:  # generic
        templates = [
            "Break the problem into smaller steps: identify what is given, what is unknown, and which formulas connect them.",
            "Rewrite the key information in your own words, then decide which rule or formula matches the question.",
            "Check units and operations carefully; a lot of mistakes come from a small algebra or arithmetic slip."
        ]

    return random.choice(templates)


def select_adaptive_question(domain_raw, skill, grade, accuracy, difficulty_level=None):
    """
    Choose a question for given skill + grade.
    Primary target: student's current difficulty_level (1/2/3) from difficulty_state.
    Fallback: use accuracy-based buckets.
    Supports both:
      { skill: { grade: [ {q,a,...}, ... ] } }
      { skill: [ {q,a,...}, ... ] }
    """
    gmap = domain_for_skill_by_grade(domain_raw, skill)
    if not gmap:
        return None

    qlist = gmap.get(grade)
    if not qlist:
        # fallback: pool from all grades
        qlist = []
        for bucket in gmap.values():
            qlist.extend(bucket)
    if not qlist:
        return None

    pool = None
    if difficulty_level is not None:
        try:
            lvl = int(difficulty_level)
        except Exception:
            lvl = None
        if lvl in (1, 2, 3):
            cand = [q for q in qlist if int(q.get('difficulty', 1)) == lvl]
            if cand:
                pool = cand

    if pool is None:
        if accuracy < 0.4:
            target_levels = [1]            # easier
        elif accuracy < 0.7:
            target_levels = [1, 2]         # mixed
        else:
            target_levels = [2, 3]         # more challenging

        filtered = [q for q in qlist if int(q.get('difficulty', 1)) in target_levels]
        pool = filtered or qlist

    q = random.choice(pool)

    # Normalise question / answer fields
    raw_q = q.get('q') or q.get('question') or ''
    raw_a = q.get('a')
    if raw_a in (None, ''):
        raw_a = q.get('answer')
    if raw_a is None:
        raw_a = ''

    # Choices (MCQ)
    choices = q.get('choices') or q.get('options') or generate_distractors(raw_a)
    # ensure correct is present
    if str(raw_a) not in [str(c) for c in choices]:
        choices = list(choices) + [str(raw_a)]

    # Deduplicate choices
    seen = set()
    deduped = []
    for c in choices:
        s = str(c)
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    random.shuffle(deduped)

    # Hint: question-authored → ML engine → smart fallback
    hint = q.get('hint')
    if not hint:
        hint = generate_backend_hint(raw_q, raw_a)

    return {
        'id': q.get('id'),
        'skill': skill,
        'grade': grade,
        'question': raw_q,
        'answer': str(raw_a),
        'choices': deduped,
        'difficulty': int(q.get('difficulty', 1)),
        'hint': hint
    }

@app.get('/api/next')
def next_q():
    sid = request.args.get('student_id', 'alex')
    preferred = request.args.get('skill')
    grade = request.args.get('grade', 'grade9').lower()
    mode = request.args.get('mode', 'practice')  # practice | quiz

    if grade not in GRADE_KEYS:
        grade = 'grade9'

    dom = _load('domain.json', {})
    deps = _load('dependencies.json', {})

    st = ensure_student(sid)
    mastery = st['mastery']

    skill = pick_skill(mastery, deps, preferred)
    if not skill:
        return jsonify({'done': True, 'message': 'All skills mastered!'})

    # accuracy from analytics
    stats = json.loads(analytics(sid).get_data())
    accuracy = stats.get(skill, {}).get("accuracy", 0.7)

    # per-student difficulty state
    diff_state = get_difficulty_state(st, skill)
    desired_level = diff_state['level']

    # Optionally tweak difficulty in quiz mode (slightly harder starting point)
    if mode == 'quiz' and desired_level < 3:
        desired_level = min(3, desired_level + 1)

    q = select_adaptive_question(dom, skill, grade, accuracy, difficulty_level=desired_level)
    if not q:
        return jsonify({'error': f'No questions for {skill} in {grade}'}), 404

    return jsonify(q)


@app.post('/api/answer')
def answer():
    b = request.get_json(force=True)
    sid = b.get('student_id', 'alex')
    skill = b.get('skill')
    correct = bool(b.get('is_correct'))
    tms = int(b.get('time_ms') or 0)
    mode = b.get('mode', 'practice')

    st = ensure_student(sid)
    m = st['mastery']
    m.setdefault(skill, 0.0)

    # mastery update
    delta = 0.12 if correct else -0.06
    if mode == 'quiz':
        # slightly stronger impact in quiz mode
        delta = 0.15 if correct else -0.08

    m[skill] = max(0.0, min(1.0, m[skill] + delta))

    # update difficulty tracking (2 correct in a row → up, wrong → down)
    update_difficulty_state(st, skill, correct)

    # persist student
    with open(student_path(sid), 'w', encoding='utf-8') as f:
        json.dump(st, f, indent=2)

    append_log(sid, {
        'ts': int(time.time() * 1000),
        'skill': skill,
        'correct': correct,
        'time_ms': tms,
        'qid': b.get('id'),
        'mode': mode
    })

    return jsonify({
        'ok': True,
        'skill': skill,
        'mastery': m[skill]
    })


def find_question_by_id(domain_raw, skill, qid):
    """
    Search across all grades for a skill to find the question with given id.
    """
    gmap = domain_for_skill_by_grade(domain_raw, skill)
    for bucket in gmap.values():
        for q in bucket:
            if str(q.get('id')) == str(qid):
                return q
    return None


@app.get('/api/mistakes/<sid>')
def mistakes(sid):
    """
    Return recent incorrect attempts (for mistake review panel).
    Optional ?limit=30
    """
    limit = int(request.args.get('limit', 30))
    logs = read_logs(sid)
    dom = _load('domain.json', {})

    mistakes_only = [r for r in logs if not r.get('correct')]
    mistakes_only.reverse()  # newest first

    out = []
    for r in mistakes_only[:limit]:
        skill = r.get('skill')
        qid = r.get('qid')
        q = find_question_by_id(dom, skill, qid) if skill and qid else None

        raw_q = ""
        raw_a = ""
        if q:
            raw_q = q.get('q') or q.get('question') or ''
            raw_a = q.get('a') or q.get('answer') or ''

        out.append({
            'ts': r.get('ts'),
            'skill': skill,
            'qid': qid,
            'question': raw_q,
            'correct_answer': raw_a
        })

    return jsonify(out)


def compute_progress_by_grade(sid):
    """
    For each grade, compute average mastery over skills that
    have questions in that grade.
    """
    st = ensure_student(sid)
    mastery = st['mastery']
    dom = _load('domain.json', {})

    res = {g: {'avg_mastery': 0.0, 'skills': []} for g in GRADE_KEYS}

    for skill, payload in dom.items():
        gmap = domain_for_skill_by_grade(dom, skill)
        for g in GRADE_KEYS:
            bucket = gmap.get(g) or []
            if bucket:
                m_val = mastery.get(skill, 0.0)
                res[g]['skills'].append({'skill': skill, 'mastery': m_val})

    for g in GRADE_KEYS:
        skills = res[g]['skills']
        if skills:
            avg = sum(s['mastery'] for s in skills) / len(skills)
        else:
            avg = 0.0
        res[g]['avg_mastery'] = round(avg, 2)

    return res


@app.get('/api/progress_by_grade/<sid>')
def progress_by_grade(sid):
    return jsonify(compute_progress_by_grade(sid))



@app.get('/')
def root():
    return app.send_static_file('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
