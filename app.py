from flask import Flask, jsonify, request
from flask_cors import CORS
import os, json, random, hashlib, time, csv, io, jwt

app = Flask(__name__, static_folder='static', static_url_path='/')
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
LOGS_DIR = os.path.join(DATA_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

JWT_SECRET = os.environ.get('MATHAID_JWT_SECRET','devsecret123')
JWT_ALGO = 'HS256'

def _load(name, default):
    try:
        with open(os.path.join(DATA_DIR, name), 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return default

def _save(name, obj):
    with open(os.path.join(DATA_DIR, name), 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)

def _hash(pw): return hashlib.sha256(pw.encode()).hexdigest()

def ensure_student(sid):
    dom = _load('domain.json', {})
    path = os.path.join(DATA_DIR, f'student_{sid}.json')
    try:
        return json.load(open(path, 'r', encoding='utf-8'))
    except FileNotFoundError:
        st = {'student_id': sid, 'mastery': {k:0.0 for k in dom}}
        json.dump(st, open(path,'w',encoding='utf-8'), indent=2)
        return st

def student_path(sid): return os.path.join(DATA_DIR, f'student_{sid}.json')
def log_path(sid): return os.path.join(LOGS_DIR, f'{sid}.jsonl')

def append_log(sid, row):
    with open(log_path(sid), 'a', encoding='utf-8') as f: f.write(json.dumps(row)+'\n')

def read_logs(sid):
    p = log_path(sid)
    if not os.path.exists(p): return []
    rows=[]; 
    with open(p,'r',encoding='utf-8') as f:
        for line in f:
            try: rows.append(json.loads(line))
            except: pass
    return rows

def token_for(username, role):
    payload={'sub':username,'role':role,'iat':int(time.time()),'exp':int(time.time())+60*60*12}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

def require_teacher(req):
    try:
        auth = req.headers.get('Authorization','').replace('Bearer ','').strip()
        if not auth: return False
        data = jwt.decode(auth, JWT_SECRET, algorithms=[JWT_ALGO])
        return data.get('role')=='teacher'
    except Exception:
        return False

@app.post('/api/auth/register')
def register():
    b = request.get_json(force=True)
    u = b.get('username','').strip().lower()
    p = b.get('password','').strip()
    role = b.get('role','student')
    if not u or not p: return jsonify({'ok':False,'error':'missing'}), 400
    users = _load('users.json', {})
    if u in users: return jsonify({'ok':False,'error':'exists'}), 400
    users[u] = {'role':role,'password_hash':_hash(p),'name':b.get('name',u.title()),'classes':[]}
    _save('users.json', users)
    if role=='student': ensure_student(u)
    return jsonify({'ok':True})

@app.post('/api/auth/login')
def login():
    b = request.get_json(force=True)
    u = b.get('username','').strip().lower()
    p = b.get('password','').strip()
    users = _load('users.json', {})
    rec = users.get(u)
    if not rec or rec['password_hash'] != _hash(p): return jsonify({'ok':False,'error':'bad credentials'}), 401
    token = token_for(u, rec['role'])
    return jsonify({'ok':True,'token':token,'user':{'username':u,'role':rec['role'],'name':rec.get('name',u.title())}})

@app.get('/api/domain')
def get_domain(): return jsonify(_load('domain.json', {}))

@app.post('/api/domain')
def save_domain():
    if not require_teacher(request): return jsonify({'ok':False,'error':'forbidden'}), 403
    body = request.get_json(force=True)
    _save('domain.json', body)
    return jsonify({'ok':True})

@app.post('/api/import/csv')
def import_csv():
    if not require_teacher(request): return jsonify({'ok':False,'error':'forbidden'}), 403
    text = request.data.decode('utf-8','ignore')
    reader = csv.DictReader(io.StringIO(text))
    dom = _load('domain.json', {})
    for row in reader:
        skill = row.get('skill','').strip()
        if not skill: continue
        dom.setdefault(skill, [])
        choices = [c.strip() for c in (row.get('choices','').split('|')) if c.strip()] or None
        dom[skill].append({'id': row.get('id') or f'{skill[:2]}{len(dom[skill])+1}', 'q':row.get('q',''), 'a':row.get('a',''),
                           'choices':choices,'hint':row.get('hint'),'difficulty':int(row.get('difficulty') or 1)})
    _save('domain.json', dom)
    return jsonify({'ok':True,'skills':list(dom.keys())})

@app.get('/api/skills')
def skills(): d = _load('domain.json', {}); return jsonify({'skills': list(d.keys())})

@app.get('/api/dependencies')
def deps(): return jsonify(_load('dependencies.json', {}))

@app.get('/api/classes')
def classes(): 
    cls = _load('classes.json', {}); res={}
    for cid, c in cls.items():
        avgs=[]; 
        for sid in c.get('students',[]):
            st = ensure_student(sid); vals=list(st['mastery'].values()); 
            if vals: avgs.append(sum(vals)/len(vals))
        res[cid]={'name':c.get('name',cid),'students':c.get('students',[]),'avg_mastery': round(sum(avgs)/len(avgs),2) if avgs else 0.0}
    return jsonify(res)

@app.get('/api/student/<sid>/progress')
def progress(sid): return jsonify(ensure_student(sid))

@app.get('/api/analytics/<sid>')
def analytics(sid):
    logs = read_logs(sid)
    by={}
    for r in logs:
        s=r['skill']; d=by.setdefault(s,{'attempts':0,'correct':0,'time_ms':[],'streak':0,'best_streak':0})
        d['attempts']+=1
        if r['correct']: d['correct']+=1; d['streak']+=1; d['best_streak']=max(d['best_streak'], d['streak'])
        else: d['streak']=0
        if 'time_ms' in r: d['time_ms'].append(r['time_ms'])
    for s,d in by.items():
        acc = d['correct']/d['attempts'] if d['attempts'] else 0
        d['accuracy']=round(acc,2); d['avg_time_ms']= int(sum(d['time_ms'])/len(d['time_ms'])) if d['time_ms'] else None; d.pop('time_ms',None)
    return jsonify(by)

def prereq_ok(mastery, skill, deps):
    reqs = deps.get(skill, [])
    return all(mastery.get(r,0.0) >= 0.8 for r in reqs)

def pick_skill(mastery, deps, preferred=None):
    if preferred and mastery.get(preferred,0.0) < 1.0 and prereq_ok(mastery, preferred, deps): return preferred
    eligible=[k for k,v in mastery.items() if v<0.95 and prereq_ok(mastery,k,deps)]
    if eligible: return sorted(eligible, key=lambda k: mastery[k])[0]
    for k,v in mastery.items():
        if v<1.0: return k
    return None

def pick_question(dom, skill, acc=0.7):
    items = dom.get(skill, [])
    if not items: return None
    target = 2 if acc>0.75 else 1
    pool=[i for i in items if i.get('difficulty',1)==target] or items
    return random.choice(pool)

@app.get('/api/next')
def next_q():
    sid = request.args.get('student_id','alex')
    preferred = request.args.get('skill')
    dom = _load('domain.json', {}); deps = _load('dependencies.json', {})
    st = ensure_student(sid); mastery = st['mastery']
    skill = pick_skill(mastery, deps, preferred)
    if not skill: return jsonify({'done':True,'message':'All skills mastered!'})
    # accuracy from analytics
    stats = json.loads(analytics(sid).get_data()) if callable(analytics) else {}
    acc = stats.get(skill, {}).get('accuracy', 0.7) if isinstance(stats, dict) else 0.7
    q = pick_question(dom, skill, acc=acc)
    if not q: return jsonify({'error': f'No questions for {skill}'}), 404
    return jsonify({'student_id':sid,'skill':skill,'question':q.get('q'),'answer':q.get('a'),'choices':q.get('choices'),
                    'hint':q.get('hint'),'id':q.get('id'),'difficulty':q.get('difficulty',1)})

@app.post('/api/answer')
def answer():
    b = request.get_json(force=True)
    sid = b.get('student_id','alex'); skill=b.get('skill')
    correct=bool(b.get('is_correct')); tms=int(b.get('time_ms') or 0)
    st = ensure_student(sid); m=st['mastery']; m.setdefault(skill,0.0)
    m[skill]=max(0.0, min(1.0, m[skill] + (0.12 if correct else -0.06)))
    json.dump(st, open(student_path(sid),'w',encoding='utf-8'), indent=2)
    append_log(sid, {'ts': int(time.time()*1000), 'skill': skill, 'correct': correct, 'time_ms': tms, 'qid': b.get('id')})
    return jsonify({'ok':True,'skill':skill,'mastery': m[skill]})

@app.get('/')
def root(): return app.send_static_file('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
