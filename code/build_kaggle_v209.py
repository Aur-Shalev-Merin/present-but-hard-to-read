"""Kaggle kernels for v209 (docs/prereg_v209_ouro_early_patch.md). Build only.
  V209_MODE=plan   astrocyte-v209-plan  (CPU, main account): v209_plan.py on v194's states and frozen plan
                   (kernel_sources astrocyte-v194-ouro, astrocyte-v194-cpu) -> v209_patch_plan.npz
  V209_MODE=smoke  astrocyte-v209-smoke (T4 x1): v209_patch.py on pilot pairs, random subspaces, control conditions only
  V209_MODE=main   astrocyte-v209-main<parts> (T4 x2): parts V209_PARTS (e.g. "0,1") of V209_NPART (default 4); v194's
                   pairs are embedded; the plan is read from astrocyte-v209-plan's output (kernel_sources; embedding it
                   made the script too large for Kaggle) and checked against its frozen sha256 before any patching
                   (amendment 1, before any main run)
  V209_ACCOUNT     aurshalevmerin (default) or aurmerin
"""
import base64, hashlib, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get('V209_MODE', 'smoke'); ACCT = os.environ.get('V209_ACCOUNT', 'aurshalevmerin')
PARTS = [int(p) for p in os.environ.get('V209_PARTS', '0,1').split(',')]; NPART = os.environ.get('V209_NPART', '4')
if MODE == 'plan':
    FILES = {f: f for f in ('v209_plan.py', 'v186_analyze.py', 'v178_analyze.py')}
elif MODE == 'smoke':
    FILES = {f: f for f in ('v209_patch.py', 'loop_grad_probe.py', 'v194_items.json')}
else:
    FILES = {f: f for f in ('v209_patch.py', 'loop_grad_probe.py', 'v194_items.json')}
    FILES['plan/v194_patch_pairs.json'] = 'results_v194/cpu/v194_patch_pairs.json'
emb = {}
for dst, src in FILES.items():
    raw = open(os.path.join(HERE, src), 'rb').read()
    b = raw.replace(b'\r\n', b'\n') if src.endswith('.py') else raw
    emb[dst] = base64.b64encode(zlib.compress(b, 9)).decode()
    print('  %s sha256 %s' % (dst, hashlib.sha256(b).hexdigest()[:16]))
body = r'''import subprocess, sys, os, base64, zlib, time, hashlib
if __name__ == '__main__':
    W = '/kaggle/working'; os.chdir(W)
    for fn, e in FILES.items():
        if os.path.dirname(fn):
            os.makedirs(os.path.dirname(fn), exist_ok=True)
        b = zlib.decompress(base64.b64decode(e)); open(fn, 'wb').write(b); print(fn, hashlib.sha256(b).hexdigest()[:16], flush=True)
    if MODE == 'main':
        import glob, shutil
        src = sorted(glob.glob('/kaggle/input/**/v209_patch_plan.npz', recursive=True))[0]
        h = hashlib.sha256(open(src, 'rb').read()).hexdigest()[:16]; print('plan', src, h, flush=True)
        if h != PLAN_SHA:
            raise SystemExit('plan hash %s != frozen %s' % (h, PLAN_SHA))
        shutil.copy(src, 'plan/v209_patch_plan.npz')
    if MODE == 'plan':
        rc = subprocess.call([sys.executable, '-u', 'v209_plan.py', '/kaggle/input', '/kaggle/input'],
                             env=dict(os.environ, OUT_DIR=W))
        for fn in FILES:
            os.remove(fn)
        raise SystemExit(rc)
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.54.1', 'datasets'])
    t0 = time.time(); procs = []
    for g, part in enumerate(PARTS):
        od = os.path.join(W, 'part%d' % part); os.makedirs(od, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), OUT_DIR=od, LOOP_MODEL='ouro', V209_PART=str(part),
                   V209_NPART=NPART, PYTHONUNBUFFERED='1', PYTORCH_ALLOC_CONF='expandable_segments:True',
                   HF_HUB_DISABLE_PROGRESS_BARS='1')
        out = open('v209_patch_stdout_part%d.txt' % part, 'w')
        procs.append((subprocess.Popen([sys.executable, '-u', 'v209_patch.py', 'extract', 'SMOKE' if MODE == 'smoke' else 'plan'],
                                       stdout=out, stderr=subprocess.STDOUT, env=env), out))
        time.sleep(60)
    rcs = []
    for p, out in procs:
        rcs.append(p.wait()); out.close(); print('patch exit', rcs[-1], 'elapsed %ds' % (time.time() - t0), flush=True)
    for part in PARTS:
        print(open('v209_patch_stdout_part%d.txt' % part).read()[-3000:], flush=True)
    for fn in FILES:
        os.remove(fn)
    if MODE == 'main':
        os.remove('plan/v209_patch_plan.npz')
    if any(rcs):
        raise SystemExit('patching failed')
'''
parts = [0] if MODE == 'smoke' else PARTS
kid = 'astrocyte-v209-' + MODE + (''.join(map(str, parts)) if MODE == 'main' else '')
PLAN_SHA = 'd4e9dc3eb360cf28'
dd = os.path.join(HERE, 'kaggle-kernel-%s-%s' % (kid[len('astrocyte-'):], ACCT)); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
    'FILES = %r\nMODE = %r\nPARTS = %r\nNPART = %r\nPLAN_SHA = %r\n' % (emb, MODE, parts, NPART, PLAN_SHA) + body)
json.dump(dict(id='%s/%s' % (ACCT, kid), title=kid, code_file='run.py', language='python', kernel_type='script',
               is_private=True, enable_gpu=MODE != 'plan', enable_internet=MODE != 'plan',
               kernel_sources={'plan': ['aurshalevmerin/astrocyte-v194-ouro', 'aurshalevmerin/astrocyte-v194-cpu'],
                               'main': ['aurshalevmerin/astrocyte-v209-plan']}.get(MODE, [])),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, ACCT, 'parts', parts)
