"""Build the v194 GPU extraction kernel (docs/prereg_v194_*.md). Build only.

  V194_MODE=pilot  python build_kaggle_v194.py   -> astrocyte-v194-pilot: pilot set, then the control-only pilot analysis
  V194_MODE=main   python build_kaggle_v194.py   -> astrocyte-v194-ouro : main + bare sets (analysis in a CPU kernel)

T4 x2: each GPU extracts one half of the selected items (V194_PART); outputs are concatenated in order.
"""
import base64, hashlib, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get('V194_MODE', 'main')
SETS = {'pilot': 'pilot', 'main': 'main,bare'}[MODE]
FILES = ['v194_extract.py', 'loop_grad_probe.py', 'v194_items.json'] + (
    ['v194_pilot.py', 'v178_analyze.py', 'v186_analyze.py'] if MODE == 'pilot' else [])
emb = {}
for f in FILES:
    raw = open(os.path.join(HERE, f), 'rb').read()
    b = raw.replace(b'\r\n', b'\n') if f.endswith('.py') else raw
    emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
    print('  %s sha256 %s' % (f, hashlib.sha256(b).hexdigest()[:16]))
body = r'''import subprocess, sys, os, base64, zlib, time, json, hashlib
if __name__ == '__main__':
    W = '/kaggle/working'; os.chdir(W)
    for fn, e in FILES.items():
        b = zlib.decompress(base64.b64decode(e)); open(fn, 'wb').write(b); print(fn, hashlib.sha256(b).hexdigest()[:16], flush=True)
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.54.1', 'datasets'])
    t0 = time.time(); procs = []
    for gpu in range(2):
        od = os.path.join(W, 'gpu%d' % gpu); os.makedirs(od, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OUT_DIR=od, LOOP_MODEL='ouro', V194_PART=str(gpu), V194_SETS=SETS,
                   PYTHONUNBUFFERED='1', PYTORCH_ALLOC_CONF='expandable_segments:True', HF_HUB_DISABLE_PROGRESS_BARS='1')
        out = open('v194_stdout_gpu%d.txt' % gpu, 'w')
        procs.append((subprocess.Popen([sys.executable, '-u', 'v194_extract.py'], stdout=out, stderr=subprocess.STDOUT, env=env), out))
        time.sleep(60)
    rcs = []
    for p, out in procs:
        rcs.append(p.wait()); out.close(); print('extract exit', rcs[-1], 'elapsed %ds' % (time.time() - t0), flush=True)
    for gpu in range(2):
        print(open('v194_stdout_gpu%d.txt' % gpu).read()[-2000:], flush=True)
    if any(rcs):
        raise SystemExit('extraction failed')
    import numpy as np
    md = os.path.join(W, 'merged'); os.makedirs(md, exist_ok=True)
    np.save(os.path.join(md, 'v194_states.npy'), np.concatenate([np.load(os.path.join(W, 'gpu%d' % g, 'v194_states.npy')) for g in range(2)]))
    for g in range(2):
        os.remove(os.path.join(W, 'gpu%d' % g, 'v194_states.npy'))
    m0, m1 = [json.load(open(os.path.join(W, 'gpu%d' % g, 'v194_meta.json'))) for g in range(2)]
    its = m0['items'] + m1['items']
    for k, it in enumerate(its):
        it['k'] = k
    json.dump(dict(m0, n=len(its), items=its, check_logit=max(m0['check_logit'], m1['check_logit']),
                   wall_s=[m0['wall_s'], m1['wall_s']]), open(os.path.join(md, 'v194_meta.json'), 'w'))
    print('merged %d items' % len(its), flush=True)
    if MODE == 'pilot':
        with open('v194_pilot.txt', 'w') as out:
            rc = subprocess.call([sys.executable, '-u', 'v194_pilot.py', md], stdout=out, stderr=subprocess.STDOUT)
        print('pilot analysis exit', rc); print(open('v194_pilot.txt').read(), flush=True)
'''
kid = {'pilot': 'astrocyte-v194-pilot', 'main': 'astrocyte-v194-ouro'}[MODE]
dd = os.path.join(HERE, 'kaggle-kernel-' + kid.replace('astrocyte-', '')); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
    'FILES = %r\nSETS = %r\nMODE = %r\n' % (emb, SETS, MODE) + body)
json.dump(dict(id='aurshalevmerin/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script',
               is_private=True, enable_gpu=True, enable_internet=True),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
