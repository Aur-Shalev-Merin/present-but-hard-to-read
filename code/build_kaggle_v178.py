"""Build the v178 Kaggle kernel (docs/prereg_v178_ouro_latent_intermediates.md; freeze e99fd90). Build only.
One kernel on aurshalevmerin, T4 x2. The frozen files run UNCHANGED: the wrapper writes the locked labels' first half and
second half to two files and points each GPU's v178_extract.py at one via V178_LABELS (an env var the frozen script
already reads); it then concatenates the outputs in label order and runs the frozen v178_analyze.py on the merged
states (whose asserts check that item order matches the full locked labels).
"""
import base64, hashlib, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['v178_extract.py', 'loop_grad_probe.py', 'v178_analyze.py', 'v178_labels.json']
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
    lab = json.load(open('v178_labels.json')); items = lab['items']; half = (len(items) + 1) // 2
    parts = [items[:half], items[half:]]
    t0 = time.time(); procs = []
    for gpu, part in enumerate(parts):
        od = os.path.join(W, 'gpu%d' % gpu); os.makedirs(od, exist_ok=True)
        lp = os.path.join(od, 'labels_part.json'); json.dump(dict(lab, items=part), open(lp, 'w'))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OUT_DIR=od, LOOP_MODEL='ouro', V178_LABELS=lp,
                   PYTHONUNBUFFERED='1', PYTORCH_ALLOC_CONF='expandable_segments:True', HF_HUB_DISABLE_PROGRESS_BARS='1')
        out = open('v178_stdout_gpu%d.txt' % gpu, 'w')
        procs.append((subprocess.Popen([sys.executable, '-u', 'v178_extract.py'], stdout=out, stderr=subprocess.STDOUT, env=env), out))
        time.sleep(60)
    rcs = []
    for p, out in procs:
        rcs.append(p.wait()); out.close(); print('extract exit', rcs[-1], 'elapsed %ds' % (time.time() - t0), flush=True)
    for gpu in range(2):
        print(open('v178_stdout_gpu%d.txt' % gpu).read()[-3000:], flush=True)
    if any(rcs):
        raise SystemExit('extraction failed')
    import numpy as np
    md = os.path.join(W, 'merged'); os.makedirs(md, exist_ok=True)
    for nm in ('v178_states_last.npy', 'v178_states_q.npy'):
        np.save(os.path.join(md, nm), np.concatenate([np.load(os.path.join(W, 'gpu%d' % g, nm)) for g in range(2)]))
        os.remove(os.path.join(W, 'gpu0', nm)); os.remove(os.path.join(W, 'gpu1', nm))
    m0, m1 = [json.load(open(os.path.join(W, 'gpu%d' % g, 'v178_meta.json'))) for g in range(2)]
    its = m0['items'] + m1['items']
    for k, it in enumerate(its):
        it['k'] = k
    json.dump(dict(m0, n=len(its), items=its, check=[m0['check'], m1['check']], wall_s=[m0['wall_s'], m1['wall_s']]),
              open(os.path.join(md, 'v178_meta.json'), 'w'))
    print('merged %d items' % len(its), flush=True)
    with open('v178_analysis.txt', 'w') as out:
        rc = subprocess.call([sys.executable, 'v178_analyze.py', md, 'v178_labels.json'], stdout=out, stderr=subprocess.STDOUT,
                             env=dict(os.environ, PYTHONWARNINGS='ignore'))
    print('analysis exit', rc, 'elapsed %ds' % (time.time() - t0))
    print(open('v178_analysis.txt').read()[-6000:], flush=True)
'''
dd = os.path.join(HERE, 'kaggle-kernel-v178-ouro'); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\n' % emb + body)
json.dump(dict(id='aurshalevmerin/astrocyte-v178-ouro', title='astrocyte-v178-ouro', code_file='run.py', language='python',
               kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
