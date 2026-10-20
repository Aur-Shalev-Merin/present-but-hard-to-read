"""Build the v194 Part B GPU kernel (causal patching; docs/prereg_v194_*.md). Build only.
astrocyte-v194-patch: kernel_sources astrocyte-v194-cpu (the frozen analysis writes the patch plan there). T4 x2: each GPU
runs half of the planned pairs (V194_PART); then the frozen Part B readout (v194_patch.py analyse).
"""
import base64, hashlib, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get('V194_PATCH_MODE') == 'smoke'   # pre-freeze code check on pilot pairs, control conditions only
FILES = ['v194_patch.py', 'loop_grad_probe.py', 'v194_items.json']
emb = {}
for f in FILES:
    raw = open(os.path.join(HERE, f), 'rb').read()
    b = raw.replace(b'\r\n', b'\n') if f.endswith('.py') else raw
    emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
    print('  %s sha256 %s' % (f, hashlib.sha256(b).hexdigest()[:16]))
body = r'''import subprocess, sys, os, base64, zlib, time, hashlib
if __name__ == '__main__':
    W = '/kaggle/working'; os.chdir(W)
    for fn, e in FILES.items():
        b = zlib.decompress(base64.b64decode(e)); open(fn, 'wb').write(b); print(fn, hashlib.sha256(b).hexdigest()[:16], flush=True)
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.54.1', 'datasets'])
    t0 = time.time(); procs = []
    for gpu in range(1 if SMOKE else 2):
        od = os.path.join(W, 'gpu%d' % gpu); os.makedirs(od, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OUT_DIR=od, LOOP_MODEL='ouro', V194_PART=str(gpu),
                   PYTHONUNBUFFERED='1', PYTORCH_ALLOC_CONF='expandable_segments:True', HF_HUB_DISABLE_PROGRESS_BARS='1')
        out = open('v194_patch_stdout_gpu%d.txt' % gpu, 'w')
        procs.append((subprocess.Popen([sys.executable, '-u', 'v194_patch.py', 'extract', 'SMOKE' if SMOKE else '/kaggle/input'], stdout=out,
                                       stderr=subprocess.STDOUT, env=env), out))
        time.sleep(60)
    rcs = []
    for p, out in procs:
        rcs.append(p.wait()); out.close(); print('patch exit', rcs[-1], 'elapsed %ds' % (time.time() - t0), flush=True)
    for gpu in range(1 if SMOKE else 2):
        print(open('v194_patch_stdout_gpu%d.txt' % gpu).read()[-3000:], flush=True)
    if any(rcs):
        raise SystemExit('patching failed')
    if SMOKE:
        raise SystemExit(0)
    with open('v194_patch_analysis.txt', 'w') as out:
        subprocess.call([sys.executable, '-u', 'v194_patch.py', 'analyse', W, '/kaggle/input'], stdout=out, stderr=subprocess.STDOUT)
    print(open('v194_patch_analysis.txt').read(), flush=True)
'''
kid = 'astrocyte-v194-patchsmoke' if SMOKE else 'astrocyte-v194-patch'
dd = os.path.join(HERE, 'kaggle-kernel-' + kid.replace('astrocyte-', '')); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\nSMOKE = %r\n' % (emb, SMOKE) + body)
json.dump(dict(id='aurshalevmerin/' + kid, title=kid, code_file='run.py', language='python',
               kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True,
               kernel_sources=[] if SMOKE else ['aurshalevmerin/astrocyte-v194-cpu']),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
