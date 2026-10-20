"""Build the v186 PRE-FREEZE pilot kernel (pilot set only; control labels only) (word control; see v186_extract.py). Build only.
One kernel (account: KAGGLE_ACCOUNT, default aurshalevmerin), T4 x2: each GPU extracts one half of the minimal-pair families (V186_PART), the outputs are
concatenated in order, analysis runs separately in a CPU-only kernel.
"""
import base64, hashlib, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['v186_extract.py', 'v186_pilot.py', 'v178_analyze.py', 'loop_grad_probe.py', 'v186_items.json']
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
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OUT_DIR=od, LOOP_MODEL='ouro', V186_PART=str(gpu), V186_SETS='pilot',
                   PYTHONUNBUFFERED='1', PYTORCH_ALLOC_CONF='expandable_segments:True', HF_HUB_DISABLE_PROGRESS_BARS='1')
        out = open('v186_stdout_gpu%d.txt' % gpu, 'w')
        procs.append((subprocess.Popen([sys.executable, '-u', 'v186_extract.py'], stdout=out, stderr=subprocess.STDOUT, env=env), out))
        time.sleep(60)
    rcs = []
    for p, out in procs:
        rcs.append(p.wait()); out.close(); print('extract exit', rcs[-1], 'elapsed %ds' % (time.time() - t0), flush=True)
    for gpu in range(2):
        print(open('v186_stdout_gpu%d.txt' % gpu).read()[-2000:], flush=True)
    if any(rcs):
        raise SystemExit('extraction failed')
    import numpy as np
    md = os.path.join(W, 'merged'); os.makedirs(md, exist_ok=True)
    np.save(os.path.join(md, 'v186_states.npy'), np.concatenate([np.load(os.path.join(W, 'gpu%d' % g, 'v186_states.npy')) for g in range(2)]))
    for g in range(2):
        os.remove(os.path.join(W, 'gpu%d' % g, 'v186_states.npy'))
    m0, m1 = [json.load(open(os.path.join(W, 'gpu%d' % g, 'v186_meta.json'))) for g in range(2)]
    its = m0['items'] + m1['items']                        # v186 family ids are global (from v186_items.json)
    for k, it in enumerate(its):
        it['k'] = k
    json.dump(dict(m0, n=len(its), items=its, check_logit=max(m0['check_logit'], m1['check_logit']),
                   wall_s=[m0['wall_s'], m1['wall_s']]), open(os.path.join(md, 'v186_meta.json'), 'w'))
    print('merged %d items' % len(its), flush=True)
    with open('v186_pilot.txt', 'w') as out:
        subprocess.call([sys.executable, 'v186_pilot.py', md], stdout=out, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONWARNINGS='ignore'))
    print(open('v186_pilot.txt').read(), flush=True)
'''
dd = os.path.join(HERE, 'kaggle-kernel-v186-pilot'); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\n' % emb + body)
json.dump(dict(id='%s/astrocyte-v186-pilot' % os.environ.get('KAGGLE_ACCOUNT', 'aurshalevmerin'), title='astrocyte-v186-pilot', code_file='run.py', language='python',
               kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
