"""Build the v199 GPU kernels (docs/prereg_v199_ouro_running_total.md). Build only.

  V199_MODE=pilot  python build_kaggle_v199.py -> astrocyte-v199-pilot: GPU 0 extracts the pilot set, then the control-only
                                                  pilot analysis; GPU 1 runs the patching smoke test on pilot pairs
                                                  (control conditions none, rand, btok; random subspaces)
  V199_MODE=main   python build_kaggle_v199.py -> astrocyte-v199-ouro : main set, one half per GPU (analysis in a CPU kernel)
  V199_MODE=patch  python build_kaggle_v199.py -> astrocyte-v199-patch: Part B, one half of the planned pairs per GPU
                                                  (kernel_sources astrocyte-v199-cpu, which writes the plan), then its readout
"""
import base64, hashlib, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get('V199_MODE', 'main')
FILES = ['v199_extract.py', 'v199_patch.py', 'loop_grad_probe.py', 'v194_items.json'] + (
    ['v199_pilot.py', 'v178_analyze.py', 'v186_analyze.py'] if MODE == 'pilot' else [])
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
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OUT_DIR=od, LOOP_MODEL='ouro', PYTHONUNBUFFERED='1',
                   PYTORCH_ALLOC_CONF='expandable_segments:True', HF_HUB_DISABLE_PROGRESS_BARS='1')
        if MODE == 'pilot':
            cmd = ['v199_extract.py'] if gpu == 0 else ['v199_patch.py', 'extract', 'SMOKE']
            env.update(V199_SETS='pilot')
        elif MODE == 'main':
            cmd = ['v199_extract.py']; env.update(V199_SETS='main', V199_PART=str(gpu))
        else:
            cmd = ['v199_patch.py', 'extract', '/kaggle/input']; env.update(V199_PART=str(gpu))
        out = open('v199_stdout_gpu%d.txt' % gpu, 'w')
        procs.append((subprocess.Popen([sys.executable, '-u'] + cmd, stdout=out, stderr=subprocess.STDOUT, env=env), out))
        time.sleep(60)
    rcs = []
    for p, out in procs:
        rcs.append(p.wait()); out.close(); print('exit', rcs[-1], 'elapsed %ds' % (time.time() - t0), flush=True)
    for gpu in range(2):
        print(open('v199_stdout_gpu%d.txt' % gpu).read()[-3000:], flush=True)
    if any(rcs):
        raise SystemExit('a GPU job failed')
    if MODE == 'main':
        import numpy as np
        md = os.path.join(W, 'merged'); os.makedirs(md, exist_ok=True)
        np.save(os.path.join(md, 'v199_states.npy'), np.concatenate([np.load(os.path.join(W, 'gpu%d' % g, 'v199_states.npy')) for g in range(2)]))
        for g in range(2):
            os.remove(os.path.join(W, 'gpu%d' % g, 'v199_states.npy'))
        m0, m1 = [json.load(open(os.path.join(W, 'gpu%d' % g, 'v199_meta.json'))) for g in range(2)]
        its = m0['items'] + m1['items']
        for k, it in enumerate(its):
            it['k'] = k
        json.dump(dict(m0, n=len(its), items=its, check_logit=max(m0['check_logit'], m1['check_logit']),
                       wall_s=[m0['wall_s'], m1['wall_s']]), open(os.path.join(md, 'v199_meta.json'), 'w'))
        print('merged %d items' % len(its), flush=True)
    elif MODE == 'pilot':
        for name, cmd in (('v199_pilot.txt', ['v199_pilot.py', os.path.join(W, 'gpu0')]),
                          ('v199_patchsmoke.txt', ['v199_patch.py', 'analyse', os.path.join(W, 'gpu1')])):
            with open(name, 'w') as out:
                rc = subprocess.call([sys.executable, '-u'] + cmd, stdout=out, stderr=subprocess.STDOUT)
            print(name, 'exit', rc); print(open(name).read(), flush=True)
    else:
        with open('v199_patch_analysis.txt', 'w') as out:
            subprocess.call([sys.executable, '-u', 'v199_patch.py', 'analyse', W], stdout=out, stderr=subprocess.STDOUT)
        print(open('v199_patch_analysis.txt').read(), flush=True)
'''
kid = 'astrocyte-v199-' + {'pilot': 'pilot', 'main': 'ouro', 'patch': 'patch'}[MODE]
dd = os.path.join(HERE, 'kaggle-kernel-' + kid.replace('astrocyte-', '')); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\nMODE = %r\n' % (emb, MODE) + body)
json.dump(dict(id='aurshalevmerin/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script',
               is_private=True, enable_gpu=True, enable_internet=True,
               kernel_sources=['aurshalevmerin/astrocyte-v199-cpu'] if MODE == 'patch' else []),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
