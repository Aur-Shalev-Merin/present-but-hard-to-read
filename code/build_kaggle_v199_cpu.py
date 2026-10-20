"""CPU-only Kaggle kernels for v199 (docs/prereg_v199_ouro_running_total.md). Build only.
  V199_CPU=selftest  -> astrocyte-v199-selftest: v199_analyze.py --selftest (synthetic worlds with the real probe; no model states)
  V199_CPU=analysis  -> astrocyte-v199-cpu     : frozen v199_analyze.py on the main states (kernel_sources astrocyte-v199-ouro);
                                                  also writes the Part B patch plan
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get('V199_CPU', 'selftest')
FILES = ['v199_analyze.py', 'v178_analyze.py', 'v186_analyze.py', 'v194_items.json']
emb = {}
for f in FILES:
    raw = open(os.path.join(HERE, f), 'rb').read()
    emb[f] = base64.b64encode(zlib.compress(raw.replace(b'\r\n', b'\n') if f.endswith('.py') else raw, 9)).decode()
body = r'''import subprocess, sys, os, base64, zlib, hashlib
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        b = zlib.decompress(base64.b64decode(e)); open(fn, 'wb').write(b); print(fn, hashlib.sha256(b).hexdigest()[:16], flush=True)
    args = ['--selftest'] if MODE == 'selftest' else ['/kaggle/input']
    out_name = 'v199_selftest.txt' if MODE == 'selftest' else 'v199_analysis.txt'
    with open(out_name, 'w') as out:
        rc = subprocess.call([sys.executable, '-u', 'v199_analyze.py'] + args, stdout=out, stderr=subprocess.STDOUT,
                             env=dict(os.environ, PYTHONWARNINGS='ignore'))
    for fn in FILES:
        os.remove(fn)
    print('exit', rc); print(open(out_name).read()[-15000:])
'''
kid = {'selftest': 'astrocyte-v199-selftest', 'analysis': 'astrocyte-v199-cpu'}[MODE]
dd = os.path.join(HERE, 'kaggle-kernel-' + kid.replace('astrocyte-', '')); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\nMODE = %r\n' % (emb, MODE) + body)
json.dump(dict(id='aurshalevmerin/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script',
               is_private=True, enable_gpu=False, enable_internet=False,
               kernel_sources=['aurshalevmerin/astrocyte-v199-ouro'] if MODE == 'analysis' else []),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
