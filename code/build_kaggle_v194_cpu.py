"""CPU-only Kaggle kernels for v194 (docs/prereg_v194_*.md). Build only.
  V194_CPU=selftest  -> astrocyte-v194-selftest: v194_analyze.py --selftest (synthetic worlds; no model states)
  V194_CPU=analysis  -> astrocyte-v194-cpu     : frozen v194_analyze.py on the main states (kernel_sources astrocyte-v194-ouro);
                                                  also writes the Part B patch plan
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get('V194_CPU', 'selftest')
FILES = ['v194_analyze.py', 'v178_analyze.py', 'v186_analyze.py', 'v194_items.json']
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
    out_name = 'v194_selftest.txt' if MODE == 'selftest' else 'v194_analysis.txt'
    with open(out_name, 'w') as out:
        rc = subprocess.call([sys.executable, '-u', 'v194_analyze.py'] + args, stdout=out, stderr=subprocess.STDOUT,
                             env=dict(os.environ, PYTHONWARNINGS='ignore'))
    for fn in FILES:
        os.remove(fn)
    print('exit', rc); print(open(out_name).read()[-15000:])
'''
kid = {'selftest': 'astrocyte-v194-selftest', 'analysis': 'astrocyte-v194-cpu'}[MODE]
dd = os.path.join(HERE, 'kaggle-kernel-' + kid.replace('astrocyte-', '')); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\nMODE = %r\n' % (emb, MODE) + body)
json.dump(dict(id='aurshalevmerin/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script',
               is_private=True, enable_gpu=False, enable_internet=False,
               kernel_sources=['aurshalevmerin/astrocyte-v194-ouro'] if MODE == 'analysis' else []),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
