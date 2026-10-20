"""Kaggle kernels for v202 (docs/prereg_v202_qwen_swamping.md). Build only.
  V202_MODE=pilot : astrocyte-v202-pilot (CPU): v202_extract.py with V202_PILOT=1 (6 families + 2 base questions on the real
                    model; tokenisation check of every item; timing), then v202_analyze.py --selftest
  V202_MODE=main  : astrocyte-v202-qwen (CPU by default; V202_GPU=1 for a T4): the full extraction, both designs
  V202_MODE=selftest : astrocyte-v202-selftest (CPU): v202_analyze.py --selftest only
  V202_MODE=cpu   : astrocyte-v202-cpu (CPU, kernel_sources astrocyte-v202-qwen): v202_analyze.py
  V202_ACCOUNT    : aurshalevmerin (default) or aurmerin
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get('V202_MODE', 'pilot'); ACCT = os.environ.get('V202_ACCOUNT', 'aurshalevmerin')
GPU = os.environ.get('V202_GPU') == '1'
FILES = {'pilot': ['v202_extract.py', 'v202_analyze.py', 'v178_analyze.py', 'v181_swamp_check.py', 'v178_labels.json'],
         'main': ['v202_extract.py', 'v178_labels.json'],
         'cpu': ['v202_analyze.py', 'v178_analyze.py', 'v181_swamp_check.py', 'v202_extract.py', 'v178_labels.json']}.get(
    MODE, ['v202_analyze.py', 'v178_analyze.py', 'v181_swamp_check.py', 'v202_extract.py', 'v178_labels.json'])
STEPS = {'pilot': [('v202_pilot_extract.txt', ['v202_extract.py'], {'V202_PILOT': '1'}),
                   ('v202_selftest.txt', ['v202_analyze.py', '--selftest'], {})],
         'main': [('v202_extract_stdout.txt', ['v202_extract.py'], {})],
         'cpu': [('v202_analysis.txt', ['v202_analyze.py', '/kaggle/input'], {})],
         'selftest': [('v202_selftest.txt', ['v202_analyze.py', '--selftest'], {})]}[MODE]
emb = {}
for f in FILES:
    raw = open(os.path.join(HERE, f), 'rb').read()
    emb[f] = base64.b64encode(zlib.compress(raw.replace(b'\r\n', b'\n') if f.endswith('.py') else raw, 9)).decode()
body = r'''import subprocess, sys, os, base64, zlib, hashlib
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        b = zlib.decompress(base64.b64decode(e)); open(fn, 'wb').write(b); print(fn, hashlib.sha256(b).hexdigest()[:16], flush=True)
    if MODE in ('pilot', 'main'):
        subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.54.1', 'datasets'])
    for out, args, env in STEPS:
        with open(out, 'w') as fh:
            rc = subprocess.call([sys.executable, '-u'] + args, stdout=fh, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, PYTHONWARNINGS='ignore', OUT_DIR='/kaggle/working', **env))
        print(out, 'exit', rc); print(open(out).read()[-6000:], flush=True)
    for fn in FILES:
        os.remove(fn)
'''
kid = {'pilot': 'astrocyte-v202-pilot', 'main': 'astrocyte-v202-qwen', 'cpu': 'astrocyte-v202-cpu',
       'selftest': 'astrocyte-v202-selftest'}[MODE]
dd = os.path.join(HERE, 'kaggle-kernel-' + kid.replace('astrocyte-', '')); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
    'FILES = %r\nSTEPS = %r\nMODE = %r\n' % (emb, STEPS, MODE) + body)
json.dump(dict(id='%s/%s' % (ACCT, kid), title=kid, code_file='run.py', language='python', kernel_type='script',
               is_private=True, enable_gpu=bool(GPU and MODE == 'main'), enable_internet=MODE in ('pilot', 'main'),
               kernel_sources=['%s/astrocyte-v202-qwen' % ACCT] if MODE == 'cpu' else []),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, 'gpu' if GPU and MODE == 'main' else 'cpu')
