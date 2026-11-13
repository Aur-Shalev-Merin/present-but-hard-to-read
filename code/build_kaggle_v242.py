"""Kaggle kernels for v242 (the two-hop clamp at a faint signal; docs/prereg_v242_twohop_faint.md). Build only.
  V242_MODE=smoke   astrocyte-v242-smoke (T4 x2): v242_faint.py with V242_SMOKE=1 (every learned subspace random; mechanics,
                    dose calibration on the GPU states and the whole-state ground truth only)            [pre-freeze]
  V242_MODE=faint   astrocyte-v242-faint (T4 x2): v242_faint.py, the frozen run and its analysis
Account: V242_ME (default aurmerin).
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__)); MODE = os.environ.get('V242_MODE', 'smoke'); ME = os.environ.get('V242_ME', 'aurmerin')
assert MODE in ('smoke', 'faint')
files = ['v235_pilot.py', 'v235_clamp.py', 'v242_faint.py', 'v235_cities.json', 'v235_pilot2_items.json']
env = {'V242_SMOKE': '1', 'V242_N': '40'} if MODE == 'smoke' else {}
steps = [('v242_faint.py', ['run'], env)]; gpu, internet, sources = True, True, []
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode()
       for f in files}
body = open(os.path.join(HERE, 'build_kaggle_v215.py'), encoding='utf-8').read().split("body = r'''")[1].split("'''")[0]
body = body.replace('v215_stdout.txt', 'v242_stdout.txt').replace("'v215 failed", "'v242 failed")
kid = 'astrocyte-v242-' + MODE
dd = os.path.join(HERE, 'kaggle-kernel-v242-' + MODE); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
    'FILES = %r\nSTEPS = %r\nINTERNET = %r\n' % (emb, steps, internet) + body)
json.dump(dict(id=ME + '/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script', is_private=True,
               enable_gpu=gpu, enable_internet=internet, kernel_sources=sources),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, 'gpu' if gpu else 'cpu', [s[0] for s in steps])
