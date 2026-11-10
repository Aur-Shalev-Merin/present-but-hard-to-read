"""Kaggle kernels for v234 (Kudo re-test with equal-answer clamp pairs; docs/prereg_v234_kudo_equal_answer.md). Build only.
  V234_MODE=smoke   astrocyte-v234-smoke (T4 x2): extraction of 40 items (accuracy) + patch mechanics (SMOKE)   [pre-freeze]
  V234_MODE=x       astrocyte-v234-x (T4 x2): extraction of all 14,400 items
  V234_MODE=plan    astrocyte-v234-plan (CPU): bases, share of b left, pairs (from astrocyte-v234-x)
  V234_MODE=patch   astrocyte-v234-patch (T4 x2): the clamp, then its analysis (from astrocyte-v234-plan)
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__)); MODE = os.environ.get('V234_MODE', 'smoke'); ME = 'aurshalevmerin'
BASE = ['v215_pilot.py', 'v215_items.py', 'v216_items.py', 'v234_items.py']
if MODE == 'smoke':
    files = BASE + ['v234_extract.py', 'v234_patch.py']
    steps = [('v234_extract.py', [], {'V234_PILOT': '1'}), ('v234_patch.py', ['extract', 'SMOKE'], {'V234_SMOKE': '1'}),
             ('v234_patch.py', ['analyse', '.'], {})]; gpu, internet, sources = True, True, []
elif MODE == 'x':
    files = BASE + ['v234_extract.py']; steps = [('v234_extract.py', [], {})]; gpu, internet, sources = True, True, []
elif MODE == 'plan':
    files = BASE + ['v234_plan.py']; steps = [('v234_plan.py', ['/kaggle/input'], {})]
    gpu, internet, sources = False, False, [ME + '/astrocyte-v234-x']
elif MODE == 'patch':
    files = BASE + ['v234_patch.py']
    steps = [('v234_patch.py', ['extract', '/kaggle/input'], {}), ('v234_patch.py', ['analyse', '.'], {})]
    gpu, internet, sources = True, True, [ME + '/astrocyte-v234-plan']
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode()
       for f in files}
body = open(os.path.join(HERE, 'build_kaggle_v215.py'), encoding='utf-8').read().split("body = r'''")[1].split("'''")[0]
body = body.replace('v215_stdout.txt', 'v234_stdout.txt').replace("'v215 failed", "'v234 failed")
kid = 'astrocyte-v234-' + MODE
dd = os.path.join(HERE, 'kaggle-kernel-v234-' + MODE); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
    'FILES = %r\nSTEPS = %r\nINTERNET = %r\n' % (emb, steps, internet) + body)
json.dump(dict(id=ME + '/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script', is_private=True,
               enable_gpu=gpu, enable_internet=internet, kernel_sources=sources),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, 'gpu' if gpu else 'cpu', [s[0] for s in steps])
