"""Kaggle kernels for v227 (replacement control for v224's share swap; docs/prereg_v227_share_replacement.md), account
aurmerin. Body from build_kaggle_v215.py.
  V227_MODE=pilot    astrocyte-v227-pilot (T4 x2): Mistral-7B-v0.1, 400 random items, b step accuracy only
  V227_MODE=extract  astrocyte-v227-x (T4 x2): 12,000 items, states at layers 18, 23, 27, 32
  V227_MODE=analyse  astrocyte-v227-a (CPU): v224_share.analyse, unchanged
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__)); MODE = os.environ.get('V227_MODE', 'pilot'); ME = 'aurmerin'
ITEMS = ['v215_pilot.py', 'v215_items.py', 'v221_items.py', 'v222_items.py', 'v224_items.py', 'v227_run.py']
if MODE in ('pilot', 'extract'):
    files = ITEMS; steps = [('v227_run.py', [MODE], {})]; gpu, internet, sources = True, True, []
    kid = 'astrocyte-v227-' + ('pilot' if MODE == 'pilot' else 'x')
else:
    files = ITEMS + ['v178_analyze.py', 'v224_share.py']; steps = [('v227_run.py', ['analyse', '/kaggle/input'], {})]
    gpu, internet, sources = False, False, [ME + '/astrocyte-v227-x']; kid = 'astrocyte-v227-a'
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode() for f in files}
body = open(os.path.join(HERE, 'build_kaggle_v215.py'), encoding='utf-8').read().split("body = r'''")[1].split("'''")[0]
body = body.replace('v215_stdout.txt', 'v227_stdout.txt').replace("'v215 failed", "'v227 failed")
dd = os.path.join(HERE, 'kaggle-kernel-' + kid.replace('astrocyte-', '')); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\nSTEPS = %r\nINTERNET = %r\n' % (emb, steps, internet) + body)
json.dump(dict(id=ME + '/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script', is_private=True,
               enable_gpu=gpu, enable_internet=internet, kernel_sources=sources), open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, 'gpu' if gpu else 'cpu', [s[0] for s in steps])
