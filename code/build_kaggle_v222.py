"""Kaggle kernels for v222 (clamp power at a distance; docs/prereg_v222_clamp_distance.md).
Build only; output streamed live to the console (body from build_kaggle_v215.py).
  V222_MODE=smoke   astrocyte-v222-smoke (T4 x2): 40 items extracted, 20 pairs with random bases (mechanics)
  V222_MODE=run     astrocyte-v222-run (T4 x2): extraction (12,000 items), plan, clamp (1,000 pairs), analysis
V222_ACCOUNT: the Kaggle account in the kernel id (default aurmerin; the kernel reads no other kernel's output).
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get('V222_MODE', 'smoke')
ME = os.environ.get('V222_ACCOUNT', 'aurmerin')
files = ['v215_pilot.py', 'v215_items.py', 'v221_items.py', 'v222_items.py', 'v215_analyze.py', 'v178_analyze.py', 'v222_clamp.py']
steps = [('v222_clamp.py', ['all'], {'V222_SMOKE': '1'} if MODE == 'smoke' else {})]
gpu, internet, sources = True, True, []
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode()
       for f in files}
body = open(os.path.join(HERE, 'build_kaggle_v215.py'), encoding='utf-8').read().split("body = r'''")[1].split("'''")[0]
body = body.replace('v215_stdout.txt', 'v222_stdout.txt').replace("'v215 failed", "'v222 failed")
kid = 'astrocyte-v222-' + MODE
dd = os.path.join(HERE, 'kaggle-kernel-v222-' + MODE); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
    'FILES = %r\nSTEPS = %r\nINTERNET = %r\n' % (emb, steps, internet) + body)
json.dump(dict(id=ME + '/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script', is_private=True,
               enable_gpu=gpu, enable_internet=internet, kernel_sources=sources),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, 'gpu' if gpu else 'cpu', [s[0] for s in steps])
