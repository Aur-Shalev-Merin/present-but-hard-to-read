"""Kaggle kernels for v236 (Kudo no-chain: pairs that share d2 and differ in d1; docs/v236_design.md). Build only.
  V236_MODE=pilot   astrocyte-v236-pilot (T4 x2): v236_pilot.py, whole-state patch map over positions and layers
                    (D1 and D2 pairs; no learned subspace)                                              [pre-design]
  V236_MODE=relay   astrocyte-v236-relay (T4 x2): v236_relay.py run, the registered relay test and its analysis
                    (docs/prereg_v236_kudo_relay.md)
Account: V236_ME (default aurmerin, where the whole study runs).
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__)); MODE = os.environ.get('V236_MODE', 'pilot'); ME = os.environ.get('V236_ME', 'aurmerin')
BASE = ['v215_pilot.py', 'v215_items.py', 'v216_items.py', 'v234_items.py']
if MODE == 'pilot':
    files = BASE + ['v236_pilot.py']; steps = [('v236_pilot.py', [], {})]; gpu, internet, sources = True, True, []
elif MODE == 'relay':
    files = BASE + ['v236_pilot.py', 'v236_relay.py']; steps = [('v236_relay.py', ['run'], {})]
    gpu, internet, sources = True, True, []
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode()
       for f in files}
body = open(os.path.join(HERE, 'build_kaggle_v215.py'), encoding='utf-8').read().split("body = r'''")[1].split("'''")[0]
body = body.replace('v215_stdout.txt', 'v236_stdout.txt').replace("'v215 failed", "'v236 failed")
kid = 'astrocyte-v236-' + MODE
dd = os.path.join(HERE, 'kaggle-kernel-v236-' + MODE); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
    'FILES = %r\nSTEPS = %r\nINTERNET = %r\n' % (emb, steps, internet) + body)
json.dump(dict(id=ME + '/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script', is_private=True,
               enable_gpu=gpu, enable_internet=internet, kernel_sources=sources),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, 'gpu' if gpu else 'cpu', [s[0] for s in steps])
