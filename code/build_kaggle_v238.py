"""Kaggle kernel for v238 (Kudo no-chain relay with different-answer pairs; docs/prereg_v238_kudo_relay_diff.md). Build only.
  V238_MODE=relay   astrocyte-v238-relay (T4 x2): v238_relay.py run, the registered test and its analysis
Account: V238_ME (default aurmerin).
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__)); MODE = os.environ.get('V238_MODE', 'relay'); ME = os.environ.get('V238_ME', 'aurmerin')
BASE = ['v215_pilot.py', 'v215_items.py', 'v216_items.py', 'v234_items.py']
assert MODE == 'relay'
files = BASE + ['v236_pilot.py', 'v236_relay.py', 'v238_relay.py']; steps = [('v238_relay.py', ['run'], {})]
gpu, internet, sources = True, True, []
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode()
       for f in files}
body = open(os.path.join(HERE, 'build_kaggle_v215.py'), encoding='utf-8').read().split("body = r'''")[1].split("'''")[0]
body = body.replace('v215_stdout.txt', 'v238_stdout.txt').replace("'v215 failed", "'v238 failed")
kid = 'astrocyte-v238-' + MODE
dd = os.path.join(HERE, 'kaggle-kernel-v238-' + MODE); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
    'FILES = %r\nSTEPS = %r\nINTERNET = %r\n' % (emb, steps, internet) + body)
json.dump(dict(id=ME + '/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script', is_private=True,
               enable_gpu=gpu, enable_internet=internet, kernel_sources=sources),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, 'gpu' if gpu else 'cpu', [s[0] for s in steps])
