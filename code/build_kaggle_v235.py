"""Kaggle kernels for v235 (pretrained positive control: a two-hop bridge; docs/v235_design.md). Build only.
  V235_MODE=pilot   astrocyte-v235-pilot (T4 x2): v235_pilot.py, what Qwen2.5-7B knows about each hop (pre-design)
  V235_MODE=pilot2  astrocyte-v235-pilot2 (T4 x2): v235_pilot2.py, ground-truth whole-state patches, random-clamp mechanics,
                    states for readout checks (no learned subspace)
  V235_MODE=smoke   astrocyte-v235-smoke (T4 x2): v235_clamp.py with V235_SMOKE=1 (random subspaces only)   [pre-freeze]
  V235_MODE=clamp   astrocyte-v235-clamp (T4 x2): v235_clamp.py, the frozen run and its analysis
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__)); MODE = os.environ.get('V235_MODE', 'pilot'); ME = os.environ.get('V235_ME', 'aurshalevmerin')
if MODE == 'pilot':
    files = ['v235_pilot.py', 'v235_cities.json', 'v235_persons.json']; steps = [('v235_pilot.py', [], {})]
    gpu, internet, sources = True, True, []
elif MODE == 'pilot2':
    files = ['v235_pilot.py', 'v235_pilot2.py', 'v235_cities.json', 'v235_pilot2_items.json']; steps = [('v235_pilot2.py', [], {})]
    gpu, internet, sources = True, True, []
elif MODE in ('smoke', 'clamp'):
    files = ['v235_pilot.py', 'v235_clamp.py', 'v235_cities.json', 'v235_pilot2_items.json']
    env = {'V235_SMOKE': '1'} if MODE == 'smoke' else {}
    steps = [('v235_clamp.py', ['run'], env), ('v235_clamp.py', ['analyse', '.'], {})]; gpu, internet, sources = True, True, []
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode()
       for f in files}
body = open(os.path.join(HERE, 'build_kaggle_v215.py'), encoding='utf-8').read().split("body = r'''")[1].split("'''")[0]
body = body.replace('v215_stdout.txt', 'v235_stdout.txt').replace("'v215 failed", "'v235 failed")
kid = 'astrocyte-v235-' + MODE
dd = os.path.join(HERE, 'kaggle-kernel-v235-' + MODE); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
    'FILES = %r\nSTEPS = %r\nINTERNET = %r\n' % (emb, steps, internet) + body)
json.dump(dict(id=ME + '/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script', is_private=True,
               enable_gpu=gpu, enable_internet=internet, kernel_sources=sources),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, 'gpu' if gpu else 'cpu', [s[0] for s in steps])
