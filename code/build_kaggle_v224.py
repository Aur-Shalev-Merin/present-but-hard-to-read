"""Kaggle kernels for v224 (share-based operation swap; docs/prereg_v224_share_swap.md). Build only; output streamed
live to the console (body from build_kaggle_v215.py). Account: V224_ACCOUNT (default aurmerin, which holds
astrocyte-v221-x for the dev mode).
  V224_MODE=smoke  astrocyte-v224-smoke (T4 x2): 200 items each, Qwen2.5-1.5B b set and Qwen2.5-7B a set (tokens, accuracy)
  V224_MODE=dev    astrocyte-v224-dev (CPU): the statistic on v221's states (development, already seen)
  V224_MODE=xb     astrocyte-v224-xb (T4 x2): Qwen2.5-1.5B, b set, 12,000 items
  V224_MODE=xa     astrocyte-v224-xa (T4 x2): Qwen2.5-7B, a set, 12,000 items
  V224_MODE=main   astrocyte-v224-main (CPU): the frozen analysis of both fresh controls and the fresh null seeds
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get('V224_MODE', 'smoke'); ME = os.environ.get('V224_ACCOUNT', 'aurmerin')
ITEMS = ['v215_pilot.py', 'v215_items.py', 'v221_items.py', 'v222_items.py', 'v224_items.py']
ANALYSIS = ITEMS + ['v178_analyze.py', 'v224_share.py']
if MODE == 'smoke':
    files = ITEMS + ['v224_extract.py']
    steps = [('v224_extract.py', [], {'V224_SET': 'b', 'V224_MODEL': '1.5b', 'V224_N': '200'}),
             ('v224_extract.py', [], {'V224_SET': 'a', 'V224_MODEL': '7b', 'V224_N': '200'})]; gpu, internet, sources = True, True, []
elif MODE == 'dev':
    files = ANALYSIS; steps = [('v224_share.py', ['dev', '/kaggle/input'], {})]; gpu, internet, sources = False, False, [ME + '/astrocyte-v221-x']
elif MODE == 'xb':
    files = ITEMS + ['v224_extract.py']; steps = [('v224_extract.py', [], {'V224_SET': 'b', 'V224_MODEL': '1.5b'})]; gpu, internet, sources = True, True, []
elif MODE == 'xa':
    files = ITEMS + ['v224_extract.py']; steps = [('v224_extract.py', [], {'V224_SET': 'a', 'V224_MODEL': '7b'})]; gpu, internet, sources = True, True, []
elif MODE == 'main':
    files = ANALYSIS; steps = [('v224_share.py', ['main', '/kaggle/input'], {})]
    gpu, internet, sources = False, False, [ME + '/astrocyte-v224-xb', ME + '/astrocyte-v224-xa']
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode() for f in files}
body = open(os.path.join(HERE, 'build_kaggle_v215.py'), encoding='utf-8').read().split("body = r'''")[1].split("'''")[0]
body = body.replace('v215_stdout.txt', 'v224_stdout.txt').replace("'v215 failed", "'v224 failed")
kid = 'astrocyte-v224-' + MODE
dd = os.path.join(HERE, 'kaggle-kernel-v224-' + MODE); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\nSTEPS = %r\nINTERNET = %r\n' % (emb, steps, internet) + body)
json.dump(dict(id=ME + '/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script', is_private=True,
               enable_gpu=gpu, enable_internet=internet, kernel_sources=sources),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, 'gpu' if gpu else 'cpu', [s[0] for s in steps])
