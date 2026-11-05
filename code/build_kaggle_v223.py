"""Kaggle kernel for v223 (exploratory swap diagnosis on v221's states). astrocyte-v223-diag (CPU), on the account
holding astrocyte-v221-x (aurmerin). Body from build_kaggle_v215.py."""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__)); ME = 'aurmerin'
files = ['v215_pilot.py', 'v215_items.py', 'v221_items.py', 'v178_analyze.py', 'v223_swapdiag.py']
steps = [('v223_swapdiag.py', ['/kaggle/input'], {})]
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode() for f in files}
body = open(os.path.join(HERE, 'build_kaggle_v215.py'), encoding='utf-8').read().split("body = r'''")[1].split("'''")[0]
body = body.replace('v215_stdout.txt', 'v223_stdout.txt').replace("'v215 failed", "'v223 failed")
kid = 'astrocyte-v223-diag'; dd = os.path.join(HERE, 'kaggle-kernel-v223-diag'); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\nSTEPS = %r\nINTERNET = %r\n' % (emb, steps, False) + body)
json.dump(dict(id=ME + '/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script', is_private=True,
               enable_gpu=False, enable_internet=False, kernel_sources=[ME + '/astrocyte-v221-x']),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
