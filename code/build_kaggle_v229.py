"""Kaggle kernel for v229 (Ouro Study 3 P1 robustness under the share swap; docs/prereg_v229_ouro_p1_robust.md).
astrocyte-v229-robust (CPU) on aurshalevmerin, which holds the v194 states. Body from build_kaggle_v215.py."""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__)); ME = 'aurshalevmerin'
files = ['v215_pilot.py', 'v215_items.py', 'v216_items.py', 'v221_items.py', 'v222_items.py', 'v224_items.py', 'v178_analyze.py',
         'v224_share.py', 'v228_rescore.py', 'v229_robust.py']
steps = [('v229_robust.py', ['/kaggle/input'], {})]
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode() for f in files}
body = open(os.path.join(HERE, 'build_kaggle_v215.py'), encoding='utf-8').read().split("body = r'''")[1].split("'''")[0]
body = body.replace('v215_stdout.txt', 'v229_stdout.txt').replace("'v215 failed", "'v229 failed")
kid = 'astrocyte-v229-robust'; dd = os.path.join(HERE, 'kaggle-kernel-v229-robust'); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\nSTEPS = %r\nINTERNET = %r\n' % (emb, steps, False) + body)
json.dump(dict(id=ME + '/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script', is_private=True, enable_gpu=False,
               enable_internet=False, kernel_sources=[ME + '/astrocyte-v194-ouro']),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
