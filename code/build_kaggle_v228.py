"""Kaggle kernel for v228 (re-score with the validated share swap; docs/prereg_v228_rescore.md). astrocyte-v228-rescore
(CPU) on aurshalevmerin, which holds the target states. Body from build_kaggle_v215.py."""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__)); ME = 'aurshalevmerin'
files = ['v215_pilot.py', 'v215_items.py', 'v216_items.py', 'v221_items.py', 'v222_items.py', 'v224_items.py', 'v178_analyze.py',
         'v224_share.py', 'v228_rescore.py']
steps = [('v228_rescore.py', ['/kaggle/input'], {})]
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode() for f in files}
body = open(os.path.join(HERE, 'build_kaggle_v215.py'), encoding='utf-8').read().split("body = r'''")[1].split("'''")[0]
body = body.replace('v215_stdout.txt', 'v228_stdout.txt').replace("'v215 failed", "'v228 failed")
kid = 'astrocyte-v228-rescore'; dd = os.path.join(HERE, 'kaggle-kernel-v228-rescore'); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\nSTEPS = %r\nINTERNET = %r\n' % (emb, steps, False) + body)
json.dump(dict(id=ME + '/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script', is_private=True, enable_gpu=False,
               enable_internet=False, kernel_sources=[ME + '/astrocyte-v215-x', ME + '/astrocyte-v216-x', ME + '/astrocyte-v194-ouro']),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
