"""Kaggle kernels for v221 (positive control in Kudo et al.'s Simple chain format; docs/prereg_v221_positive_control.md).
Build only; output streamed live to the console (body from build_kaggle_v215.py).
  V221_MODE=smoke   astrocyte-v221-smoke (T4 x2): extraction of 40 items (position check), patch mechanics (SMOKE)
  V221_MODE=x       astrocyte-v221-x (T4 x2): extraction, 12,000 items x eq / last / letb / eqb / inb (~30 min)
  V221_MODE=a       astrocyte-v221-a (CPU): Part A and the Part B plan
  V221_MODE=null    astrocyte-v221-null (CPU): the mixing null for COMBINED positions
  V221_MODE=patch   astrocyte-v221-patch (T4 x2): Part B, then its analysis
  V221_MODE=copy    astrocyte-v221-copy (CPU): their standard probe on the copied operands, v215's replication states
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get('V221_MODE', 'smoke')
ME = 'aurshalevmerin'
CORE = ['v215_pilot.py', 'v215_items.py', 'v221_items.py']
if MODE == 'smoke':
    files = CORE + ['v221_extract.py', 'v215_patch.py', 'v221_patch.py']
    steps = [('v221_extract.py', [], {'V221_PILOT': '1'}), ('v221_patch.py', ['extract', 'SMOKE'], {'V215_SMOKE': '1'}),
             ('v221_patch.py', ['analyse', '.'], {})]; gpu, internet, sources = True, True, []
elif MODE == 'x':
    files = CORE + ['v221_extract.py']; steps = [('v221_extract.py', [], {})]; gpu, internet, sources = True, True, []
elif MODE == 'a':
    files = CORE + ['v215_analyze.py', 'v221_analyze.py', 'v178_analyze.py']
    steps = [('v221_analyze.py', ['/kaggle/input'], {})]; gpu, internet, sources = False, False, [ME + '/astrocyte-v221-x']
elif MODE == 'null':
    files = CORE + ['v215_analyze.py', 'v221_analyze.py', 'v178_analyze.py', 'v215_null.py']
    steps = [('v221_analyze.py', ['null', '/kaggle/input'], {})]; gpu, internet, sources = False, False, [ME + '/astrocyte-v221-a']
elif MODE == 'patch':
    files = CORE + ['v215_patch.py', 'v221_patch.py']
    steps = [('v221_patch.py', ['extract', '/kaggle/input'], {}), ('v221_patch.py', ['analyse', '.'], {})]
    gpu, internet, sources = True, True, [ME + '/astrocyte-v221-a']
elif MODE == 'copy':
    files = ['v215_pilot.py', 'v215_items.py', 'v221_copyprobe.py']
    steps = [('v221_copyprobe.py', ['/kaggle/input'], {})]; gpu, internet, sources = False, False, [ME + '/astrocyte-v215-x']
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode()
       for f in files}
body = open(os.path.join(HERE, 'build_kaggle_v215.py'), encoding='utf-8').read().split("body = r'''")[1].split("'''")[0]
body = body.replace('v215_stdout.txt', 'v221_stdout.txt').replace("'v215 failed", "'v221 failed")
kid = 'astrocyte-v221-' + MODE
dd = os.path.join(HERE, 'kaggle-kernel-v221-' + MODE); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
    'FILES = %r\nSTEPS = %r\nINTERNET = %r\n' % (emb, steps, internet) + body)
json.dump(dict(id=ME + '/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script', is_private=True,
               enable_gpu=gpu, enable_internet=internet, kernel_sources=sources),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, 'gpu' if gpu else 'cpu', [s[0] for s in steps])
