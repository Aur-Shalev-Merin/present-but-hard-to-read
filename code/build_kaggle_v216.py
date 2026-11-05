"""Kaggle kernels for v216 (answer-only multi-step arithmetic). Build only; output streamed live to the console.
  V216_MODE=pilot     astrocyte-v216-pilot (T4 x2): v216_pilot.py, answer-only accuracy (Levels 3 and 2) (done)
  V216_MODE=smoke|x|a|null|patch   the answer-only study (docs/prereg_v216_answer_only.md), as v215's stages
  V216_MODE=yu-pilot  astrocyte-v216-yu-pilot (T4 x2): v216_yu.py gpu, Yu's task, accuracy and last-token states
  V216_MODE=yu-probe  astrocyte-v216-yu-probe (CPU): v216_yu.py probe, their probe on those states
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get('V216_MODE', 'pilot')
ME = 'aurshalevmerin'
if MODE == 'pilot':
    files = ['v216_pilot.py', 'v215_pilot.py']; steps = [('v216_pilot.py', [], {})]; gpu, internet, sources = True, True, []
elif MODE == 'smoke':                                     # v216 main: extraction 40 items, patch mechanics (SMOKE)
    files = ['v215_pilot.py', 'v215_items.py', 'v216_items.py', 'v216_extract.py', 'v216_patch.py']
    steps = [('v216_extract.py', [], {'V216_PILOT': '1'}), ('v216_patch.py', ['extract', 'SMOKE'], {'V216_SMOKE': '1'}),
             ('v216_patch.py', ['analyse', '.'], {})]; gpu, internet, sources = True, True, []
elif MODE == 'x':                                         # v216 main: extraction (~25 min)
    files = ['v215_pilot.py', 'v215_items.py', 'v216_items.py', 'v216_extract.py']; steps = [('v216_extract.py', [], {})]
    gpu, internet, sources = True, True, []
elif MODE == 'a':                                         # CPU: Part A, answer line, split, Part B plan
    files = ['v215_pilot.py', 'v215_items.py', 'v216_items.py', 'v215_analyze.py', 'v216_analyze.py', 'v178_analyze.py']
    steps = [('v216_analyze.py', ['/kaggle/input'], {})]; gpu, internet, sources = False, False, [ME + '/astrocyte-v216-x']
elif MODE == 'null':                                      # CPU: answer-mixing null for COMBINED positions
    files = ['v215_pilot.py', 'v215_items.py', 'v216_items.py', 'v215_analyze.py', 'v216_analyze.py', 'v178_analyze.py', 'v216_null.py']
    steps = [('v216_null.py', ['/kaggle/input'], {})]; gpu, internet, sources = False, False, [ME + '/astrocyte-v216-a']
elif MODE == 'patch':                                     # GPU: Part B, then its analysis
    files = ['v215_pilot.py', 'v215_items.py', 'v216_items.py', 'v216_patch.py']
    steps = [('v216_patch.py', ['extract', '/kaggle/input'], {}), ('v216_patch.py', ['analyse', '.'], {})]
    gpu, internet, sources = True, True, [ME + '/astrocyte-v216-a']
elif MODE == 'tpx':                                       # post hoc: replication-set states, implicit format (~25 min)
    files = ['v215_pilot.py', 'v215_items.py', 'v216_items.py', 'v216_extract.py', 'v216_theirprobe.py']
    steps = [('v216_theirprobe.py', ['gpu'], {})]; gpu, internet, sources = True, True, []
elif MODE == 'tp':                                        # post hoc, CPU: their standard probe on those states
    files = ['v215_pilot.py', 'v215_items.py', 'v216_items.py', 'v216_theirprobe.py']
    steps = [('v216_theirprobe.py', ['byop', '/kaggle/input'], {})]; gpu, internet, sources = False, False, [ME + '/astrocyte-v216-tpx']
    # version 1 ran ['probe', ...] (the full grid); version 2 runs the within-operation check (prereg section 10b)
elif MODE == 'yu-pilot':                                 # Yu (2411.15862) task, Qwen2.5-7B-Instruct: accuracy + states
    files = ['v216_yu.py']; steps = [('v216_yu.py', ['gpu'], {'V216_L': '3,5', 'V216_N': '2000'})]; gpu, internet, sources = True, True, []
elif MODE == 'yu-probe':                                  # CPU: their probe on the pilot's states
    files = ['v216_yu.py']; steps = [('v216_yu.py', ['probe', '/kaggle/input'], {})]
    gpu, internet, sources = False, False, [ME + '/astrocyte-v216-yu-pilot']
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode()
       for f in files}
body = open(os.path.join(HERE, 'build_kaggle_v215.py'), encoding='utf-8').read().split("body = r'''")[1].split("'''")[0]
body = body.replace('v215_stdout.txt', 'v216_stdout.txt').replace("'v215 failed", "'v216 failed")
kid = 'astrocyte-v216-' + MODE
dd = os.path.join(HERE, 'kaggle-kernel-v216-' + MODE); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
    'FILES = %r\nSTEPS = %r\nINTERNET = %r\n' % (emb, steps, internet) + body)
json.dump(dict(id=ME + '/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script', is_private=True,
               enable_gpu=gpu, enable_internet=internet, kernel_sources=sources),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, 'gpu' if gpu else 'cpu', [s[0] for s in steps])
