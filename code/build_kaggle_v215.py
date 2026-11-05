"""Kaggle kernels for v215 (Kudo et al. re-test; docs/prereg_v215_kudo_retest.md). Build only. Every step's output is
streamed to the console (live in the Kaggle UI) and to v215_stdout.txt.
  V215_MODE=pilot        astrocyte-v215-pilot (T4 x2): v215_pilot.py, states only (done)
  V215_MODE=pilot-probe  astrocyte-v215-pilot-probe (CPU): v215_pilot_probe.py (done)
  V215_MODE=smoke        astrocyte-v215-smoke (T4 x2): extraction of 40 items per part, then patch mechanics (SMOKE)
  V215_MODE=x            astrocyte-v215-x (T4 x2): extraction, fam, fams (short-chain arm), rep (~2 h)
  V215_MODE=a            astrocyte-v215-a (CPU): Part A (both arms), Part C, Part B plan from the short arm (source: x)
  V215_MODE=null         astrocyte-v215-null (CPU): mixing null for COMBINED positions (source: a)
  V215_MODE=patch        astrocyte-v215-patch (T4 x2): Part B (sources: a), then its analysis
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get('V215_MODE', 'smoke')
ME = 'aurshalevmerin'
CORE = ['v215_pilot.py', 'v215_items.py']
gpu, internet, sources = True, True, []
if MODE == 'pilot':
    files, steps = ['v215_pilot.py'], [('v215_pilot.py', [], {'V215_NGEN': '0'})]
elif MODE == 'pilot-probe':
    files, steps = ['v215_pilot_probe.py'], [('v215_pilot_probe.py', ['/kaggle/input'], {'V215_SPLIT': os.environ.get('V215_SPLIT', 'key')})]
    gpu, internet, sources = False, False, [ME + '/astrocyte-v215-pilot']
elif MODE == 'smoke':
    files = CORE + ['v215_extract.py', 'v215_patch.py']
    steps = [('v215_extract.py', [], {'V215_PART': p, 'V215_PILOT': '1'}) for p in ('fam', 'fams', 'rep')] + [
             ('v215_patch.py', ['extract', 'SMOKE'], {'V215_SMOKE': '1'}), ('v215_patch.py', ['analyse', '.'], {})]
elif MODE == 'x':
    files = CORE + ['v215_extract.py']
    steps = [('v215_extract.py', [], {'V215_PART': p}) for p in ('fam', 'fams', 'rep')]
elif MODE == 'a':
    files = CORE + ['v215_analyze.py', 'v178_analyze.py']; steps = [('v215_analyze.py', ['/kaggle/input', 'A', 'S', 'C'], {})]
    gpu, internet, sources = False, False, [ME + '/astrocyte-v215-x']
elif MODE == 'null':
    files = CORE + ['v215_analyze.py', 'v178_analyze.py', 'v215_null.py']; steps = [('v215_null.py', ['/kaggle/input'], {})]
    gpu, internet, sources = False, False, [ME + '/astrocyte-v215-a']
elif MODE == 'patch':
    files = CORE + ['v215_patch.py']
    steps = [('v215_patch.py', ['extract', '/kaggle/input'], {}), ('v215_patch.py', ['analyse', '.'], {})]
    sources = [ME + '/astrocyte-v215-a']
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode()
       for f in files}
body = r'''import subprocess, sys, os, base64, zlib, hashlib
if __name__ == '__main__':
    W = '/kaggle/working'; os.chdir(W)
    for fn, e in FILES.items():
        b = zlib.decompress(base64.b64decode(e)); open(fn, 'wb').write(b); print(fn, hashlib.sha256(b).hexdigest()[:16], flush=True)
    if INTERNET:
        subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.54.1', 'accelerate'])
    rc = 0
    with open('v215_stdout.txt', 'w') as fh:                 # stream to the console (live in the Kaggle UI) and a file
        for script, args, extra in STEPS:
            env = dict(os.environ, **extra, OUT_DIR=W, PYTHONUNBUFFERED='1', PYTHONWARNINGS='ignore', HF_HUB_DISABLE_PROGRESS_BARS='1')
            if not INTERNET:
                env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            print('=== %s %s %s' % (script, args, extra), flush=True); fh.write('=== %s %s %s\n' % (script, args, extra))
            pr = subprocess.Popen([sys.executable, '-u', script] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  env=env, text=True, bufsize=1)
            for line in pr.stdout:
                print(line, end='', flush=True); fh.write(line); fh.flush()
            rc = pr.wait()
            if rc:
                break
    for fn in FILES:
        os.remove(fn)
    if rc:
        raise SystemExit('v215 failed: %s' % rc)
'''
kid = 'astrocyte-v215-' + MODE
dd = os.path.join(HERE, 'kaggle-kernel-v215-' + MODE); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
    'FILES = %r\nSTEPS = %r\nINTERNET = %r\n' % (emb, steps, internet) + body)
json.dump(dict(id=ME + '/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script', is_private=True,
               enable_gpu=gpu, enable_internet=internet, kernel_sources=sources),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, 'gpu' if gpu else 'cpu', [s[0] for s in steps])
