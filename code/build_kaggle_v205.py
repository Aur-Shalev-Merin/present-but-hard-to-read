"""Kaggle kernels for v205 (docs/prereg_v205_swamping_robustness.md). Build only; all CPU.
  V205_MODE=pilot  astrocyte-v205-pilot  : real-Qwen pilot of both extraction parts (b1 3 families, b2 40 questions) and
                                           the two selftests
  V205_MODE=amain  astrocyte-v205-amain  : Part A on the main account's saved states (Qwen v202, Ouro v185, Study 3 v194)
  V205_MODE=aaur   astrocyte-v205-aaur   : Part A on the second account's saved states (Ouro v181), account aurmerin
  V205_MODE=b1     astrocyte-v205-b1     : Part B extraction, random-perturbation null
  V205_MODE=b2s0/b2s1/b2s2 astrocyte-v205-b2sK : Part B extraction, template-count curve, shard K of 3
  V205_MODE=b3     astrocyte-v205-b3     : Part B extraction, minimal-pair positive control for B2
  V205_MODE=bcpu   astrocyte-v205-bcpu   : Part B analysis (kernel_sources b1, b2s0-2, b3 and v202)
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get('V205_MODE', 'pilot')
ME, AUR = 'aurshalevmerin', 'aurmerin'
COMMON = ['v178_analyze.py', 'v181_swamp_check.py', 'v178_labels.json']
EXTRACT = ['v205_extract.py', 'v202_extract.py', 'v178_labels.json']
K = {
    'pilot': (ME, EXTRACT + ['v205_saved.py', 'v205_analyze.py', 'v178_analyze.py', 'v181_swamp_check.py'],
              [('v205_pilot_b1.txt', ['v205_extract.py'], {'V205_PART': 'b1', 'V205_PILOT': '1'}),
               ('v205_pilot_b2.txt', ['v205_extract.py'], {'V205_PART': 'b2', 'V205_SHARD': '2', 'V205_PILOT': '1'}),
               ('v205_pilot_b3.txt', ['v205_extract.py'], {'V205_PART': 'b3', 'V205_PILOT': '1'}),
               ('v205_selftest_saved.txt', ['v205_saved.py', 'selftest'], {}),
               ('v205_selftest_analyze.txt', ['v205_analyze.py', '--selftest'], {})], [], True),
    'selftest': (ME, ['v205_saved.py', 'v205_analyze.py', 'v178_analyze.py', 'v181_swamp_check.py'],
                 [('v205_selftest_saved.txt', ['v205_saved.py', 'selftest'], {}),
                  ('v205_selftest_analyze.txt', ['v205_analyze.py', '--selftest'], {})], [], False),
    'amain': (ME, COMMON + ['v205_saved.py', 'v194_analyze.py', 'v186_analyze.py'],
              [('v205_A_qwen202.txt', ['v205_saved.py', 'qwen202', '/kaggle/input'], {}),
               ('v205_A_ouro185.txt', ['v205_saved.py', 'ouro185', '/kaggle/input'], {}),
               ('v205_A_study3.txt', ['v205_saved.py', 'study3', '/kaggle/input'], {})],
              [ME + '/astrocyte-v202-qwen', ME + '/astrocyte-v185-ouro', ME + '/astrocyte-v194-ouro'], False),
    'aaur': (AUR, COMMON + ['v205_saved.py'], [('v205_A_ouro181.txt', ['v205_saved.py', 'ouro181', '/kaggle/input'], {})],
             [AUR + '/astrocyte-v181-ouro'], False),
    'b1': (ME, EXTRACT, [('v205_b1_stdout.txt', ['v205_extract.py'], {'V205_PART': 'b1'})], [], True),
    'b2s0': (ME, EXTRACT, [('v205_b2_stdout.txt', ['v205_extract.py'], {'V205_PART': 'b2', 'V205_SHARD': '0'})], [], True),
    'b2s1': (ME, EXTRACT, [('v205_b2_stdout.txt', ['v205_extract.py'], {'V205_PART': 'b2', 'V205_SHARD': '1'})], [], True),
    'b2s2': (ME, EXTRACT, [('v205_b2_stdout.txt', ['v205_extract.py'], {'V205_PART': 'b2', 'V205_SHARD': '2'})], [], True),
    'b3': (ME, EXTRACT, [('v205_b3_stdout.txt', ['v205_extract.py'], {'V205_PART': 'b3'})], [], True),
    'bcpu': (ME, COMMON + ['v205_analyze.py', 'v205_saved.py'], [('v205_B_analysis.txt', ['v205_analyze.py', '/kaggle/input'], {})],
             [ME + '/astrocyte-v205-' + k for k in ('b1', 'b2s0', 'b2s1', 'b2s2', 'b3')] + [ME + '/astrocyte-v202-qwen'], False),
}
acct, files, steps, sources, internet = K[MODE]
files = list(dict.fromkeys(files))
emb = {}
for f in files:
    raw = open(os.path.join(HERE, f), 'rb').read()
    emb[f] = base64.b64encode(zlib.compress(raw.replace(b'\r\n', b'\n') if f.endswith('.py') else raw, 9)).decode()
body = r'''import subprocess, sys, os, base64, zlib, hashlib
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        b = zlib.decompress(base64.b64decode(e)); open(fn, 'wb').write(b); print(fn, hashlib.sha256(b).hexdigest()[:16], flush=True)
    if INTERNET:
        subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.54.1', 'datasets'])
    for out, args, env in STEPS:
        with open(out, 'w') as fh:
            rc = subprocess.call([sys.executable, '-u'] + args, stdout=fh, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, PYTHONWARNINGS='ignore', OUT_DIR='/kaggle/working', **env))
        print(out, 'exit', rc); print(open(out).read()[-6000:], flush=True)
    for fn in FILES:
        os.remove(fn)
'''
kid = 'astrocyte-v205-' + MODE
dd = os.path.join(HERE, 'kaggle-kernel-v205-' + MODE); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
    'FILES = %r\nSTEPS = %r\nINTERNET = %r\n' % (emb, steps, internet) + body)
json.dump(dict(id='%s/%s' % (acct, kid), title=kid, code_file='run.py', language='python', kernel_type='script',
               is_private=True, enable_gpu=False, enable_internet=internet, kernel_sources=sources),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, acct)
