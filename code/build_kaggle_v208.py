"""Kaggle CPU kernels for v208 (docs/prereg_v208_input_sensitivity.md). Build only.
  V208_MODE=pilot   astrocyte-v208-pilot   : Qwen and Ouro pilots (2 prompts each: cache check, step check, timing; no
                                             responses saved) + the Part A and Part B selftests
  V208_MODE=qwenK   astrocyte-v208-qwenK   : Part A extraction, Qwen, shard K of V208_NQ (default 2)
  V208_MODE=ouroK   astrocyte-v208-ouroK   : Part A extraction, Ouro, shard K of V208_NO (default 4)
  V208_MODE=acpu    astrocyte-v208-acpu    : Part A analysis (kernel_sources: every extraction kernel)
  V208_MODE=bqwen   astrocyte-v208-bqwen   : Part B, Qwen (kernel_sources: Qwen extraction + v205 B2/B3)
  V208_MODE=bouro   astrocyte-v208-bouro   : Part B, Ouro (kernel_sources: Ouro extraction + v206 B2/B3)
  V208_ACCOUNT      aurshalevmerin (default; owns the v205/v206 kernels Part B reads) or aurmerin
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get('V208_MODE', 'pilot'); ACCT = os.environ.get('V208_ACCOUNT', 'aurshalevmerin')
NQ, NO = int(os.environ.get('V208_NQ', '2')), int(os.environ.get('V208_NO', '4'))
EXTRACT = ['v208_extract.py', 'v205_extract.py', 'v202_extract.py', 'loop_grad_probe.py', 'v178_labels.json']
ANALYZE = ['v208_analyze.py', 'v208_method.py', 'v205_analyze.py', 'v178_analyze.py', 'v181_swamp_check.py']
ext = lambda m: ['%s/astrocyte-v208-%s%d' % (ACCT, m, k) for k in range({'qwen': NQ, 'ouro': NO}[m])]
if MODE == 'pilot':
    files, internet, sources = EXTRACT + ANALYZE, True, []
    steps = [('v208_pilot_qwen.txt', ['v208_extract.py'], dict(V208_MODEL='qwen', V208_PILOT='1')),
             ('v208_pilot_ouro.txt', ['v208_extract.py'], dict(V208_MODEL='ouro', V208_PILOT='1')),
             ('v208_selftest_a.txt', ['v208_analyze.py', '--selftest'], {}),
             ('v208_selftest_b.txt', ['v208_method.py', '--selftest'], {})]
elif MODE.startswith('qwen') or MODE.startswith('ouro'):
    m, k = MODE[:4], MODE[4:]
    files, internet, sources = EXTRACT, True, []
    steps = [('v208_stdout.txt', ['v208_extract.py'], dict(V208_MODEL=m, V208_SHARD=k, V208_NSHARD=str({'qwen': NQ, 'ouro': NO}[m])))]
elif MODE == 'acpu':
    files, internet, sources = ANALYZE, False, ext('qwen') + ext('ouro')
    steps = [('v208_selftest_a.txt', ['v208_analyze.py', '--selftest'], {}),
             ('v208_partA.txt', ['v208_analyze.py', '/kaggle/input'], {})]
elif MODE == 'bqwen':
    files, internet = ANALYZE, False
    sources = ext('qwen') + ['%s/astrocyte-v205-b2s%d' % (ACCT, k) for k in range(3)] + ['%s/astrocyte-v205-b3' % ACCT]
    steps = [('v208_selftest_b.txt', ['v208_method.py', '--selftest'], {}),
             ('v208_partB_qwen.txt', ['v208_method.py', 'qwen', '/kaggle/input'], {})]
elif MODE == 'bouro':
    files, internet = ANALYZE, False
    sources = ext('ouro') + ['%s/astrocyte-v206-b2s%d' % (ACCT, k) for k in range(4)] + ['%s/astrocyte-v206-b3' % ACCT]
    steps = [('v208_selftest_b.txt', ['v208_method.py', '--selftest'], {}),
             ('v208_partB_ouro.txt', ['v208_method.py', 'ouro', '/kaggle/input'], {})]
else:
    raise SystemExit('unknown mode ' + MODE)
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
kid = 'astrocyte-v208-' + MODE
dd = os.path.join(HERE, 'kaggle-kernel-v208-' + MODE); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
    'FILES = %r\nSTEPS = %r\nINTERNET = %r\n' % (emb, steps, internet) + body)
json.dump(dict(id='%s/%s' % (ACCT, kid), title=kid, code_file='run.py', language='python', kernel_type='script',
               is_private=True, enable_gpu=False, enable_internet=internet, kernel_sources=sources),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, ACCT)
