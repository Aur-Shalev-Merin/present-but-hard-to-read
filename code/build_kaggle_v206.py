"""Kaggle CPU kernels for v206 (docs/prereg_v206_ouro_template_curve.md). Build only.
  V206_MODE=pilot  astrocyte-v206-pilot : b2 and b3 on 8 prompts each (timing; cached vs full forward check)
  V206_MODE=b2sK   astrocyte-v206-b2sK  : b2 shard K of V206_NSHARD (env, default 4); V206_N_TRAIN caps the training pool
  V206_MODE=b3     astrocyte-v206-b3    : minimal-pair control
  V206_MODE=bcpu   astrocyte-v206-bcpu  : analysis (kernel_sources: the b2 shards and b3)
  V206_ACCOUNT     aurshalevmerin (default) or aurmerin
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get('V206_MODE', 'pilot'); ACCT = os.environ.get('V206_ACCOUNT', 'aurshalevmerin')
NSH = os.environ.get('V206_NSHARD', '4'); NTR = os.environ.get('V206_N_TRAIN'); NB3 = os.environ.get('V206_N_B3', '100')
EXTRACT = ['v206_extract.py', 'v205_extract.py', 'v202_extract.py', 'loop_grad_probe.py']
base = {'V206_NSHARD': NSH, 'V206_N_B3': NB3}
if NTR:
    base['V206_N_TRAIN'] = NTR
if MODE == 'pilot':
    files, steps, sources, internet = EXTRACT, [
        ('v206_pilot_b2.txt', ['v206_extract.py'], dict(base, V206_PART='b2', V206_SHARD='0', V206_PILOT='1')),
        ('v206_pilot_b3.txt', ['v206_extract.py'], dict(base, V206_PART='b3', V206_PILOT='1'))], [], True
elif MODE.startswith('b2s'):
    files, steps, sources, internet = EXTRACT, [
        ('v206_stdout.txt', ['v206_extract.py'], dict(base, V206_PART='b2', V206_SHARD=MODE[3:]))], [], True
elif MODE == 'b3':
    files, steps, sources, internet = EXTRACT, [('v206_stdout.txt', ['v206_extract.py'], dict(base, V206_PART='b3'))], [], True
elif MODE == 'selftest':
    files = ['v206_analyze.py', 'v205_analyze.py', 'v205_saved.py', 'v178_analyze.py', 'v181_swamp_check.py']
    steps, sources, internet = [('v206_selftest.txt', ['v206_analyze.py', '--selftest'], {})], [], False
else:
    files = ['v206_analyze.py', 'v205_analyze.py', 'v205_saved.py', 'v178_analyze.py', 'v181_swamp_check.py']
    steps = [('v206_analysis.txt', ['v206_analyze.py', '/kaggle/input'], {})]
    sources = ['%s/astrocyte-v206-b2s%d' % (ACCT, k) for k in range(int(NSH))] + ['%s/astrocyte-v206-b3' % ACCT]
    internet = False
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
kid = 'astrocyte-v206-' + MODE
dd = os.path.join(HERE, 'kaggle-kernel-v206-' + MODE); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
    'FILES = %r\nSTEPS = %r\nINTERNET = %r\n' % (emb, steps, internet) + body)
json.dump(dict(id='%s/%s' % (ACCT, kid), title=kid, code_file='run.py', language='python', kernel_type='script',
               is_private=True, enable_gpu=False, enable_internet=internet, kernel_sources=sources),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, ACCT)
