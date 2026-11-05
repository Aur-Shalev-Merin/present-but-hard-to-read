"""Kaggle kernel for v214 (docs/prereg_v214_one_ref.md). Build only.
  V214_MODE=main  astrocyte-v214-main (CPU): v214_one_ref.py on the saved states of v205-b3, v206-b3, v202-qwen and
                  v213-x-qwen7b / v213-x-ouro26
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get('V214_MODE', 'main')
ME = 'aurshalevmerin'
files = ['v214_one_ref.py', 'v178_analyze.py']
sources = [ME + '/astrocyte-' + k for k in ('v205-b3', 'v206-b3', 'v202-qwen', 'v213-x-qwen7b', 'v213-x-ouro26')]
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode()
       for f in files}
body = r'''import subprocess, sys, os, base64, zlib, hashlib
if __name__ == '__main__':
    W = '/kaggle/working'; os.chdir(W)
    for fn, e in FILES.items():
        b = zlib.decompress(base64.b64decode(e)); open(fn, 'wb').write(b); print(fn, hashlib.sha256(b).hexdigest()[:16], flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONWARNINGS='ignore', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1')
    with open('v214_stdout.txt', 'w') as fh:
        rc = subprocess.call([sys.executable, '-u', 'v214_one_ref.py', '/kaggle/input'], stdout=fh, stderr=subprocess.STDOUT, env=env)
    print(open('v214_stdout.txt').read()[-8000:], flush=True)
    for fn in FILES:
        os.remove(fn)
    if rc:
        raise SystemExit('v214 failed: %s' % rc)
'''
kid = 'astrocyte-v214-' + MODE
dd = os.path.join(HERE, 'kaggle-kernel-v214-' + MODE); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\n' % emb + body)
json.dump(dict(id=ME + '/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script', is_private=True,
               enable_gpu=False, enable_internet=False, kernel_sources=sources),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
