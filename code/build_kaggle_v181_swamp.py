"""CPU-only Kaggle kernel for the v181 "swamped" measurement (post hoc). Build only.
  astrocyte-v181-swamp (second account): v181_swamp_check.py (overlap, structured LDA, learning curve) on the v181 states
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
BODY = r'''import subprocess, sys, os, base64, zlib
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    with open(OUT, 'w') as out:
        rc = subprocess.call([sys.executable, '-u', SCRIPT, '/kaggle/input'], stdout=out, stderr=subprocess.STDOUT,
                             env=dict(os.environ, PYTHONWARNINGS='ignore'))
    print('exit', rc); print(open(OUT).read()[-12000:])
'''
KERNELS = [
    ('aurmerin', 'astrocyte-v181-swamp', 'v181_swamp_check.py', 'v181_swamp_check.txt',
     ['v181_swamp_check.py', 'v178_analyze.py'], ['aurmerin/astrocyte-v181-ouro']),
]
for acct, kid, script, out, files, sources in KERNELS:
    emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode()
           for f in files}
    dd = os.path.join(HERE, 'kaggle-kernel-' + kid.replace('astrocyte-', '')); os.makedirs(dd, exist_ok=True)
    open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
        'FILES = %r\nSCRIPT = %r\nOUT = %r\n' % (emb, script, out) + BODY)
    json.dump(dict(id='%s/%s' % (acct, kid), title=kid, code_file='run.py', language='python', kernel_type='script',
                   is_private=True, enable_gpu=False, enable_internet=False, kernel_sources=sources),
              open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
    print('built', dd)
