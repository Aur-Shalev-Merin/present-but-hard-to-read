"""CPU-only Kaggle kernel: run v188_analyze.py --selftest (pre-freeze). Build only."""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['v188_analyze.py', 'v186_analyze.py', 'v178_analyze.py', 'v188_items.json']
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode()
       for f in FILES}
body = r'''import subprocess, sys, os, base64, zlib
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    with open('v188_selftest.txt', 'w') as out:
        rc = subprocess.call([sys.executable, '-u', 'v188_analyze.py', '--selftest'], stdout=out, stderr=subprocess.STDOUT,
                             env=dict(os.environ, PYTHONWARNINGS='ignore'))
    print('exit', rc); print(open('v188_selftest.txt').read()[-3000:])
'''
dd = os.path.join(HERE, 'kaggle-kernel-v188-selftest'); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\n' % emb + body)
json.dump(dict(id='aurshalevmerin/astrocyte-v188-selftest', title='astrocyte-v188-selftest', code_file='run.py',
               language='python', kernel_type='script', is_private=True, enable_gpu=False, enable_internet=False),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
