"""CPU-only Kaggle kernel: frozen v186_analyze.py on the main-run states (kernel_sources = astrocyte-v186-ouro). Build only."""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['v186_analyze.py', 'v178_analyze.py']
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode()
       for f in FILES}
body = r'''import subprocess, sys, os, base64, zlib
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    with open('v186_analysis.txt', 'w') as out:
        rc = subprocess.call([sys.executable, '-u', 'v186_analyze.py', '/kaggle/input'], stdout=out, stderr=subprocess.STDOUT,
                             env=dict(os.environ, PYTHONWARNINGS='ignore'))
    print('exit', rc); print(open('v186_analysis.txt').read()[-6000:])
'''
dd = os.path.join(HERE, 'kaggle-kernel-v186-cpu'); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\n' % emb + body)
json.dump(dict(id='aurshalevmerin/astrocyte-v186-cpu', title='astrocyte-v186-cpu', code_file='run.py',
               language='python', kernel_type='script', is_private=True, enable_gpu=False, enable_internet=False,
               kernel_sources=['aurshalevmerin/astrocyte-v186-ouro']),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
