"""CPU-only Kaggle kernels for the normalization check (post hoc), one per study. Build only.
  astrocyte-v188-rms188: v188_rms_check.py on the v188 states;  astrocyte-v188-rms186: on the v186 states.
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['v188_rms_check.py', 'v188_shown_digit_check.py', 'v188_analyze.py', 'v188_calibrate.py',
         'v186_analyze.py', 'v178_analyze.py', 'v188_items.json', 'v186_items.json']
BODY = r'''import subprocess, sys, os, base64, zlib
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    with open(OUT, 'w') as out:
        rc = subprocess.call([sys.executable, '-u', 'v188_rms_check.py', '/kaggle/input', STUDY], stdout=out,
                             stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONWARNINGS='ignore'))
    for fn in FILES:
        os.remove(fn)
    print('exit', rc); print(open(OUT).read()[-12000:])
'''
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode()
       for f in FILES}
for v in ('188', '186'):
    kid = 'astrocyte-v188-rms' + v
    dd = os.path.join(HERE, 'kaggle-kernel-v188-rms' + v); os.makedirs(dd, exist_ok=True)
    open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
        'FILES = %r\nSTUDY = %r\nOUT = %r\n' % (emb, v, 'v188_rms_check_%s.txt' % v) + BODY)
    json.dump(dict(id='aurshalevmerin/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script',
                   is_private=True, enable_gpu=False, enable_internet=False,
                   kernel_sources=['aurshalevmerin/astrocyte-v%s-ouro' % v]),
              open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
    print('built', dd)
