"""CPU-only Kaggle kernels for the swamping mechanism check (post hoc; rule in docs/v181_result.md). Build only.
  astrocyte-v185-mech (main account):   v185_swamp_mechanism.py v185 (M1, M2) and v194 (M3) on the saved states
  astrocyte-v181-mech (second account): v185_swamp_mechanism.py v181 (M2 on the v181 states)
"""
import base64, hashlib, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
BODY = r'''import subprocess, sys, os, base64, zlib, hashlib
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        b = zlib.decompress(base64.b64decode(e)); open(fn, 'wb').write(b); print(fn, hashlib.sha256(b).hexdigest()[:16], flush=True)
    for mode in MODES:
        out = 'v185_mech_%s.txt' % mode
        with open(out, 'w') as fh:
            rc = subprocess.call([sys.executable, '-u', 'v185_swamp_mechanism.py', mode, '/kaggle/input'], stdout=fh,
                                 stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONWARNINGS='ignore'))
        print(mode, 'exit', rc); print(open(out).read()[-8000:], flush=True)
    for fn in FILES:
        os.remove(fn)
'''
COMMON = ['v185_swamp_mechanism.py', 'v181_swamp_check.py', 'v178_analyze.py', 'v178_labels.json']
KERNELS = [
    ('aurshalevmerin', 'astrocyte-v185-mech', ['v185', 'v194'], COMMON + ['v194_analyze.py', 'v186_analyze.py'],
     ['aurshalevmerin/astrocyte-v185-ouro', 'aurshalevmerin/astrocyte-v194-ouro']),
    ('aurmerin', 'astrocyte-v181-mech', ['v181'], COMMON, ['aurmerin/astrocyte-v181-ouro']),
]
for acct, kid, modes, files, sources in KERNELS:
    emb = {}
    for f in files:
        raw = open(os.path.join(HERE, f), 'rb').read()
        raw = raw.replace(b'\r\n', b'\n') if f.endswith('.py') else raw
        emb[f] = base64.b64encode(zlib.compress(raw, 9)).decode(); print(kid, f, hashlib.sha256(raw).hexdigest()[:16])
    dd = os.path.join(HERE, 'kaggle-kernel-' + kid.replace('astrocyte-', '')); os.makedirs(dd, exist_ok=True)
    open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
        'FILES = %r\nMODES = %r\n' % (emb, modes) + BODY)
    json.dump(dict(id='%s/%s' % (acct, kid), title=kid, code_file='run.py', language='python', kernel_type='script',
                   is_private=True, enable_gpu=False, enable_internet=False, kernel_sources=sources),
              open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
    print('built', dd)
