"""Build the v232 kernels (docs/prereg_v232_ouro_sb.md). Build only.

  V232_MODE=cpu    python build_kaggle_v232.py -> astrocyte-v232-cpu (CPU): Part A (null first, then the v199 states) and the
                                                 projected Part B plan (kernel_sources astrocyte-v199-ouro, astrocyte-v199-cpu)
  V232_MODE=patch  python build_kaggle_v232.py -> astrocyte-v232-patch (GPU): v199_patch.py, byte for byte, one half of
                                                 v199's pairs per GPU on the projected plan (kernel_sources astrocyte-v232-cpu
                                                 only), then v232_sb.py b
"""
import base64, hashlib, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__)); ME = 'aurshalevmerin'
MODE = os.environ.get('V232_MODE', 'cpu')
ANA = ['v232_sb.py', 'v228_rescore.py', 'v224_share.py', 'v186_analyze.py', 'v178_analyze.py', 'v194_items.json']
FILES = ANA if MODE == 'cpu' else ANA + ['v199_extract.py', 'v199_patch.py', 'loop_grad_probe.py']
emb = {}
for f in FILES:
    raw = open(os.path.join(HERE, f), 'rb').read()
    b = raw.replace(b'\r\n', b'\n') if f.endswith('.py') else raw
    emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
    print('  %s sha256 %s' % (f, hashlib.sha256(b).hexdigest()[:16]))
body = r'''import subprocess, sys, os, base64, zlib, time, json, hashlib
if __name__ == '__main__':
    W = '/kaggle/working'; os.chdir(W)
    for fn, e in FILES.items():
        b = zlib.decompress(base64.b64decode(e)); open(fn, 'wb').write(b); print(fn, hashlib.sha256(b).hexdigest()[:16], flush=True)
    env0 = dict(os.environ, OUT_DIR=W, PYTHONUNBUFFERED='1', PYTHONWARNINGS='ignore')
    if MODE == 'cpu':
        with open('v232_cpu.txt', 'w') as out:
            p = subprocess.Popen([sys.executable, '-u', 'v232_sb.py', 'cpu', '/kaggle/input'], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, env=env0, text=True)
            for line in p.stdout:
                print(line, end='', flush=True); out.write(line)
            rc = p.wait()
        for fn in FILES:
            os.remove(fn)
        print('exit', rc, flush=True)
        if rc:
            raise SystemExit('v232 cpu failed')
    else:
        subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.54.1', 'datasets'])
        t0 = time.time(); procs = []
        for gpu in range(2):
            od = os.path.join(W, 'gpu%d' % gpu); os.makedirs(od, exist_ok=True)
            env = dict(env0, CUDA_VISIBLE_DEVICES=str(gpu), OUT_DIR=od, LOOP_MODEL='ouro', V199_PART=str(gpu),
                       PYTORCH_ALLOC_CONF='expandable_segments:True', HF_HUB_DISABLE_PROGRESS_BARS='1')
            out = open('v232_patch_stdout_gpu%d.txt' % gpu, 'w')
            procs.append((subprocess.Popen([sys.executable, '-u', 'v199_patch.py', 'extract', '/kaggle/input'], stdout=out,
                                           stderr=subprocess.STDOUT, env=env), out))
            time.sleep(60)
        rcs = []
        for p, out in procs:
            rcs.append(p.wait()); out.close(); print('exit', rcs[-1], 'elapsed %ds' % (time.time() - t0), flush=True)
        for gpu in range(2):
            print(open('v232_patch_stdout_gpu%d.txt' % gpu).read()[-3000:], flush=True)
        if any(rcs):
            raise SystemExit('a GPU job failed')
        with open('v232_partB.txt', 'w') as out:
            subprocess.call([sys.executable, '-u', 'v232_sb.py', 'b', W], stdout=out, stderr=subprocess.STDOUT, env=env0)
        print(open('v232_partB.txt').read(), flush=True)
'''
kid = 'astrocyte-v232-' + MODE
dd = os.path.join(HERE, 'kaggle-kernel-v232-' + MODE); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\nMODE = %r\n' % (emb, MODE) + body)
json.dump(dict(id=ME + '/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script', is_private=True,
               enable_gpu=MODE == 'patch', enable_internet=MODE == 'patch',
               kernel_sources=[ME + '/astrocyte-v199-ouro', ME + '/astrocyte-v199-cpu'] if MODE == 'cpu' else [ME + '/astrocyte-v232-cpu']),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
