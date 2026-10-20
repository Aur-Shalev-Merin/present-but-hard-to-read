"""Build the v176b Kaggle kernel (scoring fix: no single-newline terminator; Huginn to 32 loops) (answer-only GSM8K vs loops, Huginn + Ouro; docs/v176_plan.md). Build only; does not push.
One kernel on aurshalevmerin, T4 x2: each GPU runs its half of the Huginn problems, then its half of the Ouro problems.
"""
import base64, hashlib, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['v176_answer_only.py', 'loop_grad_probe.py']
JOBS = [[('huginn', '0/2'), ('ouro', '0/2')], [('huginn', '1/2'), ('ouro', '1/2')]]
emb = {}
for f in FILES:
    b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
    emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
    print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))
body = r'''import subprocess, sys, os, base64, zlib, time
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.54.1', 'datasets'])
    t0 = time.time(); procs = []
    for gpu, jobs in enumerate(JOBS):
        od = '/kaggle/working/gpu%d' % gpu; os.makedirs(od, exist_ok=True)
        cmd = ' && '.join('CUDA_VISIBLE_DEVICES=%d OUT_DIR=%s LOOP_MODEL=%s V176_SHARD=%s PYTHONUNBUFFERED=1 '
                          'PYTORCH_ALLOC_CONF=expandable_segments:True %s -u v176_answer_only.py' % (gpu, od, m, s, sys.executable)
                          for m, s in jobs)
        out = open('v176_stdout_gpu%d.txt' % gpu, 'w')
        procs.append((subprocess.Popen(cmd, shell=True, stdout=out, stderr=subprocess.STDOUT), out))
        time.sleep(60)
    for p, out in procs:
        rc = p.wait(); out.close(); print('job exit', rc, 'elapsed %ds' % (time.time() - t0), flush=True)
    for gpu in range(len(JOBS)):
        print(open('v176_stdout_gpu%d.txt' % gpu).read()[-5000:], flush=True)
'''
dd = os.path.join(HERE, 'kaggle-kernel-v176b-answeronly'); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\nJOBS = %r\n' % (emb, JOBS) + body)
json.dump(dict(id='aurshalevmerin/astrocyte-v176b-answeronly', title='astrocyte-v176b-answeronly', code_file='run.py',
               language='python', kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
