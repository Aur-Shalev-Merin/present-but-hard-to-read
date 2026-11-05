"""Kaggle GPU kernel for the v230 pilot (docs/v230_design.md, section 6). Build only.
astrocyte-v230-pilot on account aurshalevmerin (V230_ACCOUNT), T4 x2: gpu0 std:pos:0, gpu1 loop:pos:0 then std:neg:0,
in parallel, child output streamed to the console (and saved per GPU). Push with --accelerator NvidiaTeslaT4.
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__)); ACCT = os.environ.get('V230_ACCOUNT', 'aurshalevmerin')
SLUG = 'pilot'; SCRIPT = 'v230_pilot.py'
# pilot 2 (guard solver fixed): std and looped (60k steps; still 0.56, dropped). Pilot 3 (guard rank from a shuffled-label
# null, w only from its token): seed 0 guarded only (its unguarded fits reproduced exactly in pilots 1-2); seed 1 both,
# for seed spread. (cells, steps, guards)
# Pilot 4 (noise-referenced guard check, DAS m = 32, up to 60k steps; training stops 2k steps after 0.99): seeds 1 and
# 2 full scans, guarded and unguarded; seed 0 key cells only, to compare m = 32 with pilot 3's m = 16.
# (cells, steps, guards, scan)
# Pilot 5 (X / Z DAS at ';', layer-0 known answers; 30k schedule for new seeds): seeds 1, 3, 4 full scans; seeds 0 and 2
# retrained on their pilot 3-4 schedules for X / Z only (seed 0's training reproduced across pilots 1-4; the log's
# step-to-0.99 shows whether seed 2's does).
split = [('std:pos:1,std:pos:3', '30000', '1,0', 'full'),
         ('std:pos:4,std:pos:0:xz:30000,std:pos:2:xz:60000', '30000', '1,0', 'full')]
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode()
       for f in ('v220_train.py', 'v230_pilot.py')}
body = r'''import subprocess, sys, os, base64, zlib, hashlib, threading
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        b = zlib.decompress(base64.b64decode(e)); open(fn, 'wb').write(b); print(fn, hashlib.sha256(b).hexdigest()[:16], flush=True)
    procs = []
    for gpu, (cells, steps, guards, scan) in enumerate(SPLIT):
        os.makedirs('gpu%d' % gpu, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CELLS=cells, STEPS=steps, GUARDS=guards, SCAN=scan,
                   DAS_M='32', OUT_DIR='/kaggle/working/gpu%d' % gpu)
        procs.append(subprocess.Popen([sys.executable, '-u', SCRIPT], env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True))
    def pump(i, p):
        with open('gpu%d/stdout.txt' % i, 'w') as fh:
            for line in p.stdout:
                print('[gpu%d] %s' % (i, line), end='', flush=True); fh.write(line); fh.flush()
    ts = [threading.Thread(target=pump, args=(i, p)) for i, p in enumerate(procs)]
    [t.start() for t in ts]; [t.join() for t in ts]
    print('exit codes', [p.wait() for p in procs], flush=True)
    for fn in FILES:
        os.remove(fn)
'''
dd = os.path.join(HERE, 'kaggle-kernel-v230-' + SLUG); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\nSPLIT = %r\nSCRIPT = %r\n' % (emb, split, SCRIPT) + body)
json.dump(dict(id='%s/astrocyte-v230-%s' % (ACCT, SLUG), title='astrocyte-v230-' + SLUG, code_file='run.py', language='python',
               kernel_type='script', is_private=True, enable_gpu=True, enable_internet=False, kernel_sources=[]),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, ACCT)
