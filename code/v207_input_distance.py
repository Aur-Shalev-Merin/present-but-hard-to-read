"""Input-distance check (exploratory, descriptive): how far apart are digit token embeddings, compared with the
v205 word swaps and the v205 embedding noise (norm = median embedding norm)? Fetches only the needed rows of the
input-embedding matrix by HTTP range requests on the public safetensors file."""
import json, re, struct, sys
import numpy as np, requests
from transformers import AutoTokenizer

sys.path.insert(0, '.')


def swap_tokens(tok, n=10, seed=205):   # copied from v205_extract.swap_tokens
    rng = np.random.default_rng(seed); ids = rng.permutation(len(tok)); out = []
    for i in ids:
        s = tok.decode([int(i)])
        if re.fullmatch(r' [a-z]{3,8}', s):
            out.append(int(i))
        if len(out) == n:
            return out


def rows_fetcher(repo):
    base = 'https://huggingface.co/%s/resolve/main/' % repo
    r = requests.get(base + 'model.safetensors.index.json')
    fn = r.json()['weight_map'] if r.ok else None
    key = None
    for k in (['model.embed_tokens.weight'] if fn is None else list(fn)):
        if k.endswith('embed_tokens.weight'):
            key = k
    f = 'model.safetensors' if fn is None else fn[key]
    h = requests.get(base + f, headers={'Range': 'bytes=0-7'}, allow_redirects=True).content
    n = struct.unpack('<Q', h)[0]
    hdr = json.loads(requests.get(base + f, headers={'Range': 'bytes=8-%d' % (7 + n)}).content)
    t = hdr[key]; V, d = t['shape']; dt = t['dtype']; off = 8 + n + t['data_offsets'][0]
    assert dt in ('BF16', 'F16', 'F32'), dt
    bs = 4 if dt == 'F32' else 2

    def row(i):
        a = off + i * d * bs
        b = requests.get(base + f, headers={'Range': 'bytes=%d-%d' % (a, a + d * bs - 1)}).content
        if dt == 'F32':
            return np.frombuffer(b, '<f4').astype(np.float64)
        if dt == 'F16':
            return np.frombuffer(b, '<f2').astype(np.float64)
        u = np.frombuffer(b, '<u2').astype(np.uint32) << 16
        return u.view(np.float32).astype(np.float64)
    return row, V, d, dt, key


def pd(E):
    return np.array([np.linalg.norm(E[i] - E[j]) for i in range(len(E)) for j in range(i + 1, len(E))])


def run(repo, noise_norm=None):
    tok = AutoTokenizer.from_pretrained(repo)
    row, V, d, dt, key = rows_fetcher(repo)
    dig = [tok.convert_tokens_to_ids(str(k)) for k in range(10)]
    assert all(tok.decode([i]) == str(k) for k, i in enumerate(dig))
    sw = swap_tokens(tok)
    rng = np.random.default_rng(0); rnd = []
    for i in rng.permutation(len(tok)):
        if re.fullmatch(r' [a-z]{3,8}', tok.decode([int(i)])) and int(i) not in sw:
            rnd.append(int(i))
        if len(rnd) == 60:
            break
    nrm_ids = [int(i) for i in rng.choice(min(V, len(tok)), 300, replace=False)]
    from concurrent.futures import ThreadPoolExecutor
    ids = sorted(set(dig + sw + rnd + nrm_ids))
    with ThreadPoolExecutor(32) as ex:
        E = dict(zip(ids, ex.map(row, ids)))
    D = np.array([E[i] for i in dig]); W = np.array([E[i] for i in sw]); R = np.array([E[i] for i in rnd])
    med = np.median([np.linalg.norm(E[i]) for i in nrm_ids])
    dd = pd(D); ww = pd(W); rr = pd(R)
    dw = np.array([np.linalg.norm(a - b) for a in D for b in W])
    print('\n== %s  (%s, %s, V=%d, d=%d)' % (repo, key, dt, V, d))
    print('swap words:', [tok.decode([i]) for i in sw])
    print('median token norm (300 random rows): %.3f%s' % (med, '' if noise_norm is None else '  (v205 noise norm %.3f)' % noise_norm))
    print('digit norms: mean %.3f  | swap-word norms: mean %.3f' % (np.linalg.norm(D, axis=1).mean(), np.linalg.norm(W, axis=1).mean()))
    print('digit vs digit   (minimal pair input change): mean %.3f  range %.3f-%.3f' % (dd.mean(), dd.min(), dd.max()))
    print('digit vs swap word (swap input change):       mean %.3f  range %.3f-%.3f' % (dw.mean(), dw.min(), dw.max()))
    print('swap word vs swap word:                       mean %.3f' % ww.mean())
    print('random word vs random word (60 words):        mean %.3f' % rr.mean())
    if noise_norm is not None:
        print('noise input change = %.3f' % noise_norm)
    # digit differences: how much of their spread lies in a few directions?
    Dc = D - D.mean(0); s = np.linalg.svd(Dc, compute_uv=False) ** 2
    print('digit-centred variance: top1 %.2f, top3 %.2f of total' % (s[0] / s.sum(), s[:3].sum() / s.sum()))
    print('ratios: digit-digit / word-word %.2f ; digit-digit / digit-word %.2f ; digit-digit / noise %s' % (
        dd.mean() / rr.mean(), dd.mean() / dw.mean(), '%.2f' % (dd.mean() / noise_norm) if noise_norm else 'n/a'))


if __name__ == '__main__':
    run('Qwen/Qwen2.5-1.5B', noise_norm=1.032)
    run('ByteDance/Ouro-1.4B')
