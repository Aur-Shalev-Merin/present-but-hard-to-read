"""v178 label lock (no model; docs/prereg_v178_ouro_latent_intermediates.md section 4).

  python v178_labels.py            -> v178_labels.json (committed at the freeze; its sha256 goes in the freeze commit)

From apple/GSM-Symbolic main/test, keep the templates whose every instance (50) has a target:
  target = first equation result in the worked solution that is not a question number, not the final answer, an integer,
           0 < v < 1e5.
Per item: template id, instance, question, final answer, target, labels (units digit of the target; units digit of the
largest question number = copy control; units digit of the final answer), and the first 8 question numbers (input
baseline features).
"""
import collections, hashlib, json, re

NUM = r'\d+(?:,\d{3})*(?:\.\d+)?'


def nums(s):
    return [float(x.replace(',', '')) for x in re.findall(NUM, s)]


def item(r):
    final = float(r['answer'].split('####')[-1].strip().replace(',', ''))
    qn = nums(r['question']); qs = set(qn)
    tgt = None
    for line in r['answer'].split('####')[0].split('\n'):
        if '=' not in line:
            continue
        rv = nums(line.split('=')[-1])
        if not rv:
            continue
        v = rv[0]
        if v in qs or v == final or v != int(v) or not 0 < v < 1e5:
            continue
        tgt = int(v); break
    if tgt is None or final != int(final) or not qn:
        return None
    big = max(qn)
    return dict(tid=r['original_id'], inst=r['instance'], question=r['question'], final=int(final), target=tgt,
                y_target=tgt % 10, y_copy=int(big) % 10 if big == int(big) else None, y_final=int(final) % 10,
                qnums=qn[:8])


def main():
    from datasets import load_dataset
    sym = load_dataset('apple/GSM-Symbolic', 'main', split='test')
    by_t = collections.defaultdict(list)
    for r in sym:
        by_t[r['original_id']].append(item(r))
    keep = []
    for t in sorted(by_t):
        its = by_t[t]
        if len(its) == 50 and all(i is not None for i in its):
            keep += sorted(its, key=lambda i: i['inst'])
    n_copy = sum(i['y_copy'] is not None for i in keep)
    out = dict(source='apple/GSM-Symbolic main test', n_templates=len({i['tid'] for i in keep}), n_items=len(keep),
               n_copy_label=n_copy, items=keep)
    s = json.dumps(out, sort_keys=True).encode()
    open('v178_labels.json', 'wb').write(s)
    dist = collections.Counter(i['y_target'] for i in keep)
    print('templates %d, items %d, copy labels %d, sha256 %s' % (out['n_templates'], len(keep), n_copy,
                                                                  hashlib.sha256(s).hexdigest()[:16]))
    print('target units-digit distribution:', sorted(dist.items()), '| majority share %.3f' % (max(dist.values()) / len(keep)))


if __name__ == '__main__':
    main()
