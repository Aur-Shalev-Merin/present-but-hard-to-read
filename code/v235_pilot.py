"""v235 pilot 1 (GPU; docs/v235_design.md): what Qwen2.5-7B (base, fp16, two T4s) knows about the two hops.

Few-shot completion prompts (three fixed demonstrations from countries outside the candidate set):
  city_country   "The city of {X} is located in the country of"             -> the 25 candidate country names
  country_cap    "The capital of {C} is"                                     -> the 25 capitals
  city_cap       "The capital of the country where the city of {X} is located is"   (two-hop, no chain)
  person_country "{P} was born in the country of"
  person_cap     "The capital of the country where {P} was born is"         (two-hop, no chain)
Scoring: full-continuation log-probability of every candidate (argmax over the 25); for capitals also the argmax over
the candidates' first tokens (the clamp's outcome later), with first-token collisions logged.
Output: v235_pilot.json (per item: predictions, ranks, margins) and a per-country accuracy table.
"""
import json, os, sys, time
import numpy as np

OUT = os.environ.get('OUT_DIR', '.'); HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = [('Sydney', 'Australia', 'Canberra'), ('Gothenburg', 'Sweden', 'Stockholm'), ('Mombasa', 'Kenya', 'Nairobi')]
DEMO_P = [('Cate Blanchett', 'Australia', 'Canberra'), ('Greta Garbo', 'Sweden', 'Stockholm'), ('Wangari Maathai', 'Kenya', 'Nairobi')]
T = {'city_country': ('The city of {} is located in the country of', 'country'),
     'country_cap': ('The capital of {} is', 'capital'),
     'city_cap': ('The capital of the country where the city of {} is located is', 'capital'),
     'person_country': ('{} was born in the country of', 'country'),
     'person_cap': ('The capital of the country where {} was born is', 'capital')}


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v235_pilot_log.txt'), 'a').write(s + '\n')


def prompt(kind, x):
    tpl, _ = T[kind]
    if kind == 'country_cap':
        demos = [(c, cap) for _, c, cap in DEMO]
    elif kind.startswith('person'):
        demos = [(p, c if kind == 'person_country' else cap) for p, c, cap in DEMO_P]
    else:
        demos = [(ct, c if kind == 'city_country' else cap) for ct, c, cap in DEMO]
    return ''.join('%s %s.\n' % (tpl.format(a), b) for a, b in demos) + tpl.format(x)


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_grad_enabled(False); t0 = time.time()
    D = json.load(open(os.path.join(HERE, 'v235_cities.json'), encoding='utf-8'))
    P = json.load(open(os.path.join(HERE, 'v235_persons.json'), encoding='utf-8'))
    keys = sorted(D['countries']); names = [D['countries'][k]['name'] for k in keys]; caps = [D['countries'][k]['capital'] for k in keys]
    repo = 'Qwen/Qwen2.5-7B'; tok = AutoTokenizer.from_pretrained(repo)
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map='auto').eval()
    dev = m.get_input_embeddings().weight.device
    first = [tok.encode(' ' + c, add_special_tokens=False)[0] for c in caps]
    coll = {tok.decode([t]): [c for c, f in zip(caps, first) if f == t] for t in set(first)}
    log('loaded in %.0fs | capital first tokens: %s | collisions: %s' % (time.time() - t0, [tok.decode([t]) for t in first],
                                                                       {k: v for k, v in coll.items() if len(v) > 1}))
    cand = {'country': [tok.encode(' ' + n, add_special_tokens=False) for n in names],
            'capital': [tok.encode(' ' + c, add_special_tokens=False) for c in caps]}

    def score(text, kind):
        ids = tok(text).input_ids; cs = cand[T[kind][1]]; L = max(len(c) for c in cs)
        batch = torch.full((len(cs), len(ids) + L), tok.pad_token_id or 0, dtype=torch.long)
        att = torch.zeros_like(batch)
        for j, c in enumerate(cs):
            seq = ids + c; batch[j, :len(seq)] = torch.tensor(seq); att[j, :len(seq)] = 1
        lg = m(input_ids=batch.to(dev), attention_mask=att.to(dev)).logits.float().log_softmax(-1).cpu()
        lp = np.array([sum(float(lg[j, len(ids) - 1 + i, c[i]]) for i in range(len(c))) for j, c in enumerate(cs)])
        ft = lg[0, len(ids) - 1, first].numpy() if T[kind][1] == 'capital' else None
        return lp, ft
    jobs = [('country_cap', D['countries'][k]['name'], keys.index(k), k) for k in keys]
    jobs += [(kd, c['city'], keys.index(c['ckey']), c['ckey']) for c in D['cities'] for kd in ('city_country', 'city_cap')]
    jobs += [(kd, p['person'], keys.index(p['ckey']), p['ckey']) for p in P for kd in ('person_country', 'person_cap')]
    log('%d prompts; example:\n%s' % (len(jobs), prompt('city_cap', D['cities'][0]['city'])))
    res = []; t1 = time.time()
    for n, (kd, x, y, k) in enumerate(jobs):
        lp, ft = score(prompt(kd, x), kd); o = np.argsort(-lp)
        r = dict(kind=kd, x=x, ckey=k, y=y, pred=int(o[0]), rank=int(np.where(o == y)[0][0]), margin=float(lp[y] - np.max(np.delete(lp, y))))
        if ft is not None:
            r['pred_first'] = int(np.argmax(ft))
        res.append(r)
        if n + 1 in (1, 50, 200) or (n + 1) % 500 == 0 or n + 1 == len(jobs):
            log('%d/%d (%.2f s/prompt)' % (n + 1, len(jobs), (time.time() - t1) / (n + 1)))
    json.dump(dict(keys=keys, names=names, caps=caps, results=res), open(os.path.join(OUT, 'v235_pilot.json'), 'w'))
    log('\naccuracy (full continuation; first token for capitals in brackets)')
    log('%-28s %8s %8s %16s %8s %16s' % ('country', 'cap', 'city->c', 'city->cap 2hop', 'pers->c', 'pers->cap 2hop'))
    for k in keys + ['ALL']:
        def acc(kd, first=False):
            rs = [r for r in res if r['kind'] == kd and (k == 'ALL' or r['ckey'] == k)]
            if not rs:
                return '   -'
            a = np.mean([(r['pred_first'] if first else r['pred']) == r['y'] for r in rs])
            return '%.2f' % a + ('' if first else '/%d' % len(rs))
        log('%-28s %8s %8s %9s [%s] %8s %9s [%s]' % (k[:28], acc('country_cap'), acc('city_country'), acc('city_cap'), acc('city_cap', True),
                                                    acc('person_country'), acc('person_cap'), acc('person_cap', True)))
    both = [r for r in res if r['kind'] == 'city_cap']
    knows = {r['x'] for r in res if r['kind'] == 'city_country' and r['pred'] == r['y']}
    kc = [r for r in both if r['x'] in knows]
    log('two-hop city->capital: all %.3f; among cities whose country it knows %.3f (%d); first-token agrees with full %.3f; done %.0fs' % (
        np.mean([r['pred'] == r['y'] for r in both]), np.mean([r['pred'] == r['y'] for r in kc]), len(kc),
        np.mean([r['pred_first'] == r['pred'] for r in both]), time.time() - t0))


if __name__ == '__main__':
    main()
