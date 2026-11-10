"""Builds v235_pilot2_items.json from pilot 1 (results_v235/pilot/v235_pilot.json): the known cities and people, and
8 targets per country with their patch sources (seed 2350). Duplicate entries are kept once and names in two
countries dropped. Known = the country right on the full continuation and the
two-hop capital right on both the full continuation and the first token.

  python v235_pilot2_items.py
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    R = json.load(open(os.path.join(HERE, 'results_v235', 'pilot', 'v235_pilot.json'), encoding='utf-8'))
    D = json.load(open(os.path.join(HERE, 'v235_cities.json'), encoding='utf-8'))
    P = json.load(open(os.path.join(HERE, 'v235_persons.json'), encoding='utf-8'))
    keys = R['keys']; res = R['results']

    def known(kc, kk):
        c = {r['x'] for r in res if r['kind'] == kc and r['pred'] == r['y']}
        return {r['x'] for r in res if r['kind'] == kk and r['x'] in c and r['pred'] == r['y'] and r['pred_first'] == r['y']}
    kc, kp = known('city_country', 'city_cap'), known('person_country', 'person_cap')
    nc, cities = {}, []
    for c in D['cities']:
        nc.setdefault(c['city'], set()).add(c['ckey'])
    for c in D['cities']:   # duplicate entries within one country kept once; names in two countries dropped
        if c['city'] in kc and len(nc[c['city']]) == 1 and c['city'] not in {x['city'] for x in cities}:
            cities.append(c)
    persons = [p for p in P if p['person'] in kp]
    rng = np.random.default_rng(2350); by = {k: [c for c in cities if c['ckey'] == k] for k in keys}
    targets = []
    for k in keys:
        for i in rng.choice(len(by[k]), 8, replace=False):
            c = by[k][i]; oth = [x for x in cities if x['ckey'] != k]; same = [x for x in by[k] if x['city'] != c['city']]
            po = [x for x in persons if x['ckey'] != k]; ko = [x for x in keys if x != k]
            o, s, p = oth[rng.integers(len(oth))], same[rng.integers(len(same))], po[rng.integers(len(po))]
            targets.append(dict(city=c['city'], ckey=k, city_other=dict(x=o['city'], ckey=o['ckey']),
                                city_same=dict(x=s['city'], ckey=k), person=dict(x=p['person'], ckey=p['ckey']),
                                country=dict(ckey=str(ko[rng.integers(len(ko))]))))
    out = dict(keys=keys, all_cities=[dict(city=c['city'], ckey=c['ckey']) for c in cities], targets=targets,
               persons=[dict(person=p['person'], ckey=p['ckey']) for p in persons])
    json.dump(out, open(os.path.join(HERE, 'v235_pilot2_items.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
    print('%d known cities, %d known people, %d targets' % (len(cities), len(persons), len(targets)))


if __name__ == '__main__':
    main()
