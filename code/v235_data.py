"""v235 data (pilot stage; docs/v235_design.md). Builds v235_cities.json and v235_persons.json from two public downloads:
  Wikidata SPARQL: cities (P31/P279* Q515) with population > 150,000 in sovereign states (Q6256), English labels
  Compositional Celebrities (Press et al. 2022; HF hwchase17/compositional_celebrities, MIT): birthplace_capital rows
Candidate countries: 25 with >= 20 such cities and a capital whose name does not start with the country's name.
Cities: drop names shared by two countries, names containing a candidate country's name or equal to a capital, names
with characters outside Latin-1; keep the 60 most populous per country.

  python -I v235_data.py <cities.csv> <compositional_celebrities.csv>
"""
import ast, csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
COUNTRIES = {   # Wikidata label -> (name used in prompts, capital)
    "People's Republic of China": ('China', 'Beijing'), 'India': ('India', 'New Delhi'), 'Brazil': ('Brazil', 'Brasília'),
    'United States': ('the United States', 'Washington'), 'Japan': ('Japan', 'Tokyo'), 'Philippines': ('the Philippines', 'Manila'),
    'Indonesia': ('Indonesia', 'Jakarta'), 'Turkey': ('Turkey', 'Ankara'), 'Russia': ('Russia', 'Moscow'),
    'South Korea': ('South Korea', 'Seoul'), 'Pakistan': ('Pakistan', 'Islamabad'), 'Nigeria': ('Nigeria', 'Abuja'),
    'Germany': ('Germany', 'Berlin'), 'Egypt': ('Egypt', 'Cairo'), 'Iran': ('Iran', 'Tehran'), 'Argentina': ('Argentina', 'Buenos Aires'),
    'United Kingdom': ('the United Kingdom', 'London'), 'Spain': ('Spain', 'Madrid'), 'Canada': ('Canada', 'Ottawa'),
    'Poland': ('Poland', 'Warsaw'), 'Italy': ('Italy', 'Rome'), 'France': ('France', 'Paris'), 'Vietnam': ('Vietnam', 'Hanoi'),
    'Peru': ('Peru', 'Lima'), 'Colombia': ('Colombia', 'Bogotá')}
CC_NAME = {'United States': 'United States', 'United Kingdom': 'United Kingdom', "People's Republic of China": 'China'}


def latin1(s):
    try:
        s.encode('latin-1'); return True
    except UnicodeEncodeError:
        return False


def main(cities_csv, cc_csv):
    rows = list(csv.DictReader(open(cities_csv, encoding='utf-8')))
    best = {}
    for x in rows:
        if x['countryLabel'] in COUNTRIES and (x['city'] not in best or float(x['pop']) > float(best[x['city']]['pop'])):
            best[x['city']] = x
    by_name = {}
    for x in best.values():
        by_name.setdefault(x['cityLabel'], set()).add(x['countryLabel'])
    caps = {c for _, c in COUNTRIES.values()}; cnames = [n.replace('the ', '') for n, _ in COUNTRIES.values()]
    out = {}
    for x in sorted(best.values(), key=lambda x: -float(x['pop'])):
        n = x['cityLabel']
        if len(by_name[n]) > 1 or n in caps or any(c in n for c in cnames) or not latin1(n) or n.startswith('Q'):
            continue
        lst = out.setdefault(x['countryLabel'], [])
        if len(lst) < 60:
            lst.append(dict(city=n, qid=x['city'].rsplit('/', 1)[1], pop=int(float(x['pop']))))
    cities = [dict(c, country=COUNTRIES[k][0], capital=COUNTRIES[k][1], ckey=k) for k, v in out.items() for c in v]
    json.dump(dict(countries={k: dict(name=v[0], capital=v[1]) for k, v in COUNTRIES.items()}, cities=cities),
              open(os.path.join(HERE, 'v235_cities.json'), 'w', encoding='utf-8'), indent=0, ensure_ascii=False)
    inv = {v: k for k, v in CC_NAME.items()}
    persons = []
    for x in csv.DictReader(open(cc_csv, encoding='utf-8')):
        if x['category'] != 'birthplace_capital':
            continue
        md = ast.literal_eval(x['metadata']); k = inv.get(md['birth_country'], md['birth_country'])
        if k in COUNTRIES and latin1(md['person']):
            persons.append(dict(person=md['person'], ckey=k, country=COUNTRIES[k][0], capital=COUNTRIES[k][1]))
    json.dump(persons, open(os.path.join(HERE, 'v235_persons.json'), 'w', encoding='utf-8'), indent=0, ensure_ascii=False)
    print('cities per country:', {k: len(v) for k, v in out.items()})
    print('persons:', len(persons), {k: sum(p['ckey'] == k for p in persons) for k in COUNTRIES})


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
