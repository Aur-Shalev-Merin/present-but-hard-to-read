"""v194 item lock (no model): operation swap (docs/prereg_v194_*.md). Derived from v188_items.py.

  python v194_items.py          -> v194_items.json (its sha256 goes in the freeze commit)

Each template has two versions that differ only in the first step's phrase:
  plus : first step a + b, answer a + b - c        minus: first step a - b, answer a - b - c
A family fixes the template, version, name and the tens digits of a, b, c; within it the units digits (a_u, b_u, c_u)
vary. A plus family and its minus twin share name, tens digits and unit-digit combinations ('pair' id), so every plus
item has a minus twin with identical numbers. Tens digits are chosen so both answers are positive two-digit numbers:
a_t in 6..8, b_t and c_t in 1..2, with a_t - b_t - c_t >= 3 and a_t + b_t - c_t <= 8.
Sets:
  main  : 12 templates x 2 versions x 3 families x 200 of the 1,000 unit-digit combinations = 14,400 items
  bare  : 16 families "What is A plus B?" x all 100 (A_u, B_u) = 1,600 items (gate G3)
  pilot : 12 templates x 2 versions x 1 family x 30 combinations = 720 items, other names and tens (pre-freeze, control
          labels only)
"""
import hashlib, itertools, json
import numpy as np

# (gender, template with {OP}, plus phrase, minus phrase)
T = [
    ('F', '{N} had {a} stickers. Her friend {OP}. Then she used {c} stickers on a card. How many stickers does {N} have '
          'now?', 'gave her {b} more stickers', 'took {b} of her stickers'),
    ('-', 'A library had {a} books on a shelf. A volunteer {OP}. Later {c} books were checked out. How many books are '
          'on the shelf now?', 'added {b} books to the shelf', 'removed {b} books from the shelf'),
    ('M', '{N} had {a} marbles. His brother {OP}. Then {N} lost {c} marbles at school. How many marbles does {N} have '
          'now?', 'gave him {b} more marbles', 'took {b} of his marbles'),
    ('-', 'A parking lot had {a} cars. In the next hour {OP}. Later {c} cars drove away. How many cars are in the '
          'parking lot now?', '{b} more cars arrived', '{b} of the cars left'),
    ('F', '{N} picked {a} apples. Her neighbour {OP}. She gave {c} apples to her teacher. How many apples does {N} '
          'have left?', 'gave her {b} more apples', 'took {b} of her apples'),
    ('-', 'A tank held {a} litres of water. A pump {OP}. Then {c} litres leaked out. How many litres are in the tank '
          'now?', 'added {b} litres', 'drained {b} litres'),
    ('M', '{N} had {a} trading cards. At the fair he {OP}. He then gave {c} cards to his cousin. How many cards does '
          '{N} have now?', 'won {b} more cards', 'lost {b} of his cards'),
    ('-', 'A farm had {a} chickens. The farmer {OP}. A fox took {c} chickens. How many chickens are on the farm now?',
          'bought {b} more chickens', 'sold {b} of the chickens'),
    ('F', '{N} had {a} dollars in her wallet. On Monday she {OP}. On Tuesday she spent {c} dollars on lunch. How many '
          'dollars does {N} have now?', 'earned {b} more dollars', 'spent {b} dollars on a book'),
    ('-', 'A school club had {a} members. This month the club {OP}. Then {c} members moved away. How many members '
          'does the club have now?', 'gained {b} new members', 'lost {b} members'),
    ('M', '{N} had {a} songs on his playlist. He {OP}. Later he removed {c} songs he no longer liked. How many songs '
          'are on the playlist now?', 'added {b} new songs', 'deleted {b} old songs'),
    ('-', 'A bakery had {a} loaves of bread in the morning. The baker {OP}. Customers then bought {c} loaves. How many '
          'loaves are left?', 'baked {b} more loaves', 'threw away {b} stale loaves'),
]
NAMES = {'F': ['Iris', 'Clara', 'Mila', 'Hana'], 'M': ['Owen', 'Leo', 'Theo', 'Max'], '-': [''] * 4}


def tens(rng):
    while True:
        at, bt, ct = int(rng.integers(6, 9)), int(rng.integers(1, 3)), int(rng.integers(1, 3))
        if at - bt - ct >= 3 and at + bt - ct <= 8:
            return at, bt, ct


def main():
    rng = np.random.default_rng(194)
    combos = list(itertools.product(range(10), repeat=3)); items = []; fam = 0; pair = 0
    for split, nfam, ncombo, name0 in (('main', 3, 200, 0), ('pilot', 1, 30, 3)):
        for ti, (g, t, plus, minus) in enumerate(T):
            for k in range(nfam):
                name = NAMES[g][name0 + k]
                at, bt, ct = tens(rng)
                chosen = sorted(rng.choice(len(combos), ncombo, replace=False))
                for op, phrase in (('plus', plus), ('minus', minus)):
                    for j in chosen:
                        au, bu, cu = combos[j]
                        a, b, c = 10 * at + au, 10 * bt + bu, 10 * ct + cu
                        ans = a + b - c if op == 'plus' else a - b - c
                        q = t.replace('{OP}', phrase).format(N=name, a=a, b=b, c=c)
                        items.append(dict(set=split, fam=fam, pair=pair, tid=ti, op=op, question=q, a=a, b=b, c=c,
                                          answer=ans, labels=dict(a=au, b=bu, c=cu, ans=ans % 10)))
                    fam += 1
                pair += 1
    for k in range(16):
        at, bt = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        for au in range(10):
            for bu in range(10):
                a, b = 10 * at + au, 10 * bt + bu
                items.append(dict(set='bare', fam=fam, pair=pair, tid=100 + k, op='plus', question='What is %d plus %d?'
                                  % (a, b), a=a, b=b, c=0, answer=a + b, labels=dict(a=au, b=bu, c=0, ans=(a + b) % 10)))
        fam += 1; pair += 1
    out = dict(source='v194 operation swap (plus/minus first step)', items=items)
    s = json.dumps(out, sort_keys=True).encode(); open('v194_items.json', 'wb').write(s)
    for split in ('main', 'pilot', 'bare'):
        for op in ('plus', 'minus'):
            it = [i for i in items if i['set'] == split and i['op'] == op]
            if not it:
                continue
            assert all(10 <= i['answer'] <= 99 for i in it), (split, op)
            print('%-5s %-5s items %5d families %3d answers %d-%d' % (
                split, op, len(it), len({i['fam'] for i in it}), min(i['answer'] for i in it),
                max(i['answer'] for i in it)))
    print('example plus :', items[0]['question'])
    print('example minus:', items[200]['question'])
    print('sha256', hashlib.sha256(s).hexdigest()[:16])


if __name__ == '__main__':
    main()
