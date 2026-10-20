"""v186 item lock (no model): two-step word problems with an unwritten intermediate (docs/prereg_v186_*.md).

  python v186_items.py          -> v186_items.json (its sha256 goes in the freeze commit)

Answer = a + b - c; the intermediate m = a + b is never written in the question.
A family fixes the template, the name and the tens digits of a, b, c; within it, the units digits (a_u, b_u, c_u) vary.
Sets:
  main  : 12 templates x 3 families x 200 of the 1,000 unit-digit combinations (seeded)  = 7,200 items
  pilot : 12 templates x 1 family (other names / tens digits) x 60 combinations          =   720 items
          (pre-freeze: answer-only accuracy and copy-control labels only)
  bare  : 8 families "What is A plus B?" (A, B two-digit, tens fixed) x all 100 (A_u, B_u) =   800 items
          (positive control: the sum's units digit just before Ouro writes it)
Labels per item (units digits): a, b, c; m = (a+b) mod 10; ans = answer mod 10; amb = (a-b) mod 10; apc = (a+c) mod 10;
rnd = a fixed random function of (a_u, b_u) into 0..9 (seed 186).
"""
import hashlib, itertools, json
import numpy as np

T = [
    ('F', '{N} has {a} stickers. Her friend gives her {b} more stickers. She then uses {c} stickers on a card. How many '
          'stickers does {N} have left?'),
    ('-', 'A bakery baked {a} muffins in the morning and {b} muffins in the afternoon. It sold {c} muffins by closing '
          'time. How many muffins were left?'),
    ('M', '{N} had {a} dollars. He earned {b} dollars mowing lawns. Then he spent {c} dollars on a game. How many '
          'dollars does {N} have now?'),
    ('-', 'There were {a} people on a train. At the first stop, {b} people got on. At the second stop, {c} people got '
          'off. How many people are on the train now?'),
    ('-', 'A library had {a} books on a shelf. A volunteer added {b} books. Students borrowed {c} books. How many books '
          'are on the shelf now?'),
    ('F', '{N} picked {a} apples on Monday and {b} apples on Tuesday. She gave {c} apples to her neighbors. How many '
          'apples does {N} have left?'),
    ('M', 'A farmer named {N} had {a} chickens. He bought {b} more chickens. A fox took {c} chickens. How many chickens '
          'does {N} have now?'),
    ('M', '{N} collected {a} seashells. His sister gave him {b} seashells. He lost {c} seashells on the way home. How '
          'many seashells does {N} have?'),
    ('-', 'A parking lot had {a} cars. Then {b} more cars arrived. Later, {c} cars left. How many cars are in the '
          'parking lot now?'),
    ('-', 'A school ordered {a} pencils in September and {b} pencils in October. Teachers handed out {c} pencils. How '
          'many pencils are left?'),
    ('F', '{N} had {a} marbles. She won {b} marbles in a game. She gave {c} marbles to her brother. How many marbles '
          'does {N} have now?'),
    ('-', 'A tank held {a} liters of water. Then {b} liters were added. Later, {c} liters leaked out. How many liters '
          'are in the tank now?'),
]
NAMES = {'F': ['Maya', 'Sofia', 'Emma', 'Lena', 'Nora', 'Ava', 'Chloe', 'Ruth'],
         'M': ['Liam', 'Omar', 'Jack', 'Ethan', 'Noah', 'Felix', 'Hugo', 'Sam'], '-': ['-'] * 8}


def tens(rng):
    while True:
        at, bt, ct = int(rng.integers(3, 7)), int(rng.integers(2, 5)), int(rng.integers(1, 3))
        lo, hi = 10 * (at + bt - ct) - 9, 10 * (at + bt - ct) + 18
        if at + bt <= 8 and lo >= 10 and hi <= 99:
            return at, bt, ct


def labels(au, bu, cu, ans, rnd):
    return dict(a=au, b=bu, c=cu, m=(au + bu) % 10, ans=ans % 10, amb=(au - bu) % 10, apc=(au + cu) % 10,
                rnd=int(rnd[au, bu]))


def main():
    rng = np.random.default_rng(186); rnd = np.random.default_rng(1861).integers(0, 10, (10, 10))
    combos = list(itertools.product(range(10), repeat=3)); items = []; fam = 0
    for split, nfam, ncombo in (('main', 3, 200), ('pilot', 1, 60)):
        for ti, (g, t) in enumerate(T):
            for k in range(nfam):
                name = NAMES[g][k if split == 'main' else 3 + k] if g != '-' else ''
                at, bt, ct = tens(rng)
                for j in sorted(rng.choice(len(combos), ncombo, replace=False)):
                    au, bu, cu = combos[j]
                    a, b, c = 10 * at + au, 10 * bt + bu, 10 * ct + cu
                    items.append(dict(set=split, fam=fam, tid=ti, question=t.format(N=name, a=a, b=b, c=c), a=a, b=b,
                                      c=c, answer=a + b - c, labels=labels(au, bu, cu, a + b - c, rnd)))
                fam += 1
    for k in range(8):
        at, bt = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        for au in range(10):
            for bu in range(10):
                a, b = 10 * at + au, 10 * bt + bu
                items.append(dict(set='bare', fam=fam, tid=100 + k, question='What is %d plus %d?' % (a, b), a=a, b=b,
                                  c=0, answer=a + b, labels=labels(au, bu, 0, a + b, rnd)))
        fam += 1
    out = dict(source='v186 synthetic two-step problems', rnd_table=rnd.tolist(), items=items)
    s = json.dumps(out, sort_keys=True).encode(); open('v186_items.json', 'wb').write(s)
    for split in ('main', 'pilot', 'bare'):
        it = [i for i in items if i['set'] == split]
        m = np.array([i['labels']['m'] for i in it]); an = np.array([i['labels']['ans'] for i in it])
        tab = np.histogram2d(m, an, bins=10)[0]; n = tab.sum(); e = np.outer(tab.sum(1), tab.sum(0)) / n
        v = np.sqrt(((tab - e) ** 2 / np.maximum(e, 1e-9)).sum() / n / 9)
        print('%-5s items %5d families %3d answers %d-%d | Cramer V(m, answer digit) %.3f' % (
            split, len(it), len({i['fam'] for i in it}), min(i['answer'] for i in it), max(i['answer'] for i in it), v))
    print('sha256', hashlib.sha256(s).hexdigest()[:16])


if __name__ == '__main__':
    main()
