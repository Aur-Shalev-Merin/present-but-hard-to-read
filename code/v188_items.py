"""v188 item lock (no model): confirmation of v186 on NEW templates (docs/prereg_v188_*.md). Derived from v186_items.py.

  python v188_items.py          -> v188_items.json (its sha256 goes in the freeze commit)

Answer = a + b - c; the intermediate m = a + b is never written in the question.
A family fixes the template, the name and the tens digits of a, b, c; within it, the units digits (a_u, b_u, c_u) vary.
Sets:
  main  : 12 NEW templates x 3 families x 200 of the 1,000 unit-digit combinations (seeded) = 7,200 items
  bare  : 16 families "What is A plus B?" (A, B two-digit, tens fixed) x all 100 (A_u, B_u)   = 1,600 items
Labels per item (units digits): a, b, c; m = (a+b) mod 10; ans = answer mod 10; amb = (a-b) mod 10; apc = (a+c) mod 10;
rnd = a fixed random function of (a_u, b_u) into 0..9 (seed 188); abmc = (a+b-2c) mod 10 = (ans-c) mod 10;
abc = (a+b+c) mod 10 = (m+c) mod 10 [NOT used: = ans + 2c, shares ans's parity]; ab2c = (a+b+2c) mod 10 = (ans+3c) mod 10;
ap2b = (a+2b) mod 10; tapb = (2a+b) mod 10.
PRIMARY labels are mod 5 (m5 = (a+b) mod 5 and the same combinations mod 5; rnd5 = a random function of (a_u, b_u)
into 0..4, seed 1882): mod 10, every combination shares its parity with a+b or with the answer digit, which biases the
controls; 5 is prime, so mod-5 combinations that are not proportional to (a+b) or to the answer are clean controls.
"""
import hashlib, itertools, json
import numpy as np

T = [
    ('F', '{N} baked {a} cookies on Friday and {b} cookies on Saturday. Her family ate {c} of them. How many cookies are '
          'left?'),
    ('-', 'A shop had {a} umbrellas in stock. A delivery brought {b} more. Customers bought {c} umbrellas. How many '
          'umbrellas does the shop have now?'),
    ('M', '{N} saved {a} dollars in March and {b} dollars in April. He spent {c} dollars on a jacket. How many dollars '
          'does {N} have left?'),
    ('-', 'A bus left the depot with {a} passengers. At the next stop {b} passengers got on, and at the stop after that '
          '{c} passengers got off. How many passengers are on the bus?'),
    ('-', 'A zoo had {a} visitors before noon and {b} visitors after noon. By evening, {c} visitors had gone home. How '
          'many visitors were still at the zoo?'),
    ('F', '{N} has {a} beads. She buys a bag with {b} more beads. She uses {c} beads to make a bracelet. How many beads '
          'does {N} have now?'),
    ('M', '{N} caught {a} fish in the morning and {b} fish in the afternoon. He released {c} fish back into the lake. '
          'How many fish did {N} keep?'),
    ('-', 'A warehouse stored {a} boxes. Workers unloaded {b} more boxes from a truck. Then {c} boxes were shipped out. '
          'How many boxes are in the warehouse?'),
    ('F', '{N} had {a} stamps in her album. Her aunt sent her {b} stamps. She traded away {c} stamps. How many stamps '
          'are in the album now?'),
    ('-', 'A garden had {a} tulips. The gardener planted {b} more tulips. Rabbits ate {c} tulips. How many tulips are '
          'in the garden?'),
    ('M', '{N} scored {a} points in the first game and {b} points in the second game. He lost {c} points as a penalty. '
          'How many points does {N} have in total?'),
    ('-', 'A cinema sold {a} tickets online and {b} tickets at the door. Then {c} tickets were refunded. How many '
          'tickets were sold in the end?'),
]
NAMES = {'F': ['Maya', 'Sofia', 'Emma', 'Lena', 'Nora', 'Ava', 'Chloe', 'Ruth'],
         'M': ['Liam', 'Omar', 'Jack', 'Ethan', 'Noah', 'Felix', 'Hugo', 'Sam'], '-': ['-'] * 8}


def tens(rng):
    while True:
        at, bt, ct = int(rng.integers(3, 7)), int(rng.integers(2, 5)), int(rng.integers(1, 3))
        lo, hi = 10 * (at + bt - ct) - 9, 10 * (at + bt - ct) + 18
        if at + bt <= 8 and lo >= 10 and hi <= 99:
            return at, bt, ct


def labels(au, bu, cu, ans, rnd, rnd5):
    return dict(a=au, b=bu, c=cu, m=(au + bu) % 10, ans=ans % 10, amb=(au - bu) % 10, apc=(au + cu) % 10,
                rnd=int(rnd[au, bu]), abmc=(au + bu - 2 * cu) % 10, abc=(au + bu + cu) % 10, ap2b=(au + 2 * bu) % 10,
                tapb=(2 * au + bu) % 10, ab2c=(au + bu + 2 * cu) % 10,
                m5=(au + bu) % 5, amb5=(au - bu) % 5, apc5=(au + cu) % 5, ap2b5=(au + 2 * bu) % 5, tapb5=(2 * au + bu) % 5,
                abmc5=(au + bu - 2 * cu) % 5, abc5=(au + bu + cu) % 5, rnd5=int(rnd5[au, bu]))


def main():
    rng = np.random.default_rng(188); rnd = np.random.default_rng(1881).integers(0, 10, (10, 10))
    rnd5 = np.random.default_rng(1882).integers(0, 5, (10, 10))
    combos = list(itertools.product(range(10), repeat=3)); items = []; fam = 0
    for split, nfam, ncombo in (('main', 3, 200),):
        for ti, (g, t) in enumerate(T):
            for k in range(nfam):
                name = NAMES[g][4 + k] if g != '-' else ''
                at, bt, ct = tens(rng)
                for j in sorted(rng.choice(len(combos), ncombo, replace=False)):
                    au, bu, cu = combos[j]
                    a, b, c = 10 * at + au, 10 * bt + bu, 10 * ct + cu
                    items.append(dict(set=split, fam=fam, tid=ti, question=t.format(N=name, a=a, b=b, c=c), a=a, b=b,
                                      c=c, answer=a + b - c, labels=labels(au, bu, cu, a + b - c, rnd, rnd5)))
                fam += 1
    for k in range(16):
        at, bt = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        for au in range(10):
            for bu in range(10):
                a, b = 10 * at + au, 10 * bt + bu
                items.append(dict(set='bare', fam=fam, tid=100 + k, question='What is %d plus %d?' % (a, b), a=a, b=b,
                                  c=0, answer=a + b, labels=labels(au, bu, 0, a + b, rnd, rnd5)))
        fam += 1
    out = dict(source='v188 synthetic two-step problems (new templates)', rnd_table=rnd.tolist(), items=items)
    s = json.dumps(out, sort_keys=True).encode(); open('v188_items.json', 'wb').write(s)
    for split in ('main', 'bare'):
        it = [i for i in items if i['set'] == split]
        m = np.array([i['labels']['m'] for i in it]); an = np.array([i['labels']['ans'] for i in it])
        tab = np.histogram2d(m, an, bins=10)[0]; n = tab.sum(); e = np.outer(tab.sum(1), tab.sum(0)) / n
        v = np.sqrt(((tab - e) ** 2 / np.maximum(e, 1e-9)).sum() / n / 9)
        print('%-5s items %5d families %3d answers %d-%d | Cramer V(m, answer digit) %.3f' % (
            split, len(it), len({i['fam'] for i in it}), min(i['answer'] for i in it), max(i['answer'] for i in it), v))
    print('sha256', hashlib.sha256(s).hexdigest()[:16])


if __name__ == '__main__':
    main()
