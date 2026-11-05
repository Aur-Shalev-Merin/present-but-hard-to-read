# v229 post hoc: sensitivity of Ouro P1's FOLLOWS to the bar's multiplier and a needed-share floor

Post hoc (2026-11-05), after a reviewer asked how the one FOLLOWS depends on the bar's 1.5x multiplier and on a floor
on N. Inputs: `results_v229/v229_result.json` only (Part A P1 and the 270 design-matched null runs); no state touched.

P1 (copies a, b, c, the answer and the carry / borrow regressed): N 0.0059 [0.0052, 0.0066], I 0.0034 [0.0026, 0.0041],
I / N 0.57, D+ 0.00108 [0.00001, 0.00213], D- 0.0056 [0.0047, 0.0066] (97.5% CIs).

- **Bar multiplier.** The largest null I / N with signal is 0.149, so the bar is 0.20 / 0.25 / 0.30 / 0.45 at
  multipliers 1 / 1.5 / 2 / 3. P1 (0.57) passes all of them, up to a multiplier of about 3.8.
- **N floor.** P1's N (0.0059, CI lower 0.0052) clears any floor up to 0.005. Every null run above I / N 0.015 has
  N < 0.001: with any floor from 0.001 to 0.005, the worst remaining null I / N is 0.015 and the bar falls to its
  minimum, 0.20. The new floor would make the bar easier to pass, not harder.
- **The binding condition is D+.** The FOLLOWS rests on D+'s interval clearing zero by 0.00001. In a normal
  approximation (SE about 0.00047 from the 97.5% interval), a 99% interval would include zero, and P1 would be UNCLEAR.
  So the label is robust to the bar and the floor but not to the interval level: the sum's lead on plus problems is the
  fragile part, as the paper already says ("its interval just above zero").
