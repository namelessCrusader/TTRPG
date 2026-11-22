"""Fit the episode-length law (nights laid up) to the source shares. Run once; the result is written into lab/world2/world.py.
Targets (n = whole nights, n = max(1, round(x))):
  mean 8.7 and sd 20.9      Gurven et al. 2012, Tsimane (doi:10.1016/j.exger.2012.05.006)
  P(n > 3) = 0.91           same paper
  P(n >= 30) = 0.13         Sugiyama 2004, Shiwiar (doi:10.1002/ajpa.10325), a different group
A two-part lognormal mixture (a bulk and a heavy tail), five numbers."""
import math, sys
from scipy.optimize import least_squares
from scipy.stats import lognorm
K = range(1, 4000)
def pmf(p):
    w, m1, s1, m2, s2 = p
    F = lambda x: w * 0.5 * (1 + math.erf((math.log(x) - m1) / (s1 * 2 ** .5))) + (1 - w) * 0.5 * (1 + math.erf((math.log(x) - m2) / (s2 * 2 ** .5)))
    out, prev = [], 0.0
    for k in K:
        c = F(k + 0.5)
        out.append(c - prev); prev = c
    out[0] = F(1.5)
    return out
def stats(p):
    pm = pmf(p); s = sum(pm)
    mean = sum(k * q for k, q in zip(K, pm))
    sd = math.sqrt(sum(k * k * q for k, q in zip(K, pm)) - mean ** 2)
    return mean, sd, 1 - sum(pm[:3]), sum(q for k, q in zip(K, pm) if k >= 30), s
def resid(p, use_sd):
    mean, sd, g3, g30, s = stats(p)
    r = [(mean - 8.7) / 8.7, (g3 - 0.91) / 0.05, (g30 - 0.13) / 0.05]
    if use_sd:
        r.append((sd - 20.9) / 20.9)
    return r
for use_sd in (False, True):
    best = None
    for w0 in (0.6, 0.8, 0.9):
        for m1 in (1.2, 1.6):
            for m2 in (3.2, 3.6):
                r = least_squares(resid, [w0, m1, 0.5, m2, 0.6], args=(use_sd,), bounds=([0.05, 0, 0.05, 1, 0.05], [0.99, 4, 2, 6, 2]))
                if best is None or r.cost < best.cost:
                    best = r
    print("use_sd", use_sd, [round(float(v), 6) for v in best.x], "mean %.2f sd %.1f P>3 %.3f P>=30 %.3f" % stats(best.x)[:4], "cost", best.cost)
# the version used: both spreads fixed at 0.6 (a plain spread, not tuned), the three means fitted to the three shares
print("fixed spreads")
for S1, S2 in ((0.6, 0.6), (0.5, 0.5), (0.7, 0.7)):
    f = lambda q: resid([q[0], q[1], S1, q[2], S2], False)
    best = None
    for w0 in (0.7, 0.85):
        for m1 in (1.2, 1.6):
            for m2 in (3.0, 3.6):
                r = least_squares(f, [w0, m1, m2], bounds=([0.05, 0, 1], [0.99, 4, 6]))
                if best is None or r.cost < best.cost: best = r
    p = [best.x[0], best.x[1], S1, best.x[2], S2]
    print(S1, S2, [round(float(v), 5) for v in p], "mean %.2f sd %.1f P>3 %.3f P>=30 %.3f" % stats(p)[:4], "cost %.2e" % best.cost)
