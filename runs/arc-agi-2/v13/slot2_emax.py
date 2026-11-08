"""Slot-2 E[max] grid (draft, zero GPU; artifacts/verification/arc-agi-2/v13/slot2-analysis-draft-20261015.md).
Private score (pp, 120 fresh tasks) = mu + NVARC luck (SD s, fresh draw per task set even with fixed seeds) + program-channel
luck (v13 only, SD p) + program-mean uncertainty (v13 only, SD up). Common NVARC mean uncertainty cancels in A-B.
Gain = E[max(A,B)] - mu_A = sd*phi(d/sd) - d*Phi(-d/sd), d = mu_A - mu_B.
P(max >= L) for the silver/bronze objective adds the common NVARC mean uncertainty (SD U_N, shared by every candidate);
the private task-set shock is left out (shared by NVARC-family candidates and by most teams at the line).
  python3 runs/arc-agi-2/v13/slot2_emax.py"""
import math


def gain(d, sd):
    z = d / sd
    return sd * math.exp(-z * z / 2) / math.sqrt(2 * math.pi) - d * 0.5 * math.erfc(z / math.sqrt(2))


def Phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


def p_max_ge(L, ma, mb, va, vb, cov, n=4000):  # 1 - P(A<L, B<L), bivariate normal by 1-D quadrature over A
    sa, sb = math.sqrt(va), math.sqrt(vb)
    r = cov / (sa * sb)
    za, zb = (L - ma) / sa, (L - mb) / sb
    lo, h = -9.0, (za + 9.0) / n
    tot = 0.0
    for i in range(n + 1):
        a = lo + i * h
        w = 0.5 if i in (0, n) else 1.0
        tot += w * math.exp(-a * a / 2) / math.sqrt(2 * math.pi) * Phi((zb - r * a) / math.sqrt(1 - r * r))
    return 1 - tot * h


U_N = 1.0              # common NVARC-family mean uncertainty (v8 30.14 and fork 30.14 ~ two luck draws)
LINES = (31.6, 32.36)  # bronze-doc mid private line; silver public line 10-15
S_NVARC = 2.0          # fresh-task seed luck (original-family rerun SD 2.1, unverified source; E1 seed SD 1.6/30 -> ~2.7pp@120)
RERUN_SD = 0.6         # same kernel version, fixed seeds (v8 30.97/29.72/29.72) -> rho_same = 1 - RERUN_SD^2 / (2 s^2)
rho_same = 1 - RERUN_SD ** 2 / (2 * S_NVARC ** 2)
MU_NVARC = 30.14       # v8 public mean (3 fixed-seed reruns ~ one luck draw) = fork 30.14 (independent draw)
print(f"rho_same(NVARC rerun) = {rho_same:.3f}")
print("mu_v13  p   up  rhoL | gain v13+NVARC(v8/fork) | gain v13 r1+r2 (rhoP 0.8 / 0.5)")
rows = []
for mu13 in (30.8, 31.5, 32.2):
    for p in (1.0, 2.0):
        for up in (0.7, 1.2):
            for rhoL in (0.0, 0.3):  # E1 same-seed residual corr F vs v13 = -0.01 (slot2-pertask-v1.json) -> 0 is the data value
                sd_h = math.sqrt(2 * S_NVARC ** 2 * (1 - rhoL) + p ** 2 + up ** 2)
                g_h = gain(mu13 - MU_NVARC, sd_h)
                g_t = [gain(0.0, math.sqrt(2 * S_NVARC ** 2 * (1 - rho_same) + 2 * p ** 2 * (1 - rp))) for rp in (0.8, 0.5)]
                va = S_NVARC ** 2 + p ** 2 + up ** 2 + U_N ** 2
                ph = [p_max_ge(L, mu13, MU_NVARC, va, S_NVARC ** 2 + U_N ** 2, rhoL * S_NVARC ** 2 + U_N ** 2) for L in LINES]
                pt = [p_max_ge(L, mu13, mu13, va, va, rho_same * S_NVARC ** 2 + 0.8 * p ** 2 + up ** 2 + U_N ** 2) for L in LINES]
                p1 = [1 - Phi((L - mu13) / math.sqrt(va)) for L in LINES]
                rows.append((mu13, p, up, rhoL, g_h, sd_h, g_t, ph, pt, p1))
                print(f"{mu13:5.1f} {p:4.1f} {up:4.1f} {rhoL:4.1f} | {g_h:5.2f} (sd {sd_h:.2f})          | {g_t[0]:.2f} / {g_t[1]:.2f}")
print()
print("P(max >= L), L = 31.6 / 32.36: v13 alone | v13+NVARC | v13 r1+r2 (rhoP 0.8)")
for mu13, p, up, rhoL, g_h, sd_h, g_t, ph, pt, p1 in rows:
    print(f"{mu13:5.1f} {p:4.1f} {up:4.1f} {rhoL:4.1f} | {p1[0]:.2f} / {p1[1]:.2f} | {ph[0]:.2f} / {ph[1]:.2f} | {pt[0]:.2f} / {pt[1]:.2f}")
