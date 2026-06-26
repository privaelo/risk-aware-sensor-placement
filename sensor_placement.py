"""Sensor-placement demo: nominal vs risk-aware planner under capability uncertainty.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = "figures"

# Configuration

GRID_N    = 30       # region discretized into GRID_N x GRID_N points
CAND_N    = 8        # candidate sensor locations per axis (8x8 = 64 total)
SIGMA_L   = 0.01     # sensor detection range (reach ~ sqrt(sigma_l) = 0.1)
T_C       = 1.0      # observation period
K         = 5        # sensors to place
N_MC      = 10_000   # Monte Carlo samples for evaluation
N_MC_PLAN = 500      # cheaper MC used during planning
N_MC_GAP  = 200_000  # high-resolution MC for the Jensen-gap figure
KAPPA     = 10.0     # risk penalty strength (higher = more conservative)
MU        = 0.93     # mean capability (mu + 3*sigma < 1 must hold across the sweep)
SIGMA_SWEEP = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]  # uncertainty scale factors
EVAL_SEED = 123      # fixed eval RNG: equal placements -> equal metrics

# Environment

def make_world():
    """Grid of points + target intensity map (3 Gaussian hotspots)."""
    ax = np.linspace(0, 1, GRID_N)
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)          # (G, 2)
    dA  = (ax[1] - ax[0]) ** 2

    modes   = np.array([[0.30, 0.65], [0.70, 0.35], [0.55, 0.80]])
    weights = np.array([1.0, 0.8, 0.5])
    scales  = np.array([0.12, 0.10, 0.08])
    lam = np.zeros(len(pts))
    for m, w, s in zip(modes, weights, scales):
        lam += w * np.exp(-((pts - m) ** 2).sum(1) / (2 * s * s))
    lam *= 12.0     # amplitude: keeps e^-Lambda in the 0.05-0.5 regime
    return pts, lam, dA


def make_candidates():
    """64 candidate sensor locations on a uniform grid."""
    ax = np.linspace(0.05, 0.95, CAND_N)
    CX, CY = np.meshgrid(ax, ax, indexing="ij")
    return np.stack([CX.ravel(), CY.ravel()], axis=1)       # (C, 2)


def geometric_factors(cands, pts):
    """Detection weight g[i, l]: how well sensor at cands[i] covers grid point pts[l]."""
    d2 = ((cands[:, None, :] - pts[None, :, :]) ** 2).sum(2)  # (C, G)
    return np.exp(-d2 / SIGMA_L)

# Sensor units

def make_unit_pool(sigma_scale, mu=MU, sigma_small=0.001, sigma_large_base=0.007):
    """2K units with identical mean capability mu.
      - K calibrated:   tiny uncertainty (sigma_small)
      - K uncalibrated: growing uncertainty (sigma_large_base * sigma_scale)
    """
    n = 2 * K
    mus    = np.full(n, mu)
    sigmas = np.full(n, sigma_small)
    sigmas[K:] = sigma_large_base * sigma_scale
    assert np.all(mus + 3 * sigmas < 1.0), (
        f"infeasible: max(mu+3sigma)={np.max(mus + 3*sigmas):.4f} >= 1; lower sigma range")
    return mus, sigmas

# Core math

def nu_of(placement, rho, G, lam, dA):
    """Void probability nu = exp(-U) for a placement.
    """
    rho = np.asarray(rho, dtype=float)
    scalar = rho.ndim == 1

    if len(placement) == 0:
        U = lam.sum() * dA / T_C
        return float(np.exp(-U)) if scalar else np.full(rho.shape[0], np.exp(-U))

    g = G[[c for c, _ in placement]]                        # (k, G)
    R = np.atleast_2d(rho)                                  # (S, k)
    pi_C = np.prod(1.0 - R[:, :, None] * g[None, :, :], axis=1)  # (S, G)
    U    = (lam[None, :] * pi_C).sum(1) * dA / T_C          # (S,)
    return float(np.exp(-U[0])) if scalar else np.exp(-U)


def sample_rho(placement, mu, sigma, n, rng):
    """Draw n capability samples for the units in the placement."""
    units = [u for _, u in placement]
    return rng.normal(mu[units], sigma[units], size=(n, len(units)))


def evaluate(placement, mu, sigma, G, lam, dA, rng, n=N_MC):
    """Distribution of nu over n MC capability draws, plus mean/std/q05/nu_at_mu."""
    rho = sample_rho(placement, mu, sigma, n, rng)
    nus = nu_of(placement, rho, G, lam, dA)
    units = [u for _, u in placement]
    return {
        "dist": nus,
        "mean": nus.mean(),
        "std":  nus.std(),
        "q05":  np.quantile(nus, 0.05),
        "nu_at_mu": nu_of(placement, mu[units], G, lam, dA),
    }


def marginal_B(placement, swept_unit, mu, G, lam, dA):
    """B = sensitivity of the undetected mass U to capability of `swept_unit`.

    U = sum_l lam(l) dA/T_c * prod_i (1 - rho_i g_i(l)). With nu = exp(-U),
    d nu / d rho_swept = B * nu, where
        B = sum_l [lam(l) dA/T_c] * g_swept(l) * prod_{i != swept} (1 - mu_i g_i(l)).
    Evaluated at rho = mu. This is the slope driving the Jensen gap (1/2) B^2 sigma^2.
    """
    coef = lam * dA / T_C
    sub = [(c, u) for (c, u) in placement if u != swept_unit]
    swept_c = next(c for (c, u) in placement if u == swept_unit)
    rest = np.ones_like(lam)
    for c, u in sub:
        rest *= 1.0 - mu[u] * G[c]
    return float((coef * G[swept_c] * rest).sum())

# Planners

def greedy(objective, n_cands, n_units):
    """Generic greedy: pick the best (location, unit) pair one at a time.
    Locations and units are each used at most once.
    """
    placement, used_c, used_u = [], set(), set()
    for _ in range(K):
        best, best_val = None, -np.inf
        for c in range(n_cands):
            if c in used_c:
                continue
            for u in range(n_units):
                if u in used_u:
                    continue
                val = objective(placement + [(c, u)])
                if val > best_val:
                    best, best_val = (c, u), val
        placement.append(best)
        used_c.add(best[0])
        used_u.add(best[1])
    return placement


def nominal_objective(mu, sigma, G, lam, dA):
    """Score = nu at the belief mean mu; sigma is invisible to this planner.
    """
    TIE_EPS = 1e-6
    def obj(placement):
        units = [u for _, u in placement]
        return nu_of(placement, mu[units], G, lam, dA) + TIE_EPS * sigma[units].sum()
    return obj


def risk_aware_objective(mu, sigma, rng_plan, G, lam, dA):
    """Score = E[nu] - kappa * Std[nu], estimated with N_MC_PLAN samples.
    """
    rho_fixed = rng_plan.normal(mu, sigma, size=(N_MC_PLAN, len(mu)))  # (S, 2K)
    coef = lam * dA / T_C

    base      = np.ones((N_MC_PLAN, len(lam)))
    committed = ()

    def obj(placement):
        nonlocal base, committed
        prefix, (c, u) = placement[:-1], placement[-1]
        if tuple(prefix) != committed:
            base = np.ones_like(base)
            for pc, pu in prefix:
                base *= 1.0 - rho_fixed[:, pu][:, None] * G[pc][None, :]
            committed = tuple(prefix)
        pi_C = base * (1.0 - rho_fixed[:, u][:, None] * G[c][None, :])
        nus  = np.exp(-(pi_C * coef[None, :]).sum(1))
        return float(nus.mean() - KAPPA * nus.std())

    return obj

# Sweep

def run_sweep(G, lam, dA):
    results = []
    for scale in SIGMA_SWEEP:
        mu, sigma = make_unit_pool(scale)

        pl_nom  = greedy(nominal_objective(mu, sigma, G, lam, dA), G.shape[0], len(mu))
        pl_risk = greedy(risk_aware_objective(mu, sigma, np.random.default_rng(0), G, lam, dA),
                         G.shape[0], len(mu))

        r_nom  = evaluate(pl_nom,  mu, sigma, G, lam, dA, np.random.default_rng(EVAL_SEED))
        r_risk = evaluate(pl_risk, mu, sigma, G, lam, dA, np.random.default_rng(EVAL_SEED))

        results.append({
            "scale": scale, "sigma_large": float(sigma[K]),
            "pl_nom": pl_nom, "pl_risk": pl_risk,
            "nom": r_nom, "risk": r_risk,
            "same_placement": set(pl_nom) == set(pl_risk),
        })
    return results


def run_jensen(G, lam, dA):
    """Realized Jensen gap E[nu] - nu(mu) on one fixed placement across the sweep.
    """
    mu0, sigma0 = make_unit_pool(SIGMA_SWEEP[1])
    placement = greedy(nominal_objective(mu0, sigma0, G, lam, dA), G.shape[0], len(mu0))
    uncertain = [u for (_, u) in placement if u >= K]      # units the sweep perturbs
    B2_sum = sum(marginal_B(placement, u, mu0, G, lam, dA) ** 2 for u in uncertain)

    sig2, gap, sem, pred = [], [], [], []
    CHUNK = 20_000  # batch MC draws so the (chunk, k, G) tensor stays small
    for scale in SIGMA_SWEEP:
        mu, sigma = make_unit_pool(scale)
        rng = np.random.default_rng(7)
        nu_mu = nu_of(placement, mu[[u for _, u in placement]], G, lam, dA)
        s = sigma[uncertain[0]] if uncertain else 0.0      # common sigma of uncertain units
        n_sum = n_sq = 0.0
        done = 0
        while done < N_MC_GAP:
            b = min(CHUNK, N_MC_GAP - done)
            nus = nu_of(placement, sample_rho(placement, mu, sigma, b, rng), G, lam, dA)
            n_sum += nus.sum(); n_sq += (nus ** 2).sum(); done += b
        mean = n_sum / N_MC_GAP
        var  = max(n_sq / N_MC_GAP - mean ** 2, 0.0)
        sig2.append(s ** 2)
        gap.append(mean - nu_mu)
        sem.append(np.sqrt(var) / np.sqrt(N_MC_GAP))
        pred.append(0.5 * B2_sum * s ** 2 * nu_mu)
    return {"sig2": np.array(sig2), "gap": np.array(gap),
            "sem": np.array(sem), "pred": np.array(pred),
            "B2": B2_sum, "n_uncertain": len(uncertain)}

# Figures

def make_figures(results, jensen, lam, cands):
    os.makedirs(FIG_DIR, exist_ok=True)
    sig_large = [r["sigma_large"] for r in results]
    r_big = results[-1]

    # (a) Target intensity map + sensor placements
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(lam.reshape(GRID_N, GRID_N).T, origin="lower",
                   extent=[0, 1, 0, 1], cmap="viridis", alpha=0.9)
    fig.colorbar(im, ax=ax, label="Target intensity lambda(l)")
    for label, key, marker, size, color in [
        ("Nominal (uncalibrated units)",  "pl_nom",  "o", 150, "white"),
        ("Risk-aware (calibrated units)", "pl_risk", "s",  70, "red"),
    ]:
        xy = cands[[c for c, _ in r_big[key]]]
        ax.scatter(xy[:, 0], xy[:, 1], s=size, marker=marker,
                   facecolors="none", edgecolors=color, linewidths=2, label=label)
    ax.set_title(f"Sensor placements (largest sigma, kappa={KAPPA})", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_a_placements.png"), dpi=130)

    # (b) nu distributions at largest sigma
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for label, key, color in [("Nominal", "nom", "C0"), ("Risk-aware", "risk", "C1")]:
        r = r_big[key]
        ax.hist(r["dist"], bins=60, density=True, alpha=0.55,
                color=color, label=f"{label} (mean={r['mean']:.4f})")
        ax.axvline(r["q05"], color=color, ls="--", lw=1.5)
    ax.set_xlabel("nu (void probability)"); ax.set_ylabel("Density")
    ax.set_title(f"nu distributions at sigma_large={sig_large[-1]:.4f}  (dashed = q05)")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_b_distributions.png"), dpi=130)

    # (c) Downside (q05) vs sigma
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(sig_large, [r["nom"]["q05"]  for r in results], "o-", label="Nominal q05")
    ax.plot(sig_large, [r["risk"]["q05"] for r in results], "s-", label="Risk-aware q05")
    ax.set_xlabel("sigma_large (sensor uncertainty)"); ax.set_ylabel("Downside nu")
    ax.set_title("Downside nu vs sensor uncertainty")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_c_downside.png"), dpi=130)

    # (d) Jensen gap vs sigma^2
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(jensen["sig2"], jensen["pred"], "k-",
            label=r"predicted $\frac{1}{2}\sigma^2\nu(\mu)\sum_i B_i^2$" + f"  (sum={jensen['B2']:.4f}, {jensen['n_uncertain']} units)")
    ax.errorbar(jensen["sig2"], jensen["gap"], yerr=jensen["sem"], fmt="s", color="C1",
                capsize=3, label=f"MC realized gap (n={N_MC_GAP:.0e})")
    ax.set_xlabel(r"$\sigma^2$ (uncertain units)"); ax.set_ylabel(r"mean $\nu$ $-$ $\nu(\mu)$")
    ax.set_title("Jensen gap vs sigma^2")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_d_jensen.png"), dpi=130)

    plt.close("all")

# Main

def main():
    pts, lam, dA = make_world()
    cands = make_candidates()
    G = geometric_factors(cands, pts)

    nu_empty = float(np.exp(-lam.sum() * dA / T_C))
    print(f"Empty-placement void prob: e^-Lambda = {nu_empty:.3f}")
    print(f"mu = {MU}, eval seed = {EVAL_SEED}")
    print(f"Sweep over sigma scales: {SIGMA_SWEEP}\n")

    results = run_sweep(G, lam, dA)
    jensen  = run_jensen(G, lam, dA)

    # Table 1: planner comparison (reads with Figures A-C)
    print("Table 1 - Planner comparison")
    print(f"{'s_scale':>7} {'s_large':>8} {'same':>5} | "
          f"{'nom_mean':>9} {'nom_q05':>9} | {'risk_mean':>9} {'risk_q05':>9} | {'gap_q05':>9}")
    print("-" * 80)
    for r in results:
        gap = r["risk"]["q05"] - r["nom"]["q05"]
        print(f"{r['scale']:7.1f} {r['sigma_large']:8.4f} {str(r['same_placement']):>5} | "
              f"{r['nom']['mean']:9.5f} {r['nom']['q05']:9.5f} | "
              f"{r['risk']['mean']:9.5f} {r['risk']['q05']:9.5f} | {gap:+9.5f}")

    # Table 2: Jensen-gap validation, fixed placement (reads with Figure D)
    print(f"\nTable 2 - Jensen-gap validation (fixed placement, B^2={jensen['B2']:.4f})")
    print(f"{'sigma^2':>10} {'realized_gap':>13} {'SEM':>10} {'predicted':>12} {'ratio':>8}")
    print("-" * 56)
    for s2, g, e, p in zip(jensen["sig2"], jensen["gap"], jensen["sem"], jensen["pred"]):
        ratio = g / p if p > 0 else float("nan")
        print(f"{s2:10.3e} {g:13.3e} {e:10.2e} {p:12.3e} {ratio:8.2f}")

    # run_checks(G, lam, dA, results)
    make_figures(results, jensen, lam, cands)
    print("\nFigures: fig_a_placements, fig_b_distributions, "
          "fig_c_downside, fig_d_jensen (.png)")


if __name__ == "__main__":
    main()