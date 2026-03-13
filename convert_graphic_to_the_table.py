# -*- coding: utf-8 -*-
"""
Muon Energy Loss Table -- T9 / BL4S
p = 5 GeV/c
Standalone table -- no plot.
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

RNG       = np.random.default_rng(42)
MUON_MASS = 105.66
ME        = 0.511
K         = 0.307075

class Material:
    def __init__(self, name, Z, A, rho, I_eV,
                 s_a, s_k, s_x0, s_x1, s_Cbar, s_d0,
                 X0_cm, kB=0.0):
        self.name = name
        self.Z = Z; self.A = A; self.rho = rho
        self.I = I_eV * 1e-6
        self.s_a=s_a; self.s_k=s_k; self.s_x0=s_x0
        self.s_x1=s_x1; self.s_Cbar=s_Cbar; self.s_d0=s_d0
        self.X0_cm = X0_cm
        self.kB    = kB

MAT = {
    "Fe":    Material("Fe",    26,55.85, 7.87,  286.0,
                      0.14680,2.9632,-0.0012,3.1531,4.2911,0.12, 1.757),
    "Al":    Material("Al",    13,26.98, 2.70,  166.0,
                      0.08024,3.6345, 0.1708,3.0127,4.2395,0.12, 8.897),
    "Scint": Material("Scint",  6,12.01, 1.032,  64.7,
                      0.16101,3.2393, 0.1464,2.4855,3.1997,0.00,42.4, 0.0126),
    "Air":   Material("Air",    7,14.5,  1.2e-3, 85.7,
                      0.10914,3.3994, 1.7418,4.2759,11.948,0.00,30420.),
}

def _kinematics(KE, mass):
    E  = KE + mass
    p  = np.sqrt(max(E**2 - mass**2, 1e-12))
    bg = p / mass
    beta = bg / np.sqrt(1 + bg**2)
    return bg, beta, np.sqrt(1 + bg**2)

def sternheimer(bg, mat):
    x = np.log10(bg)
    if   x < mat.s_x0: return mat.s_d0 * 10**(2*(x - mat.s_x0))
    elif x < mat.s_x1: return 2*np.log(10)*x - mat.s_Cbar + mat.s_a*(mat.s_x1-x)**mat.s_k
    else:               return 2*np.log(10)*x - mat.s_Cbar

def shell_corr(beta, mat):
    eta = beta / (7.3e-3 * mat.Z**(1/3))
    if eta > 10: return 0.0
    return max(0.422377e-6*eta**-2 + 0.0304043e-6*eta**-4
               - 0.00038106e-6*eta**-6, 0.0)

def wmax(bg, mass):
    gam = np.sqrt(1 + bg**2)
    return 2*ME*bg**2 / (1 + 2*gam*ME/mass + (ME/mass)**2)

def dEdx_ionisation(KE, mass, mat):
    if mat.rho < 1e-6 or KE <= 0: return 0.0
    bg, beta, _ = _kinematics(KE, mass)
    Wm = wmax(bg, mass)
    if Wm <= mat.I or beta < 1e-6: return 0.0
    delta = sternheimer(bg, mat)
    shell = shell_corr(beta, mat)
    return max(K*mat.Z/mat.A*mat.rho/beta**2 *
               (0.5*np.log(2*ME*bg**2*Wm/mat.I**2)
                - beta**2 - delta/2 - shell), 0.0)

def dEdx_brems(KE, mass, mat):
    if mat.X0_cm <= 0 or mat.rho < 1e-6: return 0.0
    return max((KE + mass) * (ME/mass)**2 * mat.rho / mat.X0_cm, 0.0)

def dEdx_total(KE, mass, mat):
    return dEdx_ionisation(KE, mass, mat) + dEdx_brems(KE, mass, mat)

def landau_sample(KE, mass, mat, dx):
    if mat.rho < 1e-4 or dx <= 0 or KE <= 0: return 0.0
    bg, beta, _ = _kinematics(KE, mass)
    xi  = K/2 * mat.Z/mat.A * mat.rho / beta**2 * dx
    Wm  = wmax(bg, mass)
    if Wm <= mat.I or xi <= 0: return 0.0
    delta = sternheimer(bg, mat)
    mpv = max(xi*(np.log(2*ME*bg**2*xi/mat.I**2)+0.2-beta**2-delta), 0.0)
    kappa = xi / Wm
    if kappa > 10:
        sample = RNG.normal(mpv, np.sqrt(xi*Wm*(1-beta**2/2)))
    else:
        u = RNG.uniform(1e-6, 1-1e-6)
        sample = mpv + xi*(-np.log(-np.log(u))-0.5772)*0.8
    return float(np.clip(sample, dEdx_total(KE,mass,mat)*dx*0.01, KE*0.999))

def birks(dE, dEdx_val, mat):
    if mat.kB <= 0: return dE
    return dE / (1 + mat.kB * dEdx_val)

def propagate(KE0, mass, mat, thickness):
    if mat.rho < 1e-4 or thickness <= 0 or KE0 < 0.1:
        return KE0, 0.0, 0.0, 0.0
    def rhs(x, y):
        return [0.0] if y[0] < 0.1 else [-dEdx_total(y[0], mass, mat)]
    sol = integrate.solve_ivp(rhs, [0, thickness], [KE0],
                               method="RK45", rtol=1e-9, atol=1e-9,
                               max_step=0.005, dense_output=True)
    dE_mean   = KE0 - float(np.maximum(sol.sol(np.array([thickness]))[0], 0.0)[0])
    dE_landau = landau_sample(KE0, mass, mat, thickness)
    KE_out    = max(KE0 - dE_landau, 0.0)
    dE_birks  = birks(dE_landau, dEdx_total(KE0, mass, mat), mat)
    return KE_out, dE_mean, dE_landau, dE_birks

# ── Setup ─────────────────────────────────────────────────
p_GeV = 5.0
KE0   = np.sqrt((p_GeV*1000)**2 + MUON_MASS**2) - MUON_MASS

SETUP = [
    ("Trigger S0",      "Scint", 0.5),
    ("Gap",             "Air",   5.0),
    ("Fe Filter",       "Fe",   15.0),
    ("Gap",             "Air",   3.0),
    ("Cherenkov (Air)", "Air",  20.0),
    ("Gap",             "Air",   2.0),
    ("Scint X",         "Scint", 1.0),
    ("Gap",             "Air",   2.0),
    ("Al Absorber",     "Al",    8.0),
    ("Gap",             "Air",   2.0),
    ("Scint Y",         "Scint", 1.0),
    ("Gap",             "Air",   3.0),
]

stages = []
KE = KE0
for label, mat_name, thick in SETUP:
    mat   = MAT[mat_name]
    KE_in = KE
    KE_out, dE_mean, dE_lnd, dE_brk = propagate(KE, MUON_MASS, mat, thick)
    stages.append({
        "label": label, "mat": mat_name, "thick": thick,
        "KE_in": KE_in, "KE_out": KE_out,
        "dE_mean": dE_mean, "dE_landau": dE_lnd, "dE_birks": dE_brk,
    })
    KE = KE_out

active     = [s for s in stages if s["mat"] != "Air"]
total_loss = KE0 - KE

# ── Table ─────────────────────────────────────────────────
MAT_COLOR = {"Fe": "#445566", "Al": "#445566", "Scint": "#445566"}
HDR_COL   = "#445566"
ALT_COL   = "#f4f7ff"

col_labels = ["Component", "Material", "Thickness\n(cm)",
              "KE in\n(MeV)", "KE out\n(MeV)",
              "ΔE mean\n(MeV)", "ΔE Landau\n(MeV)", "ΔE Birks\n(MeV)"]
col_w = [0.20, 0.10, 0.10, 0.11, 0.11, 0.12, 0.13, 0.13]

fig, ax = plt.subplots(figsize=(14, len(active)*0.65 + 1.6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.axis("off")

ax.text(0.5, 0.985,
        f"Muon Energy Loss -- T9 / BL4S   (p = {p_GeV} GeV/c)",
        ha="center", va="top", fontsize=13,
        fontfamily="monospace", fontweight="bold", color="#222222",
        transform=ax.transAxes)

# Column x positions
xs = [sum(col_w[:i]) for i in range(len(col_w))]
xs = [x / sum(col_w) for x in xs]

row_h  = 0.76 / (len(active) + 1.5)
y_hdr  = 0.88

# Header row
for i, (lbl, x, w) in enumerate(zip(col_labels, xs, col_w)):
    ax.add_patch(plt.Rectangle(
        (x, y_hdr - row_h), w/sum(col_w), row_h,
        transform=ax.transAxes, color=HDR_COL, zorder=2))
    ax.text(x + (w/sum(col_w))/2, y_hdr - row_h/2, lbl,
            ha="center", va="center", fontsize=8.5,
            fontfamily="monospace", fontweight="bold",
            color="white", transform=ax.transAxes, zorder=3)

# Data rows
for ri, s in enumerate(active):
    y_row = y_hdr - row_h * (ri + 2)
    bg    = ALT_COL if ri % 2 == 0 else "white"
    mc    = MAT_COLOR[s["mat"]]
    birks_str = f"{s['dE_birks']:.3f}" if s["mat"] == "Scint" else "--"

    row_vals = [
        s["label"], s["mat"], f"{s['thick']:.1f}",
        f"{s['KE_in']:.2f}", f"{s['KE_out']:.2f}",
        f"{s['dE_mean']:.4f}", f"{s['dE_landau']:.4f}", birks_str,
    ]

    for ci, (val, x, w) in enumerate(zip(row_vals, xs, col_w)):
        nw = w / sum(col_w)
        # background
        ax.add_patch(plt.Rectangle(
            (x, y_row), nw, row_h,
            transform=ax.transAxes,
            color=bg,
            zorder=1))
        # border
        ax.add_patch(plt.Rectangle(
            (x, y_row), nw, row_h,
            transform=ax.transAxes,
            fill=False, edgecolor="#ccccdd", lw=0.5, zorder=2))
        # text
        highlight = ci >= 6
        tc = "#222222"
        fw = "bold" if highlight else "normal"
        ax.text(x + nw/2, y_row + row_h/2, val,
                ha="center", va="center", fontsize=8.5,
                fontfamily="monospace", color=tc, fontweight=fw,
                transform=ax.transAxes, zorder=3)

# Total row
y_tot = y_hdr - row_h * (len(active) + 2)
ax.add_patch(plt.Rectangle(
    (0, y_tot), 1.0, row_h,
    transform=ax.transAxes, color=HDR_COL, alpha=0.85, zorder=2))
ax.text(0.02, y_tot + row_h/2, "TOTAL LOSS",
        ha="left", va="center", fontsize=9,
        fontfamily="monospace", fontweight="bold",
        color="white", transform=ax.transAxes, zorder=3)
ax.text(0.82, y_tot + row_h/2, f"{total_loss:.4f} MeV",
        ha="center", va="center", fontsize=9,
        fontfamily="monospace", fontweight="bold",
        color="white", transform=ax.transAxes, zorder=3)

plt.tight_layout()
plt.savefig("muon_energy_table.png", dpi=180,
            bbox_inches="tight", facecolor="white")
print("Saved: muon_energy_table.png")
plt.show()
