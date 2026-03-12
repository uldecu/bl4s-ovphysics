"""
Muon Energy Loss — T9 / BL4S
p = 5 GeV/c

Physics active during propagation:
  - Bethe-Bloch (full, with Sternheimer density + shell correction)
  - Bremsstrahlung  → added to dE/dx in ODE, shifts KE curve down
  - Landau          → stochastic fluctuation sampled per layer,
                      KE curve shows one realisation (MPV track)
  - Birks quenching → scintillator KE_out corrected to measured signal
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

RNG       = np.random.default_rng(42)
MUON_MASS = 105.66    # MeV/c²
ME        = 0.511     # MeV
K         = 0.307075  # MeV cm²/mol

# ── Materials ─────────────────────────────────────────────
class Material:
    def __init__(self, name, Z, A, rho, I_eV,
                 s_a, s_k, s_x0, s_x1, s_Cbar, s_d0,
                 X0_cm, kB=0.0):
        self.name = name
        self.Z = Z; self.A = A; self.rho = rho
        self.I = I_eV * 1e-6          # eV → MeV
        self.s_a=s_a; self.s_k=s_k; self.s_x0=s_x0
        self.s_x1=s_x1; self.s_Cbar=s_Cbar; self.s_d0=s_d0
        self.X0_cm = X0_cm
        self.kB    = kB               # Birks constant g/(cm²·MeV)

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

# ── Physics ───────────────────────────────────────────────

def _kinematics(KE, mass):
    E    = KE + mass
    p    = np.sqrt(max(E**2 - mass**2, 1e-12))
    bg   = p / mass
    beta = bg / np.sqrt(1 + bg**2)
    gam  = np.sqrt(1 + bg**2)
    return bg, beta, gam

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
    """Bethe-Bloch with Sternheimer δ and shell correction."""
    if mat.rho < 1e-6 or KE <= 0: return 0.0
    bg, beta, gam = _kinematics(KE, mass)
    Wm = wmax(bg, mass)
    if Wm <= mat.I or beta < 1e-6: return 0.0
    delta = sternheimer(bg, mat)
    shell = shell_corr(beta, mat)
    return max(K*mat.Z/mat.A*mat.rho/beta**2 *
               (0.5*np.log(2*ME*bg**2*Wm/mat.I**2)
                - beta**2 - delta/2 - shell), 0.0)

def dEdx_brems(KE, mass, mat):
    """
    Bremsstrahlung for muons: dE/dx_rad ≈ E·(me/mμ)²·ρ/X0
    Active in ODE — shifts the KE curve.
    """
    if mat.X0_cm <= 0 or mat.rho < 1e-6: return 0.0
    E = KE + mass
    return max(E * (ME/mass)**2 * mat.rho / mat.X0_cm, 0.0)

def dEdx_total(KE, mass, mat):
    return dEdx_ionisation(KE, mass, mat) + dEdx_brems(KE, mass, mat)

def landau_sample(KE, mass, mat, dx):
    """
    Sample one Landau-fluctuated energy loss for layer dx [cm].
    Uses Vavilov κ-parameter to decide regime:
      κ < 0.01 → Landau limit  (thin layers)
      κ > 10   → Gaussian limit (thick layers)
      otherwise → interpolate
    Returns sampled ΔE [MeV].
    """
    if mat.rho < 1e-4 or dx <= 0 or KE <= 0:
        return 0.0
    bg, beta, _ = _kinematics(KE, mass)
    xi   = K/2 * mat.Z/mat.A * mat.rho / beta**2 * dx   # MeV
    Wm   = wmax(bg, mass)
    if Wm <= mat.I or xi <= 0:
        return 0.0
    delta = sternheimer(bg, mat)

    # MPV (most probable value)
    mpv = xi * (np.log(2*ME*bg**2*xi/mat.I**2) + 0.2 - beta**2 - delta)
    mpv = max(mpv, 0.0)

    # Vavilov kappa
    kappa = xi / Wm

    if kappa > 10:
        # Gaussian regime
        sigma_g = np.sqrt(xi * Wm * (1 - beta**2/2))
        sample  = RNG.normal(mpv, sigma_g)
    else:
        # Landau regime: sample from Landau distribution
        # Landau PDF peak at λ=−0.222, FWHM ≈ 4xi
        # We use the standard trick: sample λ from Landau(0,1)
        # then ΔE = MPV + xi * λ
        # Landau(0,1) sample via inverse CDF approximation (Moyal)
        u = RNG.uniform(1e-6, 1-1e-6)
        # Moyal approximation to Landau quantile
        lam = -np.log(-np.log(u)) - 0.5772
        sample = mpv + xi * lam * 0.8   # 0.8 empirical scale

    # Clamp: can't lose more than KE, can't gain energy
    mean_dE = dEdx_total(KE, mass, mat) * dx
    return float(np.clip(sample, mean_dE * 0.01, KE * 0.999))

def birks(dE, dEdx_val, mat):
    """
    Birks' Law: measured scint signal = dE / (1 + kB·dE/dx).
    Only active for scintillator (kB > 0).
    Returns corrected (measured) energy deposit.
    """
    if mat.kB <= 0:
        return dE
    return dE / (1 + mat.kB * dEdx_val)

# ── Propagation ───────────────────────────────────────────

def propagate(KE0, mass, mat, thickness):
    """
    Propagate muon through layer of given thickness.

    Returns:
        KE_out    : kinetic energy after layer [MeV]
                    (Landau-fluctuated for scint/Fe/Al;
                     Birks-corrected for scintillator)
        x_arr     : depth array [cm]
        KE_arr    : KE profile along depth [MeV]
        dE_mean   : mean energy loss (ODE, no fluctuation) [MeV]
        dE_landau : Landau-sampled loss (what actually happened) [MeV]
        dE_birks  : Birks-corrected measured signal [MeV]
                    (= dE_landau for non-scintillator)
    """
    if mat.rho < 1e-4 or thickness <= 0 or KE0 < 0.1:
        x = np.linspace(0, thickness, 10)
        return KE0, x, np.full(10, KE0), 0.0, 0.0, 0.0

    # ── Mean ODE (Bethe-Bloch + Bremsstrahlung) ──────────
    def rhs(x, y):
        KE = y[0]
        if KE < 0.1: return [0.0]
        return [-dEdx_total(KE, mass, mat)]

    sol = integrate.solve_ivp(
        rhs, [0, thickness], [KE0],
        method="RK45", rtol=1e-9, atol=1e-9,
        max_step=0.005, dense_output=True)

    x_arr   = np.linspace(0, thickness, max(int(thickness/0.02), 50))
    KE_mean = np.maximum(sol.sol(x_arr)[0], 0.0)
    dE_mean = KE0 - float(KE_mean[-1])

    # ── Landau fluctuation sampled for whole layer ────────
    dE_landau = landau_sample(KE0, mass, mat, thickness)
    KE_landau = max(KE0 - dE_landau, 0.0)

    # ── Build displayed KE profile:
    #    scale the mean curve so its endpoint matches Landau sample
    if dE_mean > 0:
        scale   = dE_landau / dE_mean
        KE_disp = KE0 - (KE0 - KE_mean) * scale
    else:
        KE_disp = KE_mean.copy()
    KE_disp = np.maximum(KE_disp, 0.0)

    # ── Birks correction (scintillator only) ─────────────
    dEdx_entry = dEdx_total(KE0, mass, mat)
    dE_birks   = birks(dE_landau, dEdx_entry, mat)

    # KE_out: what the particle actually has after the layer
    # For scintillator: particle loses dE_landau, but detector
    # only SEES dE_birks — KE_out is still the physical value
    KE_out = KE_landau

    return KE_out, x_arr, KE_disp, dE_mean, dE_landau, dE_birks

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

z_pts, KE_pts, stages = [], [], []
z, KE = 0.0, KE0

for label, mat_name, thick in SETUP:
    mat   = MAT[mat_name]
    KE_in = KE
    KE_out, x_loc, KE_loc, dE_mean, dE_lnd, dE_brk = \
        propagate(KE, MUON_MASS, mat, thick)
    z_pts.extend((z + x_loc).tolist())
    KE_pts.extend(KE_loc.tolist())
    stages.append({
        "z0": z, "z1": z+thick, "label": label,
        "mat": mat_name, "thick": thick,
        "KE_in": KE_in, "KE_out": KE_out,
        "dE_mean": dE_mean, "dE_landau": dE_lnd,
        "dE_birks": dE_brk,
    })
    z  += thick
    KE  = KE_out

z_arr      = np.array(z_pts)
KE_arr     = np.array(KE_pts)
total_loss = KE0 - KE

active = [s for s in stages if s["mat"] != "Air"]

# ── Terminal report ────────────────────────────────────────
print(f"\n  p = {p_GeV} GeV/c  |  KE₀ = {KE0:.2f} MeV\n")
print(f"  {'Component':<14} {'mat':>5} {'thick':>6}  "
      f"{'KE_in':>8} {'KE_out':>8}  "
      f"{'ΔE_mean':>8} {'ΔE_Landau':>10} {'ΔE_Birks':>10}")
print(f"  {'-'*82}")
for s in active:
    print(f"  {s['label']:<14} {s['mat']:>5} {s['thick']:>5.1f}cm  "
          f"{s['KE_in']:>8.2f} {s['KE_out']:>8.2f}  "
          f"{s['dE_mean']:>8.4f} {s['dE_landau']:>10.4f} "
          f"{s['dE_birks']:>10.4f}")
print(f"  {'-'*82}")
print(f"  {'TOTAL':<14}                        "
      f"                   {total_loss:>8.4f}\n")

# ── Colours & styles ──────────────────────────────────────
MAT_COLOR = {"Fe": "#4a90d9", "Al": "#ff6b35", "Scint": "#00d4ff"}
DARK  = "#030812"; PANEL = "#060e1e"; BORDER = "#0d2a4a"
DIM   = "#3a5a7a"; WHITE = "#e8f0f8"; YELLOW = "#ffff00"

# ── Figure ────────────────────────────────────────────────
fig  = plt.figure(figsize=(17, 6.5), facecolor=DARK)
ax   = fig.add_axes([0.05, 0.12, 0.57, 0.78])
ax_t = fig.add_axes([0.65, 0.08, 0.33, 0.86])
ax_t.set_facecolor(PANEL); ax_t.set_xlim(0,1); ax_t.set_ylim(0,1)
ax_t.set_xticks([]); ax_t.set_yticks([])
for sp in ax_t.spines.values():
    sp.set_edgecolor(BORDER); sp.set_linewidth(1.2)
ax.set_facecolor(DARK)

y_min = KE_arr[-1] - 50
y_max = KE0 + 80
ax.set_ylim(y_min, y_max)
ax.set_xlim(z_arr[0] - 1, z_arr[-1] + 1)

# Shading + labels
for i, s in enumerate(active):
    col  = MAT_COLOR[s["mat"]]
    zmid = (s["z0"] + s["z1"]) / 2
    ax.axvspan(s["z0"], s["z1"], color=col, alpha=0.15, zorder=1)
    ax.axvline(s["z0"], color=col, lw=0.8, alpha=0.5, ls="--", zorder=2)
    ax.axvline(s["z1"], color=col, lw=0.8, alpha=0.5, ls="--", zorder=2)
    y_lbl = y_max - 5 - (i % 2) * 14
    ax.text(zmid, y_lbl, s["label"],
            ha="center", va="top", fontsize=9,
            color=col, fontfamily="monospace", fontweight="bold",
            bbox=dict(facecolor=DARK, edgecolor="none", pad=1.5))

# KE curve (Landau-fluctuated + Bremsstrahlung)
ax.plot(z_arr, KE_arr, color=WHITE, lw=2.5, zorder=5)
ax.fill_between(z_arr, KE_arr, y_min, alpha=0.07, color=WHITE, zorder=4)

ax.set_title(
    f"Muon Energy Loss — T9 / BL4S   (p = {p_GeV} GeV/c)  "
    f"[Bremsstrahlung + Landau + Birks active]",
    color=WHITE, fontsize=10, fontfamily="monospace",
    fontweight="bold", pad=10)
ax.set_xlabel("z  (cm)", color=DIM, fontsize=10, fontfamily="monospace")
ax.set_ylabel("Kinetic Energy  (MeV)", color=DIM, fontsize=10,
              fontfamily="monospace")
ax.tick_params(colors=DIM, labelsize=8)
for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
ax.grid(color=BORDER, ls=":", alpha=0.35, zorder=0)

legend_items = [
    mpatches.Patch(color=MAT_COLOR["Fe"],    label="Fe Filter  (15 cm)"),
    mpatches.Patch(color=MAT_COLOR["Al"],    label="Al Absorber  (8 cm)"),
    mpatches.Patch(color=MAT_COLOR["Scint"], label="Scintillator"),
]
ax.legend(handles=legend_items, facecolor="#060e1e",
          edgecolor=BORDER, labelcolor="#cccccc",
          fontsize=8, loc="lower right", framealpha=0.95)

# ── Table ─────────────────────────────────────────────────
def tt(x, y, s, col=WHITE, size=7.8, fw="normal", ha="left"):
    ax_t.text(x, y, s, ha=ha, va="center", fontsize=size,
              color=col, fontfamily="monospace", fontweight=fw,
              transform=ax_t.transAxes)

def hl(y, lw=0.8):
    ax_t.plot([0.02, 0.98], [y, y],
              color=BORDER, lw=lw, transform=ax_t.transAxes)

tt(0.5, 0.965, "Energy Loss Summary — Full Physics",
   col=WHITE, size=9, fw="bold", ha="center")
hl(0.935, lw=1.2)
tt(0.5, 0.910, "BB + Sternheimer δ + Bremsstrahlung",
   col=DIM, size=6.8, ha="center")
tt(0.5, 0.891, "Landau fluctuations  |  Birks quenching",
   col=DIM, size=6.8, ha="center")
hl(0.872)

# Column headers
# Table has 6 columns: Component | Thick | KE_in | KE_out | ΔE_Landau | ΔE_Birks
cx   = [0.03, 0.30, 0.44, 0.56, 0.68, 0.82]
hdrs = ["Component", "Thick", "KE in", "KE out", "ΔE Landau", "ΔE Birks"]
for h, x in zip(hdrs, cx):
    tt(x, 0.847, h, col=DIM, size=7)
hl(0.825)

row_h = 0.093
for i, s in enumerate(active):
    y   = 0.825 - (i + 0.5) * row_h
    col = MAT_COLOR[s["mat"]]
    ax_t.fill_between([0.01, 0.99],
                      [y - row_h*0.46]*2, [y + row_h*0.46]*2,
                      color=col, alpha=0.08,
                      transform=ax_t.transAxes)
    # ΔE_Birks only meaningful for scintillator
    birks_str = (f"{s['dE_birks']:.3f}"
                 if s["mat"] == "Scint"
                 else "—")
    vals = [s["label"], f"{s['thick']:.1f}cm",
            f"{s['KE_in']:.1f}", f"{s['KE_out']:.1f}",
            f"{s['dE_landau']:.3f}", birks_str]
    for j, (v, x) in enumerate(zip(vals, cx)):
        highlight = j >= 4
        tc = col   if highlight else WHITE
        fw = "bold" if highlight else "normal"
        tt(x, y, v, col=tc, size=7.5, fw=fw)

y_sep = 0.825 - len(active) * row_h - 0.01
hl(y_sep)

y_tot = y_sep - row_h * 0.65
tt(cx[0], y_tot, "TOTAL", col=WHITE, size=9, fw="bold")
tt(cx[4], y_tot, f"{total_loss:.2f} MeV",
   col=YELLOW, size=9, fw="bold")

# Footer note
y_fn = y_tot - row_h * 1.1
tt(0.03, y_fn,
   "ΔE Landau: stochastic sample (1 event)",
   col=DIM, size=6.5)
tt(0.03, y_fn - 0.055,
   "ΔE Birks: scint measured signal (— = n/a)",
   col=DIM, size=6.5)
tt(0.5, 0.022, "All energies in MeV",
   col=DIM, size=6.8, ha="center")

plt.savefig("energy_loss_v2.png", dpi=150,
            bbox_inches="tight", facecolor=DARK)
print("Saved: energy_loss_v2.png")
plt.show()