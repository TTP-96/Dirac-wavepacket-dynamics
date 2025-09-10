# 1+1D Dirac equation (ħ = c = 1) 2 choices of potential (step, barrier) and option for free evolution
# i dt psi = [ -i α d/dx + beta m + V(x) I ] psi,  α = sigmax, beta = sigmaz, psi(x,t) \in C^2
# Split-operator Fourier transform (SOFT) scheme: K(dt/2) -> M+V(dt) -> K(dt/2)
# Klein tunneling shows up for large step heights V0 >= 2 m.

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# CLI arguments

parser = argparse.ArgumentParser(description="1D Dirac wavepacket + Klein tunneling (split-step FFT)")
parser.add_argument("--N", type=int, default=4096, help="grid points (2^n preferable)")
parser.add_argument("--L", type=float, default=400.0, help="domain length")
parser.add_argument("--dt", type=float, default=0.02, help="time step")
parser.add_argument("--steps", type=int, default=2000, help="total steps")
parser.add_argument("--plot-every", type=int, default=2, help="animation update stride")
parser.add_argument("--m", type=float, default=1.0, help="mass m")
parser.add_argument("--x0", type=float, default=-120.0, help="initial packet center")
parser.add_argument("--sigma", type=float, default=10.0, help="initial packet width")
parser.add_argument("--p0", type=float, default=3.0, help="initial mean momentum")
parser.add_argument("--scenario", choices=["free","step","barrier"], default="step",
                    help="free evolution, smooth step (Klein), or finite barrier")
parser.add_argument("--V0", type=float, default=3.0, help="step/barrier height")
parser.add_argument("--xstep", type=float, default=0.0, help="step position (or barrier center)")
parser.add_argument("--smooth", type=float, default=5.0, help="step/barrier smoothness (tanh width)")
parser.add_argument("--barrier-width", type=float, default=30.0, help="finite barrier half-width")
parser.add_argument("--pos-energy", action="store_true",
                    help="initialize spinor as positive-energy eigenspinor at p0 (less Zitterbewegung)")
args = parser.parse_args()


# Grid and ops

N = args.N
L = args.L
dx = L / N
x  = (np.arange(N) - N//2) * dx
k  = 2.0*np.pi * np.fft.fftfreq(N, d=dx)  # momentum space grid 

m = args.m
dt = args.dt
plot_every = args.plot_every

# Kinetic part half-step: U_K = exp(-i alpha*p*delta_t/2) = cos(a) I - i sin(a)*alpha, with a = p Deltat/2
a_half = 0.5 * k * dt
cos_a  = np.cos(a_half)
sin_a  = np.sin(a_half)

# Mass part (x-space, diagonal in spinor basis of beta): U_M = exp(-i beta*m*delta_t)
phase_plus  = np.exp(-1j*m*dt)   # upper spinor component (beta=+1)
phase_minus = np.exp(+1j*m*dt)   # for lower spinor component (beta=-1)

xm_list = []  # track expval of x to plot at end of sim

# Pots

def V_step(x, V0=3.0, x0=0.0, s=5.0):
    # Smooth step 
    return 0.5 * V0 * (1.0 + np.tanh((x - x0)/s))

def V_barrier(x, V0=3.0, x0=0.0, s=5.0, w=30.0):
    # Smooth top-hat barrier built from two tanhs
    left  = 0.5*(1.0 + np.tanh((x - (x0 - w))/s))
    right = 0.5*(1.0 - np.tanh((x - (x0 + w))/s))
    return V0*left*right

if args.scenario == "free":
    Vx = np.zeros_like(x)
elif args.scenario == "step":
    Vx = V_step(x, V0=args.V0, x0=args.xstep, s=args.smooth)
else:
    Vx = V_barrier(x, V0=args.V0, x0=args.xstep, s=args.smooth, w=args.barrier_width)

# Potential full-step: U_V = exp(-i V(x) Deltat) * I_2
phase_V = np.exp(-1j * Vx * dt)


# Initial state

def pos_energy_spinor(p, m):
    # Positive-energy eigenspinor of H(p)=alpha*p +beta*m with alpha=sigmax, beta=sigmaz
    # E = sqrt(p^2 + m^2), u propto the 4-vec [E+m, p]^T
    E = np.sqrt(p**2 + m**2)
    u0 = E + m
    u1 = p
    # normalize
    norm = np.sqrt(np.abs(u0)**2 + np.abs(u1)**2)
    return np.array([u0, u1], dtype=complex) / norm

x0= args.x0
sigma = args.sigma
p0  = args.p0

envelope = np.exp(-0.5*((x - x0)/sigma)**2) * np.exp(1j*p0*x)
envelope /= np.sqrt(np.sum(np.abs(envelope)**2) * dx)

if args.pos_energy:
    s = pos_energy_spinor(p0, m)
else:
    s = np.array([1.0+0j, 0.0+0j])  # shows Zitterbewegung clearly

psi = np.vstack([s[0]*envelope, s[1]*envelope])  #  (2,N)


# Time evolution 

def kinetic_half_step(psi_x):
    u_k = np.fft.fft(psi_x[0])
    v_k = np.fft.fft(psi_x[1])
    # Apply U_K: [u'; v'] = cos(a)[u; v] - i sin(a) α [u; v] = cos(a)[u; v] - i sin(a) [v; u]
    u_kp = cos_a * u_k - 1j * sin_a * v_k
    v_kp = cos_a * v_k - 1j * sin_a * u_k
    u_x  = np.fft.ifft(u_kp)
    v_x  = np.fft.ifft(v_kp)
    return np.vstack([u_x, v_x])

def pot_full_step(psi_x):
    # U_{M+V} = e^{-i V dt} * e^{-i beta m dt}; [commute since both diagonal in spinor basis]
    psi_x[0] *= phase_V * phase_plus
    psi_x[1] *= phase_V * phase_minus
    return psi_x

def step():
    global psi, t
    psi = kinetic_half_step(psi)
    psi = pot_full_step(psi)
    psi = kinetic_half_step(psi)
    t += dt


# Helpers 

def density(psi_x):
    return (np.abs(psi_x[0])**2 + np.abs(psi_x[1])**2).real

def ex_x(psi_x):
    ρ = density(psi_x)
    return (np.sum(x*ρ)*dx) / (np.sum(ρ)*dx)

def reflect_transmit(psi_x, xcut, margin=10.0):
    # Integrate away from step to avoid the interface region artifacts
    ρ = density(psi_x)
    left  = x < (xcut - margin)
    right = x > (xcut + margin)
    ρtot = np.sum(ρ)*dx
    R = np.sum(ρ[left])*dx / ρtot # The reflection and transmission coefficients are just
    T = np.sum(ρ[right])*dx / ρtot # the integrated density on each side of the step, normalized to total
    return R, T


# Animate
fig, ax = plt.subplots(figsize=(9.5,4.8))
line_tot, = ax.plot([], [], lw=1.6, label=r'$|\psi|^2$')
line_u,   = ax.plot([], [], lw=1.0, alpha=0.85, label=r'$|\psi_1|^2$')
line_v,   = ax.plot([], [], lw=1.0, alpha=0.85, label=r'$|\psi_2|^2$')

# Plot potential scaled for reference
Vscale = max(1e-12, np.max(Vx))
if Vscale > 0:
    Vref = (Vx / Vscale) * 0.8 * np.max(density(psi))
    line_V, = ax.plot(x, Vref, ls='--', lw=1.0, alpha=0.7, label=f'V(x) scaled (÷{Vscale:.2g})')
else:
    line_V = None

ax.set_xlim(x[0], x[-1])
ax.set_ylim(0, max(density(psi).max()*1.3, 1e-3))
ax.set_xlabel("x")
ax.set_ylabel("ρ")
ax.legend(loc='upper right')

title = ax.text(0.01, 0.98, "Title", transform=ax.transAxes, va='top', ha='left')

# Vertical line at step center
if args.scenario in ("step","barrier"):
    ax.axvline(args.xstep, ls=':', lw=0.9, alpha=0.7)

t = 0.0
def init():
    line_tot.set_data([], [])
    line_u.set_data([], [])
    line_v.set_data([], [])
    if line_V is not None:
        line_V.set_ydata((Vx / Vscale) * 0.8 * np.max(density(psi)))

    return (line_tot, line_u, line_v) if line_V is None else (line_tot, line_u, line_v, line_V)



def update(frame):

    for _ in range(plot_every):
        step()


    ρ = density(psi)
    line_tot.set_data(x, ρ)
    line_u.set_data(x, np.abs(psi[0])**2)
    line_v.set_data(x, np.abs(psi[1])**2)

    xm = ex_x(psi)
    # track expval of x to plot at end of sim
    xm_list.append(xm)
   
    norm = np.sum(ρ)*dx

    if args.scenario in ["step","barrier"]:

        R, T = reflect_transmit(psi, xcut=args.xstep, margin=max(10.0, args.smooth))
        title.set_text(f"{args.scenario.upper()} | t={t:6.3f}  ⟨x⟩={xm:7.3f}  norm={norm:.6f}  R={R:.3f}  T={T:.3f}")
    else:
        title.set_text(f"FREE | t={t:6.3f}  ⟨x⟩={xm:7.3f}  norm={norm:.6f}")

    return (line_tot, line_u, line_v) if line_V is None else (line_tot, line_u, line_v, line_V)

ani = FuncAnimation(fig, update, init_func=init, frames=args.steps//plot_every, interval=20, blit=False)
plt.tight_layout()
plt.show()

# Plot expval of x vs t at end of sim
if len(xm_list) > 1:
    fig2, ax2 = plt.subplots(figsize=(6,4.5))
    tlist = np.arange(len(xm_list)) * dt * plot_every
    ax2.plot(tlist, xm_list, lw=1.4)
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"⟨x⟩")
    ax2.set_title("Expectation value of position vs time")
    plt.tight_layout()
    plt.show()

