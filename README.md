# pIKKT4D Hybrid Monte Carlo

Python/Torch implementation of Hybrid Monte Carlo for the polarized $D=4$ IKKT (IIB) matrix model (see [A. Martina, arXiv:2507.17813](https://arxiv.org/pdf/2507.17813)). The code cleanly separates the model-specific action and observables (`model.py`) from the HMC integrator (`hmc.py`).

## Model

The four Hermitian matrices $X_I$ (with $I=1\ldots4$) and their fermionic partners $\psi$ are evolved with two supersymmetric mass deformations

**Type I, $SO(4)$ invariant, single coupling constant $\lambda$**

$$
S_{D=4,\text{type I}}=\frac{1}{\lambda}\,\text{Tr}\Bigl[-\frac14 [X_I,X_J]^2 -\frac{i}{2}\bar\psi \Gamma^I [X_I,\psi] + X_I^2 + \bar\psi\psi \Bigr]
$$

**Type II, $SO(3)$ invariant, two coupling constants $\lambda$ and $\omega$**

$$
\begin{aligned}
S_{D=4,\text{type II}}=\frac{1}{\lambda}\,\text{Tr}\Bigl[&-\frac14 [X_I,X_J]^2 -\frac{i}{2}\bar\psi \Gamma^I [X_I,\psi] + i \frac{2}{3}(1+\omega)\,\epsilon_{ijk} X_i X_j X_k \\
&+ \frac{1}{3}\left(\omega + \frac{2}{3}\right) X_i X_i + \frac{\omega}{3} X_4^2 - \frac{1}{3} \bar\psi \Gamma^{123} \psi \Bigr]
\end{aligned}
$$

In this codebase the couplings and the type of mass deformation map to `omega`, `coupling` and `pIKKT_type` in `ModelParams`:
- `pIKKT_type=1` selects the Type I action.
- `pIKKT_type=2` selects the Type II action.

Fermion determinants are evaluated via the reduced Majorana/Weyl matrices in `model.py`.

## Setup

The project uses Python 3 with PyTorch. Ensure `torch` is installed with CUDA support if available; otherwise it will fall back to CPU (significantly slower!). No extra dependencies are required beyond NumPy and Matplotlib for analysis.

## Usage

With this code, we can run $N=45$ with $400$ steps in under $2$ hours on `NVIDIA GeForce RTX 2080 Ti`.

Basic run (Type I, $N=10,\ \lambda=100$). Outputs stored to `outputs/eigenvalues_myrun_*.dat` and `outputs/comm2_myrun_*.dat`:

```bash
python main.py --pIKKT-type 1 --ncol 10 --niters 300 --coupling 100.0 --fresh --name myrun --data-path outputs
```

Type II with $N=10,\ \omega = 1$ and $\lambda=100$:

```bash
python main.py --pIKKT-type 2 --omega 1.0 --ncol 10 --niters 300 --coupling 100.0 --fresh --name myrun --data-path outputs
```

Resume from a checkpoint:

```bash
python main.py --resume --name myrun --data-path outputs
```

Key flags (see `cli.py` for defaults):
- `--pIKKT-type {1,2}`: choose model variant.
- `--ncol N`: matrix size \(N\).
- `--niters K`: HMC trajectories to run.
- `--step-size`, `--nsteps`: leapfrog integrator controls ($\Delta t$ and steps).
- `--omega`: anisotropy/cubic coupling (used for type II).
- `--save`, `--save-every`: checkpoint cadence.
- `--data-path`: directory for outputs/checkpoints.
- `--seed`: RNG seed for deterministic runs.
- `--spin`: initialize with a fuzzy-sphere background of spin $j$ (optional).

Outputs:
- Eigenvalues and correlators are written to `data_path` with `eigenvalues_*.dat` and `comm2_*.dat`.
- Checkpoints are stored as `config_*.pt`.

## Code structure

- `main.py` — CLI entry point, I/O, thermalization, and trajectory loop.
- `cli.py` — argument parsing and validation.
- `model.py` — action, fermion determinants, observables, spin backgrounds.
- `hmc.py` — model-agnostic leapfrog integrator and Metropolis step.
- `algebra.py` — Hermitian/traceless projections and adjoint actions.  
