# Neural Distribution

Machine learning models for **(un)conditional sampling**: learn a distribution function `p(x)` or a conditional
distribution `p(x | c)` from samples and draw new samples from it. Built with Keras 3 / TensorFlow.

The main use case is fitting distribution functions, for example velocity distributions that depend on a temperature,
or **scattering kernels** `K(v_i → v_r)` and **collision kernels** `q(g' | g)`, optionally with a physical
**detailed-balance loss**.

## Models

| Name (`build_model`) | Class | Conditional | Exact `log_prob` | Notes |
|---|---|---|---|---|
| `vae` | `VAE` | no | no (ELBO) | Gaussian latent space, MSE reconstruction |
| `cvae` | `cVAE` | yes | no (ELBO) | encoder and decoder receive the condition |
| `realnvp` | `RealNVP` | no | yes | affine coupling layers, needs `x_dim >= 2` |
| `crealnvp` | `cRealNVP` | yes | yes | every coupling network receives the condition |
| `crealnvp_db` | `cRealNVPDB` | yes | yes | `cRealNVP` + backward NLL + detailed-balance loss |
| `splineflow` | `SplineFlow` | no | yes | rational-quadratic spline coupling layers (neural spline flow), needs `x_dim >= 2` |
| `csplineflow` | `cSplineFlow` | yes | yes | every spline network receives the condition |
| `csplineflow_db` | `cSplineFlowDB` | yes | yes | `cSplineFlow` + backward NLL + detailed-balance loss |

The conditional models are thin subclasses of the unconditional ones: `cond_dim=0` means unconditional.

## Installation

```bash
git clone https://github.com/schuettem/NeuralDist.git
cd NeuralDist
pip install -e ".[notebooks]"      # add ".[dev]" for pytest and ruff, ".[all]" for both
```

If you run the notebooks, make sure the Jupyter kernel uses the same environment
(`python -m ipykernel install --user --name neuraldist`).

## Quick start

```python
import numpy as np
from neuraldist.generative_model.registry import build_model
import neuraldist.generative_model.vae       # registers "vae" and "cvae"
import neuraldist.generative_model.realnvp   # registers "realnvp", "crealnvp", and "crealnvp_db"
import neuraldist.generative_model.splineflow  # registers "splineflow", "csplineflow", and "csplineflow_db"
from neuraldist.training import fit_model
```

**Unconditional** (`p(x)`):

```python
flow = build_model("realnvp", x_dim=3, n_coupling=6)
fit_model(flow, x, epochs=100)              # x: array (N, 3)

flow.sample(1000)                           # (1000, 3)
flow.log_prob(x[:5])                        # (5,)
```

**Conditional** (`p(x | c)`), e.g. a velocity distribution that depends on the temperature:

```python
model = build_model("crealnvp", x_dim=3, cond_dim=1, n_coupling=6)
fit_model(model, v, T, epochs=100)          # v: (N, 3), T: (N, 1)

temps = np.array([[0.5], [1.0], [2.0]], dtype="float32")
model.sample(1000, cond=temps)              # (3, 1000, 3): 1000 samples per condition
model.log_prob(v[:5], T[:5])                # (5,)
```

**VAE / cVAE.** The networks are configured with lists of layer dicts; the output layers are derived from `x_dim`
and `latent_dim`:

```python
cvae = build_model(
    "cvae", x_dim=3, cond_dim=1, latent_dim=3, kl_weight=0.1,
    encoder_hidden_layers=[{"neurons": 64, "activation": "relu"}, {"neurons": 32, "activation": "relu"}],
)
fit_model(cvae, v, T, epochs=150, validation_split=0.2, seed=0)
cvae.sample(100, cond=temps)                # (3, 100, 3)
```

**Spline flows.** `splineflow` / `csplineflow` have the same interface as the RealNVP models (`forward`, `inverse`,
`log_prob`, `sample`). Each coupling layer maps the transformed dimensions with a monotonic rational-quadratic spline
(`n_bins` bins) on `[-tail_bound, tail_bound]` and is the identity outside, so standardize the data or choose
`tail_bound` to cover it. A spline coupling layer is much more expressive than an affine one, so fewer layers are
usually enough:

```python
flow = build_model("csplineflow", x_dim=3, cond_dim=1, n_coupling=4, n_bins=8, tail_bound=5.0)
fit_model(flow, v, T, epochs=100)
```

**Saving and loading.** Weights are stored with Keras. A new model with the **same architecture** must be built and
initialized with `warmup()` before loading:

```python
model.warmup()
model.save_weights("model.weights.h5")

restored = build_model("crealnvp", x_dim=3, cond_dim=1, n_coupling=6)
restored.warmup()
restored.load_weights("model.weights.h5")
```

## Conventions

- `sample(n_samples, cond=None)` returns `(n_samples, x_dim)` for unconditional models and
  `(B, n_samples, x_dim)` for a condition of shape `(B, cond_dim)`.
- A conditional model requires `cond`, an unconditional model rejects it (clear `ValueError`).
- Flows additionally provide `log_prob(x, cond=None)`, `forward` (data to latent) and `inverse` (latent to data).
- VAEs provide `encode` and `decode`.
- Training metrics are logged by Keras under the names in the model's `loss_names`
  (`loss`, `recon`, `kl` for VAEs, `loss` for flows).

## Training with `fit_model`

`neuraldist.training.fit_model` compiles and fits a model on arrays in one call. It chooses `(x,)` or `(x, cond)`
batches from the model type, checks all shapes, and returns the Keras `History`.

| Argument | Meaning |
|---|---|
| `epochs`, `batch_size`, `learning_rate`, `optimizer` | the usual training settings (default optimizer: Adam) |
| `validation_split`, `validation_data` | validation on a split of the arrays or on separate arrays |
| `callbacks`, `seed`, `verbose` | Keras callbacks, reproducible shuffling and split |
| `compile_model=False` | keep a model you have already compiled yourself |
| `compile_kwargs` | extra arguments for `model.compile` (e.g. `nll_bwd`, `db_loss`) |
| `db_data`, `db_batch_size` | detailed-balance points (see below) |

`x` may also be a ready-made `tf.data.Dataset`.

## Detailed balance

A transition kernel `q(z' | z)` that is in detailed balance with an equilibrium weight `W` satisfies

```
W(z) · q(z' | z)  =  W(z') · q(T z | T z')          (T: time reversal)
```

The DB losses penalize the squared difference of the logarithms of both sides. They can be evaluated on **arbitrary**
pairs `(z, z')`: they only need `W`, not the true kernel. This makes them physics-based regularizers.

| Module | Class | Physics | `z` | `W(z)` |
|---|---|---|---|---|
| `losses/db.py` | `DBLoss` | generic residual + weight schedules | – | – |
| `losses/surface_db.py` | `SurfaceDBLoss` | gas-surface scattering `K(v_i → v_r)` | incident velocity `v_i` | `\|n\| · M(v)` (flux-weighted wall Maxwellian) |
| `losses/collision_db.py` | `CollisionDBLoss` | particle-particle collisions `q(g' \| g)` | relative velocity `g = v1 − v2` | `B(\|g\|) · M_μ(g)` (colliding pairs) |

Both physics classes are subclasses of `DBLoss`, build the DB data (`make_db_data`, `sample_db_points`), evaluate both
sides of the equation for a model (`db_sides`) and work with the same models (`crealnvp_db`, `csplineflow_db`) and `fit_model`.
`get_config()["loss:db:type"]` is `"surface"` or `"collision"`.

`db_data` is a dict of arrays with these keys (all in the abstract variables the flow works in). The key names stem from
the gas-surface case but are generic: `v_i` is the condition, `v_r` the target of the kernel.

| Key | Meaning | surface | collision |
|---|---|---|---|
| `v_i`, `v_r` | condition and target of the forward pair | `(t, log u)` of `v_i`, `v_r` | `g`, `g'` |
| `v_i_bwd`, `v_r_bwd` | condition and target of the reversed pair `(T z' → T z)` | `(−t_r, log u_r)`, `(−t_i, log u_i)` | `−g'`, `−g` |
| `log_det_jacobian_fwd`, `log_det_jacobian_bwd` | log-Jacobian of the map from abstract to physical variables | `−log u` of the target | `0` |
| `log_M_i`, `log_M_r` | log equilibrium weight `log W` at condition / target | `log(\|n\| M)` | `log(B M_μ)` |

**Notes**

- `fit_model` combines the small MD set and the (usually larger) DB set internally and cycles the smaller one.
  One epoch is one pass over the larger set. When you compare runs with and without DB points, pass the same
  `steps_per_epoch` to both, otherwise the run without DB points gets far fewer optimizer steps.
- `csplineflow_db` is a drop-in replacement for `crealnvp_db` (same `compile` options, `db_data` and callbacks).
  Both share the DB training logic in `generative_model/db_mixin.py` (`DetailedBalanceMixin`). For the spline
  flow, the abstract variables should lie inside `[-tail_bound, tail_bound]`.
- `symmetric=True` makes the coupling networks see only `|cond|`. Use it only if the kernel really is invariant under
  sign changes of the condition components.
- A model compiled with a DB loss needs `db_data`, and `db_data` needs a model compiled with a DB loss;
  anything else raises an error.

### Gas-surface scattering (`SurfaceDBLoss`)

For a wall in equilibrium the scattering kernel satisfies

```
|n_i| · M(v_i) · K(v_i → v_r)  =  |n_r| · M(v_r) · K(−v_r → −v_i)
```

The flow works in the abstract variables `y = (t, log u)` (tangential components `t`, normal speed `u = |n|`), so it
needs no constraint `u > 0`.

```python
import neuraldist.generative_model.realnvp.crealnvp_db         # registers "crealnvp_db"
from neuraldist.losses import SurfaceDBLoss, DBWeightScheduler, LinearRampWeight

db_loss = SurfaceDBLoss(mass=1.0, kT=1.0, epoch_weighting=LinearRampWeight(0.0, 1.0, 20))
md = db_loss.make_db_data(t_i, u_i, t_r, u_r)                  # MD pairs in physical variables -> abstract
model = build_model("crealnvp_db", x_dim=3, cond_dim=3, n_coupling=6, symmetric=False)

fit_model(
    model, md["v_r"], md["v_i"],                               # MD pairs: target and condition
    db_data=db_loss.sample_db_points(4096, t_dim=2, rng=0),    # arbitrary pairs, passed separately
    compile_kwargs={"db_loss": db_loss},
    callbacks=[DBWeightScheduler()],                           # ramps the DB weight up over 20 epochs
    epochs=60, batch_size=256, steps_per_epoch=16,
)
```

`SurfaceDBLoss.from_abstract(y)` turns sampled `y` back into `(t, u)`.

### Particle-particle collisions (`CollisionDBLoss`)

For a binary collision the centre-of-mass velocity is conserved, so the kernel is learned in the relative frame,
`q(g' | g)` with `g = v1 − v2`. In equilibrium it satisfies

```
B(|g|) · M_μ(g) · q(g' | g)  =  B(|g'|) · M_μ(g') · q(−g | −g')
```

with the Maxwellian `M_μ` of the relative velocity (reduced mass `μ`) and the collision-rate factor
`B(|g|) = |g| σ_T ~ |g|^a` (VHS: `a = 2 − 2ω`, hard spheres `a = 1`, Maxwell molecules `a = 0`).

```python
from neuraldist.losses import CollisionDBLoss, to_relative_frame, from_relative_frame

db_loss = CollisionDBLoss(reduced_mass=0.5, kT=1.0, rate_exponent=1.0, epoch_weighting=LinearRampWeight(0, 1, 20))
model = build_model("crealnvp_db", x_dim=3, cond_dim=3, n_coupling=6)
fit_model(model, g_post, g_pre,                                # MD collisions in the relative frame
          db_data=db_loss.sample_db_points(4096, rng=0),       # arbitrary pairs (g, g')
          compile_kwargs={"db_loss": db_loss}, callbacks=[DBWeightScheduler()], epochs=60)
```

`make_db_data(g_pre, g_post)` / `make_db_data_lab(v1, v2, v1', v2', m1, m2)` build DB points from given pairs,
`from_relative_frame(G, g', m1, m2)` turns sampled `g'` back into lab-frame velocities.
`time_reversal=False` uses `q(g | g')` on the right-hand side (isotropic interactions).
Elastic collisions conserve `|g|` exactly, so a full 3D density only exists for inelastic kernels (e.g. energy
exchange with internal degrees of freedom in equilibrium).

## Notebooks

| Notebook | Content |
|---|---|
| `vae_cvae_distribution_fitting.ipynb` | fit a 2D Gaussian mixture and (temperature-dependent) Maxwell velocity distributions with VAE and cVAE; influence of `kl_weight` |
| `realnvp_crealnvp_distribution_fitting.ipynb` | the same targets with RealNVP and cRealNVP, including exact densities, KL estimates, `n_coupling` and the limits of extrapolation |
| `crealnvp_surface_detailed_balance.ipynb` | gas-surface: learn a 2D Cercignani-Lampis scattering kernel from few MD pairs with and without `SurfaceDBLoss` |
| `crealnvp_surface_detailed_balance_3d.ipynb` | the same in 3D (two tangential components and the normal component) |
| `crealnvp_collision_detailed_balance.ipynb` | particle-particle: learn an inelastic collision kernel `q(g' \| g)` (Maxwell molecules, different masses) with `CollisionDBLoss`; lab-frame check and the influence of the rate law |
| `splineflow_csplineflow_distribution_fitting.ipynb` | the same targets with SplineFlow and cSplineFlow, including the influence of `n_coupling` and `n_bins` |
| `csplineflow_surface_detailed_balance_3d.ipynb` | the 3D gas-surface kernel with `csplineflow_db` instead of `crealnvp_db` |
| `csplineflow_collision_detailed_balance.ipynb` | the collision kernel `q(g' \| g)` with `csplineflow_db` instead of `crealnvp_db` |

All notebooks compare against analytically known distributions, so the quality of every fit can be quantified.
They need `scipy` and `matplotlib`.

## Adding a new model

1. Subclass `GenerativeModel` and set `loss_names` (it must contain `"loss"`).
2. Implement `compute_losses(x, cond=None, training=True)` (returns a dict, `"loss"` is optimized) and
   `sample(n_samples, cond=None)`. Optionally implement `log_prob`.
3. Register it with `@register_model("name")` and import the module where the models are collected.


## Tests

```bash
pip install -e ".[dev]"
pytest
```
## Citation

If you use NeuralDist in your work, please cite the paper and the repository
(GitHub also offers "Cite this repository" based on [CITATION.cff](CITATION.cff)):

```bibtex
@article{schuette2026conditional,
  title     = {Conditional normalizing flow for gas--surface scattering from thermal to hypersonic velocities},
  author    = {Sch{\"u}tte, Miklas and Hocker, Stephen and Lipp, Hansj{\"o}rg and Roth, Johannes and Fasoulas, Stefanos and Pfeiffer, Marcel},
  journal   = {Physics of Fluids},
  volume    = {38},
  number    = {9},
  pages     = {097005},
  year      = {2026},
  publisher = {AIP Publishing},
  doi       = {10.1063/5.0350421}
}

@software{schuette2026neuraldist,
  title   = {{NeuralDist}: Machine learning models for learning (un)conditional distribution functions},
  author  = {Sch{\"u}tte, Miklas},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/schuettem/NeuralDist}
}
```
