# DeadliftDynamics

**A from-scratch study of constrained feedback control for a nonlinear, four-link planar system.**

<p align="center">
  <img src="assets/deadlift.gif" alt="Animation of the controlled four-link simulation" width="360">
</p>

The deadlift provides the geometry; the project is primarily numerical. At every time step, the simulator assembles a coupled dynamics system, solves for the next configuration, evaluates a multi-objective score, estimates its gradient through perturbed simulations, and projects the updated controls back onto their feasible set.

## The problem in one minute

- **Dynamic system:** four rigid segments connected by planar pivots, represented by their angles, angular velocities, centres of mass, forces and torques.
- **Control variables:** five actuator intensities constrained to $[0,1]^5$.
- **Numerical step:** a 12 × 12 linear system couples four next-step angles with eight joint-reaction components.
- **Controller:** a local numerical gradient is estimated independently along each control coordinate, then used in a projected ascent step.
- **Objective:** increase shoulder height while trading off horizontal centre-of-mass position and velocity, segment alignment, and rapid changes in alignment.

This is a feedback-control experiment, not a claim about optimal lifting technique or human biomechanics.

## Numerical method

### 1. Coupled dynamics

[`src/state.py`](src/state.py) defines the articulated chain and [`src/bone.py`](src/bone.py) maps segment angles to positions, actuator directions, forces and torques. For $n=4$ segments, [`src/matrix.py`](src/matrix.py) constructs the coefficients of

$$A(q_t)X = B(q_t,q_{t-1},F,C), \qquad X \in \mathbb{R}^{3n},\; n=4,$$

and [`src/update.py`](src/update.py) solves it with `numpy.linalg.solve`. The first four components of `X` update the segment angles; the remaining eight represent joint reactions. The 1 ms finite-difference step makes the dynamics solve the inner loop of the simulation.

### 2. Gradient estimation and constrained update

At each simulation step, [`src/brain.py`](src/brain.py) evaluates the objective after perturbing each of the five controls by $-10^{-3}$, $0$, and $+10^{-3}$. A least-squares slope gives one component of the numerical gradient. The controller then applies the existing projected update

$$e_{k+1}=\operatorname{clip}_{[0,1]}\!\left(e_k+\alpha\gamma^k\nabla Q(e_k)\right),$$

with decay factor $\gamma=0.7$, repeating until the largest control change is below $10^{-3}$. Each trial state is restored before the next perturbation, so all coordinate estimates start from the same configuration.

### 3. Objective trade-offs

The hand-designed objective combines five competing terms:

1. reward upward shoulder displacement;
2. penalise horizontal centre-of-mass error relative to a target;
3. penalise horizontal centre-of-mass speed;
4. penalise horizontal misalignment of selected segment endpoints;
5. penalise rapid changes in that misalignment.

The algorithm therefore seeks a locally useful control at each step. It does not optimise the full trajectory globally and carries no guarantee of convergence or optimality.

## Existing output

The animation visualises the resulting state sequence. Segment lines show the articulated configuration, while actuator colours vary with their control intensities. It illustrates the solver–controller loop; it is not physical validation. The original recording is available as an [MP4](assets/deadlift_video.mp4).

![Bar-endpoint height and horizontal centre-of-mass position over the existing one-second reference run](assets/trajectory.png)

Over the existing 1,000-step reference window, the bar endpoint moves from 0.225 m to 0.513 m and the maximum horizontal centre-of-mass deviation from the controller's 0.14 m target is 0.0045 m. These are outputs of the chosen parameterisation, not empirical performance measurements. The trajectory becomes unstable beyond the documented window, so the repository does not demonstrate a complete lift.

## Nicolas's contribution

Nicolas implemented the complete numerical pipeline in this repository:

- articulated geometry and state propagation;
- force, torque and matrix assembly;
- the 12 × 12 dynamics solve;
- bounded controls, numerical gradient estimation and projected updates;
- objective diagnostics and Pygame visualisation.

No external physics engine or automatic-differentiation framework is used.

## Run the existing demo

Tested with Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/demo.py
```

The final command runs the existing bounded, headless reference simulation, prints its recorded metrics and writes `assets/trajectory.png`. The original interactive visualisation is launched with `python src/main.py` and runs until its Pygame window is closed.

## Approximate parameterisation and limits

The current geometry uses four uniform rods and five line actuators. Segment dimensions, masses, attachment locations and maximum forces are approximate modelling parameters. The nominal 175 kg load is folded into the arm segment's mass and inertia rather than represented as a point mass at the bar.

The model also omits three-dimensional motion, joint limits, passive tissues, activation dynamics, fatigue, collision handling and a detailed foot–ground contact model. There is no calibration against motion capture or force-plate data, no time-step convergence study, and no demonstrated energy-conservation result.

## Read the core implementation

| Path | What to inspect |
| --- | --- |
| [`src/matrix.py`](src/matrix.py) | Assembly of the coupled dynamics matrix and right-hand side |
| [`src/update.py`](src/update.py) | Per-step solve and state update |
| [`src/brain.py`](src/brain.py) | Objective, numerical gradient and projected controller |
| [`src/state.py`](src/state.py) | Segment, actuator and initial-state parameters |
| [`src/bone.py`](src/bone.py) | Kinematics and force/torque geometry |
| [`src/demo.py`](src/demo.py) | Existing bounded reference run |

For a quick code review, follow `state.py` → `matrix.py` / `update.py` → `brain.py`.
