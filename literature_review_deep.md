# Deep Literature Review: Zero-Shot RL, World Models & Compositional Planning (2025-2026)

## Executive Summary

This review analyzes 8 recent papers that collectively form the state-of-the-art in zero-shot reinforcement learning, world models, and compositional planning. The papers cluster into three interconnected research programs:

1. **The TD-JEPA / FB Ecosystem** (Bagatella, Pirotta, Touati, Farebrother, Tirinzoni, Bellemare, Lazaric at FAIR/Meta)
2. **The JEPA / Hierarchical World Model Line** (LeCun, Ballas, Terver, Zhang at FAIR)
3. **Structural World Models & Spectral Methods** (Tang et al., Chandrasekar & Machado)

---

## Part I: Core Algorithms and Theory

---

### 1. TD-JEPA: Latent-Predictive Representations for Zero-Shot RL
*Marco Bagatella, Matteo Pirotta, Ahmed Touati* — arXiv 2025

#### Architecture
TD-JEPA learns two asymmetric encoders:
- **φ (backward encoder)**: Maps state-action pairs to latent representations
- **ψ (forward encoder)**: Maps successor states to latent representations
- **Forward predictor**: Predicts temporal-difference Q-values in latent space

The key innovation is **TD-learning in representation space** rather than value space:
```
Q(s, a, z) = F_θ(s, a, z)^T B_φ(s')  # latent inner product
```

#### Training Objectives
1. **TD Loss**: `(pred - target).pow(2).sum(-1).mean()` with EMA target networks (τ=0.005)
2. **Orthonormality Loss**: `offdiag + diag` with L2-normalized encodings (||φ||=||ψ||=√d)
3. **Actor Loss**: `(-Q/|Q|) + bc_coeff * MSE` (behavior cloning regularization)

#### Zero-Shot Adaptation
At test time, reward inference projects task rewards onto ψ-space via least-squares:
```
z_task = argmin_z ||r - ψ(next_obs)^T z||^2
```

This allows recovery of optimal policies for any reward without retraining.

#### Critical Implementation Details
- **Separate optimizers** for φ, ψ, forward predictor, actor (each with own Adam)
- **z-sampling**: 50% random sphere + 50% ψ(next_obs), with optional whitening (`scale_train_goals`)
- **Ensemble**: Multiple forward predictors with shared embeddings, per-member loss
- **Architecture**: Encoder 4-layer MLP (512 hidden, ReLU body, LN→Tanh head), Predictor 2-layer

#### OGBench Performance
- Evaluated on `antmaze-medium-navigate-v0` and `large-diverse-v0`
- Zero-shot: task z inferred from physics-state relabeling + reward least-squares fit
- Reference sweep uses `TDJEPAFlowBCAgent` (flow-matching actor) with `bc_coeff=0.3`
- Our reproduction achieved **42% at 150K steps** (up from 0% at 25K) with healthy z-space

---

### 2. Soft Forward-Backward Representations for Zero-Shot RL with General Utilities
*Marco Bagatella, Thomas Rupf, Georg Martius, Andreas Krause* — arXiv:2602.06769

#### Problem Extension
Standard FB solves **Linear RL**: `J^π = ⟨M^π, R⟩` (reward linear in occupancy measure).

Soft FB extends to **General Utilities**: `J^π = f(M^π)` where f is any differentiable function of the occupancy measure. This captures:
- Distribution matching
- Pure exploration (maximize entropy of state-action visitations)
- Active learning
- Learning from observations

#### Key Theoretical Results

**Theorem 4.1** (Expressiveness): For any bounded reward vector R, if the FB decomposition holds:
```
M^z = F_z^T B  (low-rank factorization)
π_z(a|s) ∝ exp(Q_z(s,a) / (1-||z||))  (softmax policy)
```
where `Q_z(s,a) = F_θ(s,a,z)^T z`, then π_z is the **optimal maximum-entropy policy** for reward R when z = B^T R.

**Theorem 4.2** (General Utilities): For any differentiable function f of occupancy measures, the set of policies {π_z} retrieved by Soft FB contains a near-optimal policy for each GU instance.

#### Reparameterization Trick
The key practical innovation:
```
Q_z = F_z^T z + M^z H^{π_z}
    ∝ (1/||z||) F_z^T z' + (1-||z'||) M^z H^{π_z}

where z' = z / (||z|| + 1), so 0 ≤ ||z'|| < 1
```

This maps all policies into a **bounded embedding space** while preserving optimal policies (invariant to Q-function scaling).

**Deterministic policies** are retrieved as ||z'|| → 1 (entropy regularization vanishes).
**Stochastic policies** are retrieved for ||z'|| < 1.

#### Training Objectives
1. **FB Bellman Residual**:
```
L_FB(θ, φ) = E_{z~U(Z), (s_t,a_t,s_{t+1})~ρ} [
    (F_θ(s_t,a_t,z)^T B_φ(s') - γ F̄(s_{t+1},a_{t+1},z)^T B̄(s'))^2
    - 2 F_θ(s_t,a_t,z)^T B_φ(s_{t+1})
]
```

2. **Entropy Critic TD Loss**:
```
L_H(η) = E [ (Q_{H,η}(s_t,a_t,z) - γ Q̄_{H,η}(s_{t+1},a_{t+1},z) - log π_ψ(a_{t+1}|s_{t+1},z))^2 ]
```

3. **Policy KL Loss**:
```
L_π(ψ) = E_{z~U(Z), s~ρ, a~π_ψ} [
    (1-||z||)[log π_ψ(a|s,z) - Q_H(s,a,z)] - F_θ(s,a,z)^T z
]
```

#### Inference: Zero-Order Search
At test time, optimize the general utility directly:
```
z* ≈ argmax_{z∈Z} f(M̂^z)
```

Two measure estimation methods:
- **Implicit**: Importance weighting using FB representations: `M̂^{π^z_S}(s) ∝ E_{s_0~μ_0, a_0~π} F_θ(s_0,a_0,z)^T B_φ(s) ρ(s)`
- **Explicit**: Flow-based generative model of successor measure (Farebrother et al., 2025)

Search uses random shooting or CEM over low-dimensional z ∈ R^d.

#### Experiments
- **Didactic**: 2D environments showing policy stochasticity increases with ||z||
- **DMC**: Standard zero-shot benchmarks ( walker, cheetah, hopper, quadruped, humanoid )
- **General Utilities**: Linear RL, Stochastic Imitation Learning, Pure Exploration
- **Result**: SFB matches/exceeds standard FB on linear tasks while solving non-linear tasks FB cannot

#### Connection to TD-JEPA
Both use asymmetric encoders and latent inner products for Q-values. Key differences:
- **TD-JEPA**: TD-learning with target networks, orthonormality regularizer, TruncatedNormal actor
- **Soft FB**: Maximum-entropy policies, FB decomposition, explicit flow-based measure inference
- **Unification**: TD-JEPA could be viewed as a deterministic limit of Soft FB (||z|| → 1)

---

### 3. Zero-Shot Off-Policy Learning
*Arip Asadulaev, Maksim Bobrin, Salem Lahlou, Dmitry Dylov, Fakhri Karray, Martin Takac* — arXiv:2602.01962

#### Core Insight
Discovered a **theoretical connection between successor measures and stationary density ratios**:
```
M^π(s,a) ∝ ρ^π(s,a) / ρ^D(s,a)  # measure = optimal policy density / data density
```

This allows inferring **optimal importance sampling ratios** on the fly:
```
w*(s,a) = M^{π_task}(s,a) / M^{π_data}(s,a)
```

#### Algorithm
1. Pretrain successor measure models for data-collecting policy π_data
2. At test time, infer task-specific measure M^{π_task} via reward-conditioned least-squares
3. Compute density ratios w*(s,a) for off-policy correction
4. Use corrected samples for value estimation / policy optimization

#### Integration with FB
The method "seamlessly integrates into forward-backward representation frameworks" — it can be added as an off-policy correction layer on top of FB/TD-JEPA representations.

#### Benchmarks
- **SMPL Humanoid**: Motion tracking tasks
- **ExoRL**: Continuous control
- **OGBench**: Long-horizon navigation tasks

**Result**: "Fast adaptation to new tasks in a training-free regime."

#### Connection to TD-JEPA
This addresses a weakness in our current implementation: the off-policy nature of reward inference. When we project task rewards onto ψ-space using least-squares, we're implicitly assuming on-policy samples. The density-ratio correction could improve reward inference accuracy for out-of-distribution task z's.

---

## Part II: World Models for Planning

---

### 4. Compositional Planning with Jumpy World Models
*Jesse Farebrother, Matteo Pirotta, Andrea Tirinzoni, Marc G. Bellemare, Alessandro Lazaric, Ahmed Touati* — arXiv:2602.19634

#### Concept
Instead of planning over primitive actions, **compose pre-trained policies** as temporally extended actions. The challenge: compounding errors in long-horizon predictions make it hard to estimate visitation distributions from policy sequences.

#### Geometric Horizon Models (GHMs)
A GHM models the **successor measure** m^π_γ as a generative model:
```
m^π_γ(X|s,a) = (1-γ) Σ γ^k Pr(S_{k+1} ∈ X | S_0=s, A_0=a, π)
```

Interpretation: probability that the geometrically-distributed halting state lies in X.

#### Flow-Matching Formulation
The GHM is parameterized as an ODE:
```
dψ_t(X_0|s,a)/dt = v_t(ψ_t(X_0|s,a)|s,a)
ψ_0(X_0|s,a) = X_0
```

where v_t is learned via **Temporal Difference Flow Matching**:
```
L_td-flow(θ; θ̄) = (1-γ) E [ ||v_t(X_t|S,A;θ) - (S'-X_0)||^2 ]
                 + γ E [ ||v_t(X_t|S,A;θ) - v_t(X_t|S',A';θ̄)||^2 ]
```

- First term: conditional flow-matching to one-step transition P(·|S,A)
- Second term: marginal flow-matching to bootstrapped successor measure

#### Horizon Consistency Objective
Novel contribution: align predictions across **multiple timescales**:
```
ℓ_k(θ) = (1-γ^k) ℓ^{→}_k(θ) + γ^k [(1-β^k)/(1-β^k) ℓ^{↷β}_k(θ) + (γ^k-β^k)/(1-β^k) ℓ^{↷γ}_k(θ)]
```

where β ~ U([0,γ]) is an intermediate discount factor. This enforces consistency between:
- One-step transitions (γ^0 = 0)
- Intermediate horizons (β)
- Full horizons (γ)

#### Geometric Switching Policies (GSPs)
Policies switch stochastically: each policy π_{z_i} executes for a geometrically distributed duration T_i ~ Geom(α_i).

Effective discount factor for policy i:
```
β_i = γ(1-α_i)
```

**Theorem 1** (Successor Measure Decomposition): The successor measure of a GSP decomposes as a mixture:
```
m^ν_γ(ds+|s,a) = Σ_{k=1}^n w_k ∫ m^{π_{z_1}}_{β_1}(ds_1|s,a) π_{z_2}(da_1|s_1) ... m^{π_{z_k}}_{β_k}(ds+|s_{k-1},a_{k-1})
```

with mixture weights:
```
w_k = (1-γ)/(1-β_k) Π_{i<k} (γ-β_i)/(1-β_i)
```

#### Compositional Planning Algorithm
```
function CompPlan(s):
    for i = 1...M:
        (z_1^(i), ..., z_K^(i)) ~ ρ(·|s)   # sample policy sequence from proposal
        a_1^(i) ~ π_{z_1^(i)}(·|s)           # sample initial action
        
        # Monte Carlo Q-value estimation
        for j = 1...N:
            (S_0, A_0) = (s, a_1^(i))
            for k = 1...K:
                S_k ~ m^{π_{z_k}}_{β_k}(·|S_{k-1}, A_{k-1})  # jumpy transition
                A_k ~ π_{z_{k+1}}(·|S_k)
            Q̂^(i,j) = (1-γ)^{-1} Σ w_k · r(S_k)
        Q̂^(i) = (1/N) Σ_j Q̂^(i,j)
    
    i* = argmax_i Q̂^(i)
    return (a_1^(i*), z_1^(i*))
```

#### Proposal Distributions
Two options for sampling policy sequences:
1. **Goal-Conditioned**: Chain GHM samples toward goal: `z_k ~ m^{π_g}_{β_k}(·|z_{k-1}, A_{k-1})`
2. **Unconditional**: Chain behavior policy's GHM: `z_k ~ m^μ_{β_k}(·|z_{k-1})`

#### Experimental Results
Evaluated on **OGBench** with 5 base policies per domain:
- **CRL** (Contrastive RL)
- **GC-1S** (Goal-Conditioned 1-Step)
- **GC-BC** (Goal-Conditioned BC)
- **GC-TD3** (Goal-Conditioned TD3)
- **HFBC** (Humanoid Foundation Model BC)

**antmaze-medium**: CompPlan improves over zero-shot across all base policies, especially weak ones:
- GC-1S zero-shot: 0.400 → CompPlan: 0.930
- GC-BC zero-shot: 0.400 → CompPlan: 0.870
- CRL zero-shot: 0.950 → CompPlan: 0.970

**antmaze-large**: Even more dramatic:
- GC-1S zero-shot: 0.100 → CompPlan: 0.770
- GC-TD3 zero-shot: 0.300 → CompPlan: 0.500

**Key finding**: Compositional planning provides a **200% relative improvement** over action-level planning on long-horizon tasks.

#### Connection to TD-JEPA
- **Same authors**, same ecosystem
- TD-JEPA provides the **base policies** {π_z} that Jumpy WMs compose
- The GHM could be learned on top of TD-JEPA's ψ-space
- The flow-matching formulation (Algorithm 1) is a natural extension of TD-JEPA's latent dynamics

---

### 5. Hierarchical Planning with Latent World Models
*Wancong Zhang, Basile Terver, Artem Zholus, Soham Chitnis, Harsh Sutaria, Mido Assran, Randall Balestriero, Amir Bar, Adrien Bardes, Yann LeCun, Nicolas Ballas* — arXiv:2604.03208

#### Concept
Learn **world models at multiple temporal scales** within a shared latent space, then plan hierarchically:
- **High-level planner**: Long-horizon world model → predicts subgoal latents
- **Low-level planner**: Short-horizon world model → plans primitive actions to reach subgoal

#### Architecture
```
High-Level WM:   z_{t+H} = f_H(z_t, a_t^{macro})    # H-step lookahead
Low-Level WM:    z_{t+1} = f_1(z_t, a_t^{micro})    # 1-step lookahead
```

Both operate in the **same latent space** (not separate spaces), enabling direct coupling via latent-state matching.

#### Key Innovation: Learned Action Encoder
Compresses sequences of primitive actions between waypoints into **latent macro-actions**:
```
a^{macro} = Encoder(a_{t:t+h})
```

This reduces the high-level search space dimensionality.

#### Training
- Task-agnostic next-latent prediction (no task-specific rewards)
- Self-supervised from unlabeled offline trajectories
- Hierarchical levels interact only at inference time

#### Real-World Results
**Franka robot pick-&-place**:
- Flat WM: **0%** success (no oracle subgoals)
- Hierarchical WM: **70%** success

**Push-T** (simulated manipulation):
- Flat: 17% → Hierarchical: 61% (+44%)

**Diverse Maze**:
- Flat: 44% → Hierarchical: 83% (+39%)

**Planning efficiency**: Hierarchical matches or exceeds success with **~3× lower planning time**.

#### Connection to TD-JEPA
- **LeCun's JEPA group** — same architectural philosophy (predict in latent space, not pixel space)
- Could combine: TD-JEPA for task-conditioned representations, HWM for hierarchical execution
- The shared latent space principle aligns with TD-JEPA's ψ-space being shared across tasks

---

### 6. HaM-World: Soft-Hamiltonian World Models with Selective Memory
*Haoyun Tang, Haodong Cui, Keyao Xu, Kun Wang, Zhandong Mei* — arXiv:2605.05951

#### Motivation
World models become unstable as planning horizons grow. Two missing structures:
1. **History-conditioned memory**: Partial observability and action delay make single-frame latents insufficient
2. **Geometric organization**: Configuration, momentum, and task semantics are entangled in monolithic representations

#### Architecture
Latent state decomposed into:
```
z = (q, p, c)

where:
- q: configuration (position-like)
- p: momentum (velocity-like)
- c: context (semantic, dissipative, non-conservative factors)
```

#### Dynamics
**(q, p)** evolves through an **energy-derived Hamiltonian vector field** plus learnable residual:
```
dq/dt = ∂H/∂p + residual_q
 dp/dt = -∂H/∂q + residual_p
```

**c** captures everything else (semantic, dissipative).

#### Memory: Mamba Selective State-Space
Uses Mamba as the history-conditioned input to the latent dynamics:
```
h_t = SSM(x_t, h_{t-1})  # selective memory
z_t = f(h_t)             # latent state
```

This provides **input-dependent memory** for approximate Markov completeness.

#### Unified Interface
The planner sees a **single latent state** shared by:
- Dynamics prediction
- Reward/value estimation
- Imagined rollouts
- CEM action search

#### Results
**DeepMind Control Suite** (4 tasks):
- Highest Avg. AUC: **117.9** (+9.5% over baseline)
- Long-horizon rollout error: **45%** of baseline
- Wins **11/12** planning cells (k ∈ {3,5,7})

**OOD Perturbations** (12 conditions: dynamics shifts, action delay, observation masking):
- Highest return in **every condition**
- Average OOD-return gains: **10.2%** (Finger Spin), **13.6%** (Reacher Easy)

**Mechanism diagnostics**:
- Bounded action-free Hamiltonian-energy drift
- Structured energy variation under policy rollouts
- Coherent control-induced energy transfer

#### Connection to TD-JEPA
- Both learn structured latent dynamics
- TD-JEPA: orthonormal encoders + TD Q-predictions (energy = ||φ||² + ||ψ||²)
- HaM-World: Hamiltonian structure + selective memory
- **Potential combination**: Use TD-JEPA's ψ-space as the "context" c, while HaM-World provides (q,p) dynamics

---

## Part III: Spectral Methods & Option Discovery

---

### 7. The Laplacian Keyboard: Beyond the Linear Span
*Siddarth Chandrasekar, Marlos C. Machado* — arXiv:2602.07730

#### Concept
Graph Laplacian eigenvectors provide a natural basis for RL. Prior work (Proto-Value Functions, FB) is limited to the **linear span** of these eigenvectors. The Laplacian Keyboard goes **beyond** by:

1. Constructing a **task-agnostic library of options** from Laplacian eigenvectors
2. Forming a **behavior basis** guaranteed to contain the optimal policy for any reward within the linear span
3. Learning a **meta-policy** to dynamically stitch options, enabling policies outside the original span

#### Theoretical Guarantees
**Theorem**: The Laplacian Keyboard's behavior basis contains the optimal policy for any reward within the linear span of Laplacian eigenvectors.

**Corollary**: The meta-policy's stitching error is bounded by the spectral gap of the graph Laplacian.

#### Algorithm
1. Compute graph Laplacian L from state-transition graph
2. Extract top-k eigenvectors {v_1, ..., v_k}
3. Construct options: each option terminates when the eigenvector's sign changes
4. Meta-policy learns to compose options via GPI (Generalized Policy Improvement)

#### Connection to TD-JEPA
- **TD-JEPA's orthonormality loss** encourages encoders to become orthogonal, similar to Laplacian eigenvectors
- If TD-JEPA's ψ-space converges to the graph Laplacian's eigenbasis, it gains the same theoretical guarantees
- The "z-space collapse" we observed (cos ~0.999) is the opposite of this — the encoders were not orthogonal
- **Recommendation**: Monitor the eigenvalue spectrum of Cov(ψ) during training to verify spectral structure

---

## Part IV: Synthesis & Unified Perspective

---

### The Three Pillars of Zero-Shot RL

| Pillar | Papers | Key Mechanism |
|--------|--------|---------------|
| **Representation Learning** | TD-JEPA, Soft FB, Laplacian Keyboard | Learn task-agnostic representations from offline data |
| **Planning & Composition** | Jumpy WMs, Hierarchical WMs, HaM-World | Compose/plan in latent space at multiple scales |
| **Off-Policy Correction** | Zero-Shot Off-Policy | Density ratios for stable value estimation |

### The Unified Pipeline

```
Offline Data → Representation Learning → Planning/Composition → Zero-Shot Task Adaptation
     ↓              ↓                        ↓                      ↓
  Transitions   TD-JEPA / FB           Jumpy / Hierarchical       Reward Inference
  Unlabeled     Asymmetric encoders    Multi-scale world models     Least-squares / Density ratios
                ψ-space (task-agnostic)  Flow-matching GHMs         CEM / Zero-order search
```

### Key Technical Decisions for TD-JEPA Practitioners

#### 1. Actor Design
- **TD-JEPA (original)**: TruncatedNormal Gaussian actor, BC coefficient 0.3
- **Soft FB**: Maximum-entropy policy, reparameterized z' with ||z'|| ∈ [0,1]
- **Jumpy WMs**: Policy sequences with geometric switching
- **Recommendation**: If eval success is low but z-space is healthy, try Soft FB's entropy-regularized actor

#### 2. Reward Inference
- **TD-JEPA**: Least-squares projection onto ψ-space
- **Zero-Shot Off-Policy**: Density ratio correction via successor measures
- **Soft FB**: Zero-order search over z (CEM)
- **Recommendation**: Combine TD-JEPA's least-squares with density-ratio weighting for OOD robustness

#### 3. Long-Horizon Stability
- **TD-JEPA**: Single-step TD in latent space
- **Jumpy WMs**: Multi-step GHMs with horizon consistency
- **HaM-World**: Hamiltonian structure + selective memory
- **Hierarchical WMs**: Multi-scale planning
- **Recommendation**: For long-horizon tasks (antmaze-large, giant), add hierarchical planning or Hamiltonian structure

#### 4. Exploration
- **TD-JEPA**: Random z-sampling (50% sphere, 50% ψ(next_obs))
- **FB-MEBE**: Maximum-entropy behavior exploration (maximize entropy of achieved behaviors)
- **Soft FB**: Stochastic policies via entropy regularization
- **Recommendation**: If offline data coverage is poor (Task 5 had only 2/10000 positives), use FB-MEBE-style exploration

### Critical Hyperparameters Across Papers

| Parameter | TD-JEPA | Soft FB | Jumpy WMs | Hierarchical WMs | HaM-World |
|-----------|---------|---------|-----------|-------------------|-----------|
| Encoder depth | 4 | 4 | 4 | Varies | 4 |
| Predictor depth | 2 | 2 | Flow-matching | 2 | 2 |
| Latent dim | 50 | 50-512 | 50 | 25-75 | 128-512 |
| Batch size | 256 | 256 | 256 | 256 | 256 |
| Learning rate | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| EMA τ | 0.005 | 0.005 | 0.005 | 0.005 | 0.005 |
| BC coefficient | 0.3 | N/A | N/A | N/A | N/A |
| Entropy coef | N/A | ||z'|| | N/A | N/A | N/A |
| psi_ortho_coef | 1.0 | 1.0 | N/A | N/A | N/A |

### Open Questions & Future Directions

1. **Can we combine TD-JEPA + HaM-World?** Use TD-JEPA's ψ-space as the "context" c, with Hamiltonian (q,p) dynamics
2. **Can we add horizon consistency to TD-JEPA?** The multi-timescale objective from Jumpy WMs could stabilize long-horizon predictions
3. **Can we use Soft FB's general utilities?** Extend TD-JEPA beyond additive rewards to distribution matching, exploration
4. **Can we add hierarchical planning?** Use TD-JEPA base policies with HWM's multi-scale MPC
5. **Can we improve reward inference with density ratios?** Integrate Zero-Shot Off-Policy's stationary measure correction

### The Path to 80%+ Zero-Shot on OGBench

Based on the literature, the winning combination for hard OGBench tasks:

1. **Base**: TD-JEPA architecture (asymmetric encoders, orthonormality, TD loss)
2. **Actor**: Soft FB's maximum-entropy policy (stochasticity helps exploration)
3. **Inference**: Density-ratio corrected least-squares (Zero-Shot Off-Policy)
4. **Long-horizon**: Add jumpy world models or hierarchical planning for composition
5. **Stability**: Hamiltonian structure or horizon consistency for rollout stability
6. **Exploration**: FB-MEBE-style maximum-entropy data collection if online

---

## Appendix: Experimental Details from Papers

### OGBench Evaluation Protocol (from Jumpy WMs paper)
- **Domains**: antmaze-medium, antmaze-large, antmaze-giant, cube-1, cube-2, cube-3, cube-4
- **Tasks per domain**: 5 goal-conditioned tasks
- **Base policies evaluated**: CRL, GC-1S, GC-BC, GC-TD3, HFBC
- **Evaluation**: 10 trajectories per task, mean/std over 3 seeds

### Base Policy Hyperparameters (from Jumpy WMs Appendix F)
- **CRL**: Contrastive RL with goal relabeling
- **GC-1S**: 1-step goal-conditioned RL
- **GC-BC**: Behavior cloning with goal conditioning
- **GC-TD3**: Goal-conditioned TD3
- **HFBC**: Humanoid Foundation Model BC (pre-trained on diverse humanoid data)

### Flow-Matching Training (from Jumpy WMs Algorithm 1)
- Vector field: 4-layer MLP, 512 hidden units
- ODE solver: Euler method, 10 steps
- Prior p_0: Standard Gaussian
- Training: 1M gradient steps
- Batch size: 256
- Optimizer: Adam, lr=1e-4

---

*Deep literature review compiled from arXiv PDFs and Semantic Scholar citation graphs, May 2026.*
