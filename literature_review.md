# Literature Review: Zero-Shot RL, World Models & Representation Learning (2025-2026)

A survey of 8 recent papers on zero-shot reinforcement learning, compositional planning, world models, and representation learning, with connections to the TD-JEPA framework.

---

## 1. Core Architecture: Latent-Predictive Representations

### TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning
*Marco Bagatella, Matteo Pirotta, Ahmed Touati* — arXiv 2025

The foundational work. TD-JEPA trains asymmetric encoders (φ backward, ψ forward) with a joint-embedding predictive architecture, learning task-agnostic representations from offline data. A forward predictor estimates temporal-difference Q-values in latent space, enabling zero-shot policy recovery for any reward via reward inference (least-squares projection onto ψ-space). The architecture uses separate optimizers, EMA target networks, and orthonormality regularization to prevent representation collapse.

**Key insight**: TD-learning in latent space (rather than value space) separates representation learning from policy optimization, allowing generalization to unseen reward functions.

**Related to this review**: All 8 papers below either build on, extend, or complement the TD-JEPA paradigm.

---

## 2. Compositional Planning with Jumpy World Models

**arXiv:2602.19634** — *Jesse Farebrother, Matteo Pirotta, Andrea Tirinzoni, Marc G. Bellemare, Alessandro Lazaric, Ahmed Touati* (2026)

**Abstract**: Rather than reasoning over primitive actions, they study agents that compose pre-trained policies as temporally extended actions. The key challenge is compounding errors in long-horizon predictions when estimating the visitation distribution induced by sequencing policies. They learn **jumpy world models** — predictive models of multi-step dynamics that capture state occupancies induced by pre-trained policies across multiple timescales in an off-policy manner. Building on Temporal Difference Flows (arXiv:2503.09817), they add a consistency objective that aligns predictions across timescales. Results show 200% relative improvement over primitive-action planning on long-horizon tasks.

**Connection to TD-JEPA**: Both papers share authors (Pirotta, Touati, Farebrother, Tirinzoni, Bellemare, Lazaric) and the same research group at FAIR. While TD-JEPA learns latent representations for zero-shot reward adaptation, Jumpy World Models learn multi-step dynamics for compositional planning. The two approaches are complementary: TD-JEPA provides the zero-shot task-conditioning mechanism, while jumpy models provide temporal abstraction for long-horizon control.

**Related work to follow**:
- Temporal Difference Flows (arXiv:2503.09817) — the base framework for the consistency objective
- Geometric Policy Composition (arXiv:2206.08736) — theoretical foundation for policy sequencing
- π*0.6: a VLA That Learns From Experience (2025) — 116 cites, visual-language-action models

---

## 3. Hierarchical Planning with Latent World Models

**arXiv:2604.03208** — *Wancong Zhang, Basile Terver, Artem Zholus, Soham Chitnis, Harsh Sutaria, Mido Assran, Randall Balestriero, Amir Bar, Adrien Bardes, Yann LeCun, Nicolas Ballas* (2026)

**Abstract**: Model predictive control (MPC) with learned world models generalizes zero-shot to new environments, but learned models struggle with long-horizon control due to prediction error accumulation and exponentially growing search spaces. They learn latent world models at multiple temporal scales and perform hierarchical planning across these scales, enabling long-horizon reasoning while reducing inference-time planning complexity. The approach is a modular planning abstraction applicable across diverse latent world-model architectures. Real-world robotic pick-&-place achieves 70% success (vs. 0% for single-level), with up to 4× less planning compute in simulation.

**Connection to TD-JEPA**: Both are JEPA-family approaches (LeCun, Balestriero, Bardes, Bar are co-authors). While TD-JEPA focuses on representation learning for zero-shot reward adaptation, this paper focuses on planning with world models at multiple temporal scales. The hierarchical planning abstraction could be combined with TD-JEPA's representations: use TD-JEPA to condition the policy on any task, then use hierarchical latent models for long-horizon execution.

**Related work to follow**:
- What Drives Success in Physical Planning with JEPA World Models? (Terver et al 2025) — 14 cites
- A Lightweight Library for Energy-Based JEPAs (Terver et al 2026) — 3 cites
- Dynamics-Aligned Latent Imagination (Röder et al 2025) — contextual world models

---

## 4. HaM-World: Soft-Hamiltonian World Models with Selective Memory

**arXiv:2605.05951** — *Haoyun Tang, Haodong Cui, Keyao Xu, Kun Wang, Zhandong Mei* (2026)

**Abstract**: World models become unstable as planning horizons grow. They argue this reflects two missing structures: (1) history-conditioned memory for approximate Markov completeness, and (2) geometric organization separating configuration (q), momentum (p), and task semantics (c). HaM-World decomposes latent state into canonical (q, p) subspace and context c, using Mamba selective state-space memory as history-conditioned input. The (q, p) pair evolves through an energy-derived Hamiltonian vector field plus learnable residual dynamics, while c captures semantic/dissipative factors. On DMC tasks, highest Avg. AUC (117.9, +9.5%), reduces long-horizon rollout error to 45% of baseline, and wins 11/12 planning cells. Under 12 OOD perturbations, achieves highest return in every condition.

**Connection to TD-JEPA**: Both learn structured latent dynamics, but with different inductive biases. TD-JEPA uses orthonormal encoders + TD Q-predictions for task conditioning. HaM-World uses Hamiltonian structure + selective memory for stable long-horizon rollouts. The two could be combined: TD-JEPA's ψ-space for task conditioning, HaM-World's (q,p,c) decomposition for dynamics.

**Related work to follow**:
- SOMBRL: Scalable and Optimistic Model-Based RL (Sukhija et al 2025)
- DyMoDreamer: World Modeling with Dynamic Modulation (Zhang et al 2025)
- RoboScape: Physics-informed Embodied World Model (Shang et al 2025) — 25 cites

---

## 5. Soft Forward-Backward Representations for Zero-Shot RL with General Utilities

**arXiv:2602.06769** — *Marco Bagatella, Thomas Rupf, Georg Martius, Andreas Krause* (2026)

**Abstract**: Forward-backward (FB) algorithms retrieve families of policies that approximately solve any standard RL problem (additive rewards, linear in occupancy). This paper extends FB to **general utilities**, where the objective is an arbitrary differentiable function of the occupancy measure — capturing distribution matching, pure exploration, and other non-additive-reward tasks. They introduce a maximum entropy (soft) variant of FB that recovers stochastic policies from offline data. Coupled with zero-order search over compact policy embeddings, this sidesteps iterative optimization and optimizes general utilities directly at test-time.

**Connection to TD-JEPA**: Same first author (Bagatella). TD-JEPA is a specific instance of latent-predictive architectures for standard RL. This paper generalizes the FB framework to arbitrary utility functions. The "soft" maximum-entropy formulation may help with exploration in TD-JEPA's actor training. The compact policy embeddings for zero-order search could replace TD-JEPA's current Gaussian actor with a more expressive distribution.

**Related work to follow**:
- Optimistic Task Inference for Behavior Foundation Models (Rupf et al 2025)
- Epistemically-guided FB Exploration (Urpí et al 2025)
- Intention-Conditioned Flow Occupancy Models (Zheng et al 2025)

---

## 6. The Laplacian Keyboard: Beyond the Linear Span

**arXiv:2602.07730** — *Siddarth Chandrasekar, Marlos C. Machado* (2026)

**Abstract**: Laplacian eigenvectors provide a natural basis for approximating reward functions in RL, but are typically limited to their linear span. They introduce the **Laplacian Keyboard (LK)**, a hierarchical framework that constructs a task-agnostic library of options from Laplacian eigenvectors, forming a behavior basis guaranteed to contain the optimal policy for any reward within the linear span. A meta-policy learns to dynamically stitch these options, enabling efficient learning of policies outside the original linear constraints. Theoretical bounds on zero-shot approximation error are established.

**Connection to TD-JEPA**: Both use spectral/orthogonal representations for zero-shot RL. TD-JEPA learns orthogonal encoders via the orthonormality regularizer (φ_orth, ψ_orth). The Laplacian Keyboard uses the graph Laplacian's eigenvectors as a predefined orthogonal basis. The key difference: TD-JEPA learns representations end-to-end via TD-learning, while LK uses the environment's state-transition graph Laplacian. LK's option-stitching meta-policy could be combined with TD-JEPA's learned representations.

**Related work to follow**:
- Constructing an Optimal Behavior Basis for the Option Keyboard (Alegre et al 2025)
- Regularized Latent Dynamics Prediction for BFMs (Jajoo et al 2026)

---

## 7. Zero-Shot Off-Policy Learning

**arXiv:2602.01962** — *Arip Asadulaev, Maksim Bobrin, Salem Lahlou, Dmitry Dylov, Fakhri Karray, Martin Takac* (2026)

**Abstract**: Off-policy learning from fixed datasets suffers from distributional shift and value overestimation. These issues worsen in zero-shot RL where agents must adapt to new tasks without additional training. They discover a theoretical connection between **successor measures** and stationary density ratios, enabling inference of optimal importance sampling ratios on the fly. This performs stationary distribution correction with an optimal policy for any task in a training-free regime. Benchmarked on SMPL Humanoid motion tracking, ExoRL continuous control, and long-horizon OGBench tasks. Seamlessly integrates into forward-backward frameworks.

**Connection to TD-JEPA**: Both address zero-shot RL from offline data, but focus on different failure modes. TD-JEPA prevents representation collapse via orthonormality and learns task-conditioned Q-values. This paper addresses the off-policy distributional shift via successor measure density ratios. The two could be combined: use TD-JEPA for representation learning and task conditioning, then use the density-ratio correction for better off-policy value estimation.

**Related work to follow**:
- Zero-Shot Adaptation of BFMs to Unseen Dynamics (Bobrin et al 2025)
- Fast Adaptation with Behavioral Foundation Models (Sikchi et al 2025) — 17 cites
- BFM-Zero: Promptable BFM for Humanoid Control (Li et al 2025) — 22 cites

---

## 8. Maximum Entropy Behavior Exploration for Sim2Real Zero-Shot RL

**arXiv:2603.25464** — *Jiajun Hu, Nuria Armengol Urpi, Jin Cheng, Stelian Coros* (2026)

**Abstract**: Online zero-shot RL for quadrupedal control on real robots, building on the FB algorithm. Undirected exploration yields low-diversity data, leading to poor downstream performance and policies impractical for hardware deployment. They introduce **FB-MEBE**, an online zero-shot RL algorithm combining unsupervised behavior exploration (maximizing entropy of achieved behavior distribution) with a regularization critic that shapes recovered policies toward natural, physically plausible behaviors. Demonstrates improved performance in simulated downstream tasks and natural policies deployable to hardware without finetuning.

**Connection to TD-JEPA**: Both are FB-family zero-shot RL approaches. TD-JEPA is offline-only (no online exploration). This paper shows the value of online exploration for real-world deployment. The maximum-entropy behavior exploration could be integrated into TD-JEPA's training pipeline if online data collection is available. The regularization critic for "natural behaviors" is relevant for sim-to-real transfer of TD-JEPA policies.

**Related work to follow**:
- π*0.6: a VLA That Learns From Experience (Physical Intelligence 2025) — 116 cites
- BFM-Zero: Promptable BFM for Humanoid Control (Li et al 2025)
- Zero-Shot Whole-Body Humanoid Control via BFMs (Tirinzoni et al 2025) — 47 cites

---

## 9. Bonus: Representational Effective Theory for LLMs

**arXiv:2605.09294** — *Muhammed Ustaomeroglu, Guannan Qu* (2026)

**Abstract**: They propose **Representational Effective Theory (RET)**, a framework for describing LLM computation in terms of learned macrostates rather than microscopic details. RET learns macrostates from hidden-state trajectories using a **BYOL/JEPA-style self-supervised objective**, coarse-graining activations into macrovariables that preserve higher-level structure relevant for prediction and interpretation. These macrovariables reveal "mental-state" trajectories of reasoning, capture semantic structure, support early prediction of behavioral outcomes (e.g., sycophancy), and provide causal handles for steering generations.

**Connection to TD-JEPA**: RET uses the same JEPA-style self-supervised learning for representation learning, but applies it to LLM interpretability rather than RL. This validates the JEPA paradigm as a general framework for learning useful macroscopic representations from sequential data. The "effective theory" perspective (coarse-graining while preserving relevant structure) is the same principle underlying TD-JEPA's orthonormal encoders.

---

## Synthesis: Common Themes Across All Papers

### Theme 1: Zero-Shot Task Adaptation
Every paper in this collection addresses zero-shot or few-shot adaptation to new tasks without task-specific training. The mechanisms differ:
- **TD-JEPA**: Reward inference via least-squares projection onto learned ψ-space
- **Jumpy World Models**: Policy composition via multi-step dynamics prediction
- **Hierarchical Planning**: Multi-scale MPC with latent world models
- **Soft FB**: General utility optimization via maximum-entropy representations
- **Laplacian Keyboard**: Option stitching from spectral basis
- **Off-Policy ZSRL**: Successor measure density ratios

### Theme 2: World Models & Latent Dynamics
Most papers learn world models in latent space:
- TD-JEPA: Forward predictor for Q-values
- HaM-World: Hamiltonian-structured latent dynamics
- Hierarchical Planning: Multi-scale latent models
- Jumpy Models: Multi-step occupancy predictions

The key challenge across all: **stable long-horizon prediction** without compounding errors.

### Theme 3: Representation Learning & Orthogonality
Multiple papers use orthogonal or structured representations:
- TD-JEPA: L2-normalized encoders with orthonormality regularizer
- Laplacian Keyboard: Graph Laplacian eigenvectors as orthogonal basis
- HaM-World: Canonical (q,p) decomposition
- RET: JEPA-style coarse-grained macrostates

### Theme 4: The FAIR/Meta Behavioral Foundation Model Ecosystem
There's a tight cluster of papers from the same research group:
- **TD-JEPA** (Bagatella, Pirotta, Touati)
- **Jumpy World Models** (Farebrother, Pirotta, Tirinzoni, Touati, Bellemare, Lazaric)
- **Hierarchical Planning** (Zhang, Terver, Zholus, LeCun, Ballas)
- **Soft FB** (Bagatella, Rupf, Martius, Krause)
- **FB-MEBE / Zero-Shot Humanoid** (Hu, Urpi, Tirinzoni, Touati)

These form a coherent research program: learning general-purpose behavioral representations from offline data, then composing/planning/conditioning on them at test time.

### Theme 5: JEPA as a Unifying Framework
JEPA (Joint-Embedding Predictive Architecture) appears explicitly or implicitly in:
- TD-JEPA (the namesake)
- Hierarchical Planning (LeCun's group)
- RET (BYOL/JEPA-style objective for LLMs)

The JEPA paradigm — predict in representation space rather than pixel/reconstruction space — is emerging as a core architecture for both embodied AI and foundation model interpretability.

---

## Recommended Reading Order for TD-JEPA Practitioners

1. **Start**: TD-JEPA (Bagatella et al 2025) — the core architecture
2. **Extensions**: Soft FB (Bagatella et al 2026) — general utilities beyond additive rewards
3. **Planning**: Jumpy World Models (Farebrother et al 2026) — temporal abstraction for composition
4. **Hierarchical**: Hierarchical Planning with Latent WMs (Zhang et al 2026) — multi-scale MPC
5. **Stability**: HaM-World (Tang et al 2026) — Hamiltonian structure for long-horizon stability
6. **Off-policy**: Zero-Shot Off-Policy Learning (Asadulaev et al 2026) — density ratio correction
7. **Exploration**: FB-MEBE (Hu et al 2026) — maximum entropy exploration for real robots
8. **Options**: Laplacian Keyboard (Chandrasekar & Machado 2026) — spectral basis for option discovery
9. **Theory**: RET (Ustaomeroglu & Qu 2026) — JEPA-style macrostates for LLMs

---

## Citation Summary Table

| Paper | arXiv ID | Key Contribution | Citations (Semantic Scholar) |
|-------|----------|------------------|------------------------------|
| TD-JEPA (base) | 2025 (not in this list) | Latent-predictive zero-shot RL | ~8 |
| Compositional Planning | 2602.19634 | Jumpy world models for policy composition | — |
| Hierarchical Planning | 2604.03208 | Multi-scale latent world models + MPC | — |
| HaM-World | 2605.05951 | Hamiltonian + selective memory world models | — |
| Soft FB | 2602.06769 | General utilities via max-entropy FB | 1 |
| Laplacian Keyboard | 2602.07730 | Spectral options beyond linear span | — |
| Zero-Shot Off-Policy | 2602.01962 | Successor measure density ratios | — |
| FB-MEBE | 2603.25464 | Max-entropy exploration for sim2real | — |
| RET | 2605.09294 | JEPA macrostates for LLM interpretability | — |

---

*Compiled from arXiv metadata and Semantic Scholar citation graphs, May 2026.*
