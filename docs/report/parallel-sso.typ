#import "@preview/charged-ieee:0.1.4": ieee
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *

#show: codly-init.with()

#show: ieee.with(
  title: [Parallel Shark Smell Optimization],
  abstract: [
    The Shark Smell Optimization (SSO) algorithm is a metaheuristic optimization technique inspired by the hunting behavior of sharks. Due to the high computational cost of population-based optimization, this paper presents various parallel implementations of SSO using MPI, OpenMP and hybrid MPI+OpenMP programming models. The MPI version distributes the population across processes, while the hybrid approach combines distributed-memory and shared-memory parallelism to improve resource utilization on multicore systems. Performance evaluation on standard benchmark functions analyzes execution time, speedup, and scalability for different population sizes and processor configurations.

    > Qui scriviamo riassunto sui risultati
  ],
  authors: (
    (
      name: "Pietro Cipriani",
      location: [000000],
      email: "pietro.cipriani@studenti.unitn.it"
    ),
    (
      name: "Filippo De Grandi",
      location: [257824],
      email: "filippo.degrandi@studenti.unitn.it"
    ),
    (
      name: "Giacomo Vettore",
      location: [000000],
      email: "giacomo.vettore@studenti.unitn.it"
    ),
  ),
  index-terms: ("parallelization", "hpc", "mpi", "openmp", "sharks", "smelling"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction
/*
An accurate description of the assigned project should be provided, including analysis of the sequential algorithm that solves the problem addressed in the project.
- Pseudo-code, examples, graphs, figures, application instances etc. may be provided.
*/

Meta-heuristic optimization algorithms have become a widespread tool for solving complex nonlinear optimization problems where deterministic or gradient-based approaches are computationally infeasible #footnote[We found this "bestiary" with a particular hot-take on meta-heuristic algorithms @ec-bestiary]. Population-based optimization methods are effective in exploring high-dimensional search spaces and handling multimodal objective functions. These techniques are widely applied in engineering optimization, machine learning, scheduling, computational biology, and scientific computing.

\

Although the Shark Smell Optimization (SSO) algorithm demonstrates strong optimization capabilities on nonlinear benchmark functions, its iterative population-based structure introduces substantial computational costs as the number of sharks, problem dimensionality, and iteration count increase. Parallelization becomes essential to reduce execution time and improve scalability.

\

This paper investigates multiple parallel implementations of the Shark Smell Optimization algorithm using Message Passing Interface (MPI), OpenMP, and hybrid MPI+OpenMP programming models. In addition to comparing distributed-memory and shared-memory paradigms, the study analyzes different parallelization strategies within the algorithm itself. Specifically, parallel execution is applied at three distinct levels:

- `shark-level` parallelism, where candidate solutions are distributed across population elements;
- `dimension-level` parallelism, where computations across search-space dimensions are parallelized;
- `hybrid` parallelism, combining both shark-level and dimension-level decomposition.

By combining these algorithmic decomposition strategies with MPI, OpenMP, and hybrid MPI+OpenMP execution models, the work evaluates multiple parallel variants of SSO and analyzes their scalability, synchronization behavior, communication overhead, and computational efficiency on multicore HPC architectures.

== The Shark Smell Optimization Algorithm

The SSO @sso algorithm is a population-based metaheuristic inspired by sharks' ability to locate prey by sensing and following odor gradients.

#figure(image("images/shark.png", width: 70%), caption: [Schematic illustration of a shark movement to odor source.])

The method maintains a population of candidate solutions (`shark`s) that move through the search space combining global exploration and focused local exploitation.

#pagebreak()

#codly(header: box(text(weight: "bold", [Shark data structure])), languages: codly-languages)
#show raw: set text(font: "JetBrains Mono", size: 8pt)
```C
struct Shark {
  double *position; // Current position of the shark.
  double *speed;    // Current speed of the shark.
  double pos_score; // The OF(position) value.
};
```

Each shark iteratively updates its position and speed based on a combination of inertia (momentum from previous movement) and attraction toward promising regions of the search space (global best or local neighborhood best). Additionally, a local rotational search is performed around improved sharks to refine their positions.

The main stages of the SSO algorithm are described in the following pseudo-code, which highlights the key computational steps:

#codly(header: box(text(weight: "bold", [Serial Algorithm])), languages: codly-languages)
#show raw: set text(font: "JetBrains Mono", size: 8pt)
```C
// Shark loop
for shark_index in 0 .. cfg.np-1:
  shark = sharks[shark_index]
  // Stages loop
  for k in 0 .. cfg.k_max-1:
    R1 = uniform(0,1)
    R2 = uniform(0,1)

    // 1) per-dimension derivative and speed update
    for dim in 0 .. cfg.nd-1:
      derivative = eval_derivative(shark.position, cfg.nd, cfg.obj, dim)
      grad_term = cfg.eta * R1 * derivative
      mom_term  = cfg.alpha * R2 * shark.speed[dim]
      shark.speed[dim] = grad_term + mom_term
      // apply velocity limiter using cfg.beta
      if abs(shark.speed[dim]) > abs(cfg.beta * shark.speed[dim]):
        shark.speed[dim] = cfg.beta * shark.speed[dim]

    // 2) forward movement and clamp
    for dim in 0 .. cfg.nd-1:
      shark.position[dim] = clamp(shark.position[dim] + shark.speed[dim], domain[dim])

    // 3) rotational local search
    best = OF(shark.position, cfg.nd, cfg.obj)
    best_r3 = 0
    for m in 0 .. cfg.rotations-1:
      r3 = uniform(-1,1)
      for dim in 0 .. cfg.nd-1:
        candidate[dim] = clamp(shark.position[dim] * (1 + r3), domain[dim])
      val = OF(candidate, cfg.nd, cfg.obj)
      if val > best: best = val; best_r3 = r3
    if best_r3 != 0:
      for dim in 0 .. cfg.nd-1:
        shark.position[dim] = clamp(shark.position[dim] * (1 + best_r3), domain[dim])

    // 4) update global best
    cur_min = -best
    if cur_min < best_min:
      best_min = cur_min
      copy(best_pos, shark.position)

// result in best_min, best_pos
```

=== Design intuition
- Inertia preserves momentum and enables smooth exploration of the search space.
- Attraction terms bias motion toward high-quality regions (global best or local neighborhood best).
- Rotational local search acts as an exploitation operator similar to a focused local search or mutation that samples around a candidate on a small hypervolume to refine its position.


= Parallel Design

The SSO algorithm contains various sources of parallelism. The evaluation and update of individual sharks can generally be executed independently, making shark-level decomposition highly suitable for distributed-memory execution. Conversely, dimension-level decomposition is a more complicated form of parallelism that can be applied within the evaluation of a single shark's position and speed updates, which may be more efficient in shared-memory environments. Hybrid approaches can combine these strategies to leverage both inter-node and intra-node parallelism. \

This work therefore investigates the interaction between:

- the programming model:
  - MPI,
  - OpenMP,
  - hybrid MPI+OpenMP;

- and the algorithmic decomposition strategy:
  - shark-level parallelization,
  - dimension-level parallelization,
  - hybrid shark-dimension parallelization.

The comparative evaluation of these configurations enables a detailed analysis of scalability, workload distribution, communication overhead, synchronization costs, and overall parallel efficiency.

== Shark-level Parallelism

== Dimension-level Parallelism

== Parallelism over Stages

Although the outer "stage" loop over the stage index `k` might at first seem parallelizable, it is inherently sequential: each iteration at stage `k` consumes and modifies data produced at stage `k-1` (for example updated shark positions, velocities and fitness values). These true data dependencies prevent concurrent execution of different `k` iterations without violating correctness. Any approach to overlap would require complex checkpointing or synchronization that typically outweighs potential gains. Therefore, parallelism is applied at the shark and dimension levels rather than across stage iterations.

/*
Preliminary study about the opportunities for parallelism inherent in the problem sequential algorithm.
- State of the art analysis should be performed on parallel design strategies
- Alternative designs, related to different parallelization strategies, should be considered and discussed at this stage, appropriately motivating why some operations lend themselves to effective parallelization and others do not as well as why some data structures may or may not minimize the burden of communication and synchronization.
- Reference to prior work as well as link with related work must be clearly stated in the report.
- Hybrid parallelization strategies, with data dependencies must be discussed too
*/

= Implementation
/*
Discuss pros and cons of each strategy / implementation. The report can include some code snippets related to the most critical and interesting parts.
- A link to the repo must be provided too.
- Hybrid parallelization is recommended though it is not mandatory
*/

= Performance and Scalability Analysis
/*
The student must analyze the performance of the developed implementation in terms of execution time, speedup, and efficiency.
- Both strong scalability and weak scalability should be evaluated where possible.
