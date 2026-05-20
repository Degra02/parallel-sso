#import "@preview/charged-ieee:0.1.4": ieee
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *

#show: codly-init.with()
#let code-size = 7pt

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
- `hybrid` parallelism, combining both the other strategies.

By combining these algorithmic decomposition strategies with MPI, OpenMP, and hybrid MPI+OpenMP execution models, the work evaluates multiple parallel variants of SSO and analyzes their scalability, synchronization behavior, communication overhead, and computational efficiency on multicore HPC architectures.

== The Shark Smell Optimization Algorithm

The SSO @sso algorithm is a population-based metaheuristic inspired by sharks' ability to locate prey by sensing and following odor gradients.

#figure(image("images/shark.png", width: 70%), caption: [Schematic illustration of a shark's movement to odor source.])

The method maintains a population of candidate solutions (`shark`s) that move through the search space combining global exploration and focused local exploitation.

#pagebreak()

#figure(
  [
    #codly(languages: codly-languages)
    #show raw: set text(font: "JetBrains Mono", size: code-size)
    ```C
    struct Shark {
      double *position; // Current position of the shark.
      double *speed;    // Current speed of the shark.
      double pos_score; // The OF(position) value.
    };
    ```
  ],
  caption: [Shark data structure]
)

Each shark iteratively updates its position and speed based on a combination of inertia (momentum from previous movement) and attraction toward promising regions of the search space (global best or local neighborhood best). Additionally, a local rotational search is performed around improved sharks to refine their positions.

The main stages of the SSO algorithm are described in the following pseudo-code, which highlights the key computational steps:


#codly(header: box(text(weight: "bold", [Serial Algorithm])), languages: codly-languages)
#show raw: set text(font: "JetBrains Mono", size: code-size)
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
/*
Preliminary study about the opportunities for parallelism inherent in the problem sequential algorithm.
- State of the art analysis should be performed on parallel design strategies
- Alternative designs, related to different parallelization strategies, should be considered and discussed at this stage, appropriately motivating why some operations lend themselves to effective parallelization and others do not as well as why some data structures may or may not minimize the burden of communication and synchronization.
- Reference to prior work as well as link with related work must be clearly stated in the report.
- Hybrid parallelization strategies, with data dependencies must be discussed too
*/

The SSO algorithm contains various sources of parallelism. The evaluation and update of individual sharks can generally be executed independently, making shark-level decomposition highly suitable for distributed-memory execution. Conversely, dimension-level decomposition is a more complicated form of parallelism that can be applied within the evaluation of a single shark's position and speed updates, which may be more efficient in shared-memory environments. Hybrid approaches can combine these strategies to leverage both inter-node and intra-node parallelism. \

In the following graph, we illustrate the slowdown of the serial algorithm as various parameters (number of sharks, dimensions, iterations) increase.

#figure(image("images/serial_raw_slowdown.png", width: 100%), caption: [Execution time of the serial SSO algorithm as a function of the number of sharks, dimensions, stages and rotations.])

The graph above underlines how the SSO algorithm would benefit from parallelization, especially as the problem size increases. The computational cost grows significantly with the number of sharks, dimensions, and iterations, making it crucial to explore parallel design strategies to achieve practical execution times for larger problem instances.

Although the outer "stage" loop over the stage index `k`, in the graph indicated by the red line, might at first seem parallelizable, it is inherently sequential: each iteration at stage `k` consumes and modifies data produced at stage `k-1` (for example updated shark positions, velocities and fitness values). These true data dependencies prevent concurrent execution of different `k` iterations without violating correctness. Any approach to overlap would require complex checkpointing or synchronization that typically outweighs potential gains. Therefore, parallelism is applied at the shark and dimension levels rather than across stage iterations.

\

This work therefore investigates the interaction between:

- the programming model:
  - MPI,
  - OpenMP,
  - hybrid MPI+OpenMP;

- and the algorithmic decomposition strategy:
  - `shark-level` parallelization,
  - `dimension-level` parallelization,
  - `rotation-level` parallelization,
  - `hybrid` parallelization.

\

The comparative evaluation of these configurations enables a detailed analysis of scalability, workload distribution, communication overhead, synchronization costs, and overall parallel efficiency.

== Shark-level Parallelism

== Dimension-level Parallelism

== Rotation-level Parallelism

= Implementation
/*
Discuss pros and cons of each strategy / implementation. The report can include some code snippets related to the most critical and interesting parts.
- A link to the repo must be provided too.
- Hybrid parallelization is recommended though it is not mandatory
*/

All code and scripts related to the project are available in a public repository @parallel-sso-repo.

== MPI Implementations

== OpenMP Implementations

== Hybrid MPI+OpenMP Implementations


= Performance and Scalability Analysis
/*
The student must analyze the performance of the developed implementation in terms of execution time, speedup, and efficiency.
- Both strong scalability and weak scalability should be evaluated where possible.
*/

In this section, we present a performance analysis of the various parallel implementations of the SSO algorithm. We evaluate execution time, speedup and efficiency across the different parallelization strategies and programming models. The experiments are conducted on a multicore HPC cluster, and we analyze both strong and weak scalability.
The section is organized as follows:
- Experimental setup and benchmark functions
- Analysis with varying number of processes
- Analysis with varying number of threads
- Analysis with both processes and threads (hybrid)
- Analysis with different execution modes in the PBS system.

== Experimental Setup and Benchmark Functions

To standardize our performance evaluation, we decided to stick with a fixed set of parameters for the SSO algorithm across all implementations. The set has been chosen to be sufficiently large to allow for meaningful performance measurements while keeping execution times manageable. The parameters are as follows:
- Number of sharks (`np`): 1000
- Number of dimensions (`nd`): 200
- Number of stages (`k_max`): 1000
- Number of rotations (`rotations`): 50
\

The rest of the parameters (such as `eta`, `alpha`, `beta`) are set to their default values as defined in the code, that have been taken from the original SSO paper.

In order to launch several tests with different parameters, we developed a set of scripts that automate the execution of the various implementations on the HPC cluster. The scripts allow us to easily vary the number of chunks to select on the cluster, processes and threads used in the MPI and OpenMP methods.
Regarding the execution modes in the PBS system, we decided to stick with the `excl` mode, which ensures that the allocated resources are not shared with other jobs. This choice allows us to minimize interference and obtain more consistent performance measurements.

#figure([
  #codly(languages: codly-languages)
  #show raw: set text(font: "JetBrains Mono", size: code-size)
  ```bash
  ...
  for procs in "${PROC_VALUES[@]}"; do
    for thrds in "${THRD_VALUES[@]}"; do
      while [[ $(qstat -u $USER | wc -l) -ge 30 ]]; do
        sleep 10
      done
      sed \
        -e "s/\${N_THRD}/$thrds/g" \
        -e "s/\${N_PROC}/$procs/g" \
        -e "s/\${JOB_NAME}/$JOB_NAME/g" \
        -e "s/\${N_CPUS}/$N_CPUS/g" \
        -e "s/\${PLACE}/$PLACE/g" \
        "$EXEC" | qsub
    done
  done
```],
caption: [A portion of the script used to launch the tests on the HPC cluster. The script iterates over different values of processes and threads (user-defined), checks for the number of running jobs, and submits new jobs using `qsub` with the appropriate parameters.]
)<launch_script>

#figure(
  [
    #codly(languages: codly-languages)
    #show raw: set text(font: "JetBrains Mono", size: code-size)
    ```bash
    #!/bin/env bash
    #PBS -l select=${N_CPUS}:ncpus=${N_PROC}:mpiprocs=${N_PROC}:mem=2gb
    #PBS -l place=${PLACE}:excl
    #PBS -l walltime=0:5:00
    #PBS -q shortCPUQ

    # output & error files
    #PBS -N ${JOB_NAME}_${N_PROC}_${N_THRD}

    module load OpenMPI/4.1.6-GCC-13.2.0

    mpirun -n $(( ${N_CPUS} * ${N_PROC} )) ./parallel-sso/sso_hybrid_sharks -p 1000 -d 200 -k 1000 -m 50 -t ${N_THRD}
    ```
  ],
  caption: [An example PBS script, edited by @launch_script. The script specifies resource requirements, loads necessary modules, and runs the executable with the defined parameters for sharks, dimensions, stages, rotations, and threads.]
) <pbs_script>


The implementation contains three different objective functions (in `sso/ofuncs.c`), taken from the original SSO paper, that are used for testing the algorithm's performance and scalability. In our experiments, we used the Rastrigin function @rastrigin (set as default in the code), as the differences between the various objective functions were negligible in terms of execution time and speedup.

#figure(image("images/Rastrigin.png", width: 100%), caption: [Rastrigin function, a common benchmark for optimization algorithms.])


== Varying number of processes

== Varying number of threads

== Hybrid parallelism

== Playing with PBS

= Conclusion

This paper presented a comprehensive study of parallel implementations of the Shark Smell Optimization algorithm using MPI, OpenMP, and hybrid MPI+OpenMP programming models. By exploring different levels of parallelism (shark-level, dimension-level, rotation-level and hybrid), we analyzed the trade-offs between computational efficiency, communication overhead, and synchronization costs.

Performance evaluation on standard benchmark functions demonstrated significant reductions in execution time and improved scalability with increasing problem size and processor count. Future work could explore more advanced parallelization strategies, such as dynamic load balancing or GPU acceleration, to further enhance the performance of SSO on large-scale optimization problems.
