# Job-Shop Scheduling Optimization using Genetic Algorithm and Mixed Integer Programming

## Executive Summary

This report analyses the **Job-Shop Scheduling Problem (JSSP)** using the classical **Fisher and Thompson FT06** benchmark instance. The objective is to minimise makespan, defined as the completion time of the final operation in the schedule. FT06 contains **6 jobs**, **6 machines**, and **36 operations**, with a known optimal makespan of **55**.

Two optimisation approaches were implemented and compared: a **Genetic Algorithm (GA)** and a **Mixed Integer Programming (MIP)** formulation. Both methods produced feasible schedules with makespan **55**, giving a **0.00% optimality gap** against the known FT06 optimum. The GA reached the optimum through an operation-based chromosome, schedule decoder, tournament selection, Job Order Crossover, swap mutation, elitism, and early stopping. The MIP model reached the same objective value and proved optimality using start-time variables, binary sequencing variables, precedence constraints, and machine non-overlap constraints.

The comparison shows that MIP is the stronger method when a proof of optimality is required and the problem size is small enough for exact solution. GA is more flexible and scalable for larger or more dynamic scheduling environments where a high-quality schedule is more practical than waiting for a proof of optimality. A hybrid GA-MIP approach is a strong future direction because GA can generate strong feasible schedules while MIP can validate, improve, or bound them.

## 1. Introduction and Optimization Context

### 1.1 Problem Context

The **Job-Shop Scheduling Problem (JSSP)** is a classical production-scheduling problem in operations research. A set of jobs must be processed on a set of machines, where each job consists of a fixed sequence of operations. Each operation requires a specified machine for a known processing time. The scheduling decision is to determine the start time of every operation and the processing order of operations that compete for the same machine.

JSSP represents many real production environments, including machine shops, maintenance workshops, printing systems, semiconductor manufacturing, assembly operations, and resource-constrained service systems. In these environments, poor scheduling can increase waiting time, machine idle time, work-in-progress inventory, and late delivery risk. An optimized schedule improves throughput, machine utilisation, and delivery reliability.

This report compares a metaheuristic search method, **Genetic Algorithm**, with an exact mathematical optimization method, **Mixed Integer Programming**, on the same benchmark instance. This comparison is useful because JSSP has both practical importance and high combinatorial complexity.

### 1.2 Optimization Objective

The objective is to minimize **makespan**, which is the completion time of the final operation in the schedule:

$$
\min C_{\max}
$$

where $C_{\max}$ is the total schedule length. Minimising makespan means completing all production work as early as possible while respecting job-order and machine-capacity constraints.

### 1.3 Mathematical Formulation

Let:

- $J = \{1,2,\dots,n\}$ be the set of jobs.
- $M = \{1,2,\dots,m\}$ be the set of machines.
- $O_{jk}$ be operation $k$ of job $j$, where $j \in J$ and $k = 1,2,\dots,m$ for FT06.
- $p_{jk}$ be the processing time of operation $O_{jk}$.
- $\mu_{jk}$ be the machine required by operation $O_{jk}$.
- $H$ be a sufficiently large scheduling horizon. In this work, $H$ is the sum of all processing times.

The main decision variables are:

- $S_{jk} \geq 0$: start time of operation $O_{jk}$.
- $C_{\max} \geq 0$: makespan of the full schedule.
- $y_{jk,\ell r} \in \{0,1\}$: binary sequencing variable for a fixed ordered pair of operations $O_{jk}$ and $O_{\ell r}$ requiring the same machine.

The objective is:

$$
\min C_{\max}
$$

Subject to job precedence constraints:

$$
S_{j,k+1} \geq S_{jk} + p_{jk}
$$

for all $j \in J$ and $k = 1,\dots,m-1$.

Machine capacity is enforced using disjunctive Big-M constraints. For any two distinct operations $O_{jk}$ and $O_{\ell r}$ requiring the same machine, where $\mu_{jk}=\mu_{\ell r}$, using one fixed ordering for the pair:

$$
S_{jk} + p_{jk} \leq S_{\ell r} + H(1 - y_{jk,\ell r})
$$

$$
S_{\ell r} + p_{\ell r} \leq S_{jk} + H y_{jk,\ell r}
$$

These constraints ensure that one operation is processed before the other and that both cannot overlap on the same machine.

The makespan constraints are:

$$
C_{\max} \geq S_{jk} + p_{jk}
$$

for every operation $O_{jk}$.

The model also requires non-negativity and binary-domain constraints:

$$
S_{jk} \geq 0\quad(j\in J,\; k=1,\dots,m),\qquad
C_{\max} \geq 0,\qquad
y_{jk,\ell r} \in \{0,1\}\quad((j,k)<(\ell,r),\; \mu_{jk}=\mu_{\ell r})
$$

### 1.4 Practical Meaning of the Constraints

| Constraint Type | Practical Meaning |
| --- | --- |
| Job precedence | Operations of the same job must follow the required technological order. |
| Machine capacity | A machine cannot process more than one operation at the same time. |
| Non-preemption | Once an operation starts, it must finish without interruption. |
| Makespan | The schedule length must cover the completion time of all operations. |
| Binary sequencing | The model decides which operation goes first when two operations require the same machine. |

### 1.5 Literature Context

The JSSP has been studied using exact optimization, constructive heuristics, metaheuristics, and hybrid algorithms. The literature supports both exact and heuristic approaches because scheduling problems are structured enough for mathematical modelling but difficult enough that exact methods may not scale to large cases.

| Study | Method / Focus | Relevance to This Analysis |
| --- | --- | --- |
| Garey, Johnson, and Sethi (1976) | Complexity of flow-shop and job-shop scheduling | Establishes the computational difficulty of scheduling problems and explains why heuristic search is needed for larger instances. |
| Adams, Balas, and Zawack (1988) | Shifting Bottleneck Procedure | Shows the importance of bottleneck machines and problem-specific scheduling heuristics. |
| Cheng, Gen, and Tsujimura (1996/1999) | GA representation and hybrid GA strategies for JSSP | Supports careful chromosome design, crossover, mutation, and decoding for feasible job-shop schedules. |
| Goncalves, Mendes, and Resende (2005) | Hybrid GA with random keys and local search | Demonstrates that GA can perform strongly on JSSP benchmarks when combined with scheduling-specific decoding. |
| Ku and Beck (2016) | MIP formulations for the classical JSSP | Supports using MIP as a rigorous exact benchmark and highlights scalability limitations. |
| Cebi, Atac, and Sahingoz (2020) | Review of JSSP solution algorithms | Places GA and MIP within the wider set of exact, heuristic, and metaheuristic approaches. |
| King and Hildebrand (2024) | Integer programming, shifting bottleneck, and decision diagrams | Shows that comparative and hybrid scheduling approaches remain relevant in recent research. |

The literature confirms that exact methods such as MIP are valuable because they can prove optimality for small and moderate instances. However, the binary sequencing structure grows rapidly as problem size increases. Metaheuristics such as GA do not guarantee optimality, but they can explore large solution spaces efficiently and can be adapted to complex real-world constraints.

### 1.6 Modelling Strategy

The analysis uses both GA and MIP because the two methods answer different questions. GA tests whether an evolutionary search can find a high-quality feasible schedule efficiently. MIP provides a mathematically rigorous benchmark and, for FT06, can prove whether the obtained makespan is globally optimal.

GA is suitable because JSSP is naturally sequence-based. A chromosome can represent an operation sequence, and a decoder can convert that sequence into a feasible schedule. MIP is suitable because JSSP can be formulated using start-time variables, binary sequencing variables, precedence constraints, and machine-capacity constraints.

| Aspect | Genetic Algorithm | Mixed Integer Programming |
| --- | --- | --- |
| Method type | Metaheuristic search | Exact mathematical optimization |
| Main strength | Flexibility and scalability | Feasibility control and optimality proof |
| Main limitation | No guaranteed optimality | Scalability weakens as binary variables grow |
| Analytical role | Practical search-based optimizer | Exact benchmark and validation method |

## 2. Benchmark Data and Problem Analysis

### 2.1 Benchmark Instance

The benchmark instance used for this analysis is the **Fisher and Thompson FT06** instance, also known as MT06 in some job-shop benchmark libraries. It is a classic deterministic JSSP instance with:

| Property | Value |
| --- | ---: |
| Jobs | 6 |
| Machines | 6 |
| Operations per job | 6 |
| Total operations | 36 |
| Known optimal makespan | 55 |
| Scheduling objective | Minimize makespan |

FT06 is useful for comparative analysis because it is small enough for exact MIP solution, but large enough to demonstrate GA representation, decoding, crossover, mutation, convergence, and schedule feasibility.

### 2.2 Job Routes and Processing Times

Each row below shows the machine and processing time required by the six operations of each job.

| Job | Operation Sequence |
| --- | --- |
| J1 | (M3, 1), (M1, 3), (M2, 6), (M4, 7), (M6, 3), (M5, 6) |
| J2 | (M2, 8), (M3, 5), (M5, 10), (M6, 10), (M1, 10), (M4, 4) |
| J3 | (M3, 5), (M4, 4), (M6, 8), (M1, 9), (M2, 1), (M5, 7) |
| J4 | (M2, 5), (M1, 5), (M3, 5), (M4, 3), (M5, 8), (M6, 9) |
| J5 | (M3, 9), (M2, 3), (M5, 5), (M6, 4), (M1, 3), (M4, 1) |
| J6 | (M2, 3), (M4, 3), (M6, 9), (M1, 10), (M5, 4), (M3, 1) |

The processing-time matrix is:

| Job | O1 | O2 | O3 | O4 | O5 | O6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J1 | 1 | 3 | 6 | 7 | 3 | 6 |
| J2 | 8 | 5 | 10 | 10 | 10 | 4 |
| J3 | 5 | 4 | 8 | 9 | 1 | 7 |
| J4 | 5 | 5 | 5 | 3 | 8 | 9 |
| J5 | 9 | 3 | 5 | 4 | 3 | 1 |
| J6 | 3 | 3 | 9 | 10 | 4 | 1 |

The machine-route matrix is:

| Job | O1 | O2 | O3 | O4 | O5 | O6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J1 | M3 | M1 | M2 | M4 | M6 | M5 |
| J2 | M2 | M3 | M5 | M6 | M1 | M4 |
| J3 | M3 | M4 | M6 | M1 | M2 | M5 |
| J4 | M2 | M1 | M3 | M4 | M5 | M6 |
| J5 | M3 | M2 | M5 | M6 | M1 | M4 |
| J6 | M2 | M4 | M6 | M1 | M5 | M3 |

The total processing time is:

$$
H = \sum_{j\in J}\sum_{k=1}^{m} p_{jk} = 197
$$

This is used as the Big-M scheduling horizon in the MIP formulation.

### 2.3 Workload and Lower-Bound Analysis

Machine workload identifies possible bottleneck resources.

| Machine | Total Workload |
| --- | ---: |
| M1 | 40 |
| M2 | 26 |
| M3 | 26 |
| M4 | 22 |
| M5 | 40 |
| M6 | 43 |

Machine M6 has the largest workload, with 43 time units. Job workload analysis gives:

| Job | Total Processing Time |
| --- | ---: |
| J1 | 26 |
| J2 | 47 |
| J3 | 34 |
| J4 | 35 |
| J5 | 25 |
| J6 | 30 |

The longest single job is J2 with 47 time units. Therefore, a simple lower bound on the makespan is:

$$
C_{\max} \geq \max(43, 47) = 47
$$

The known optimum is 55, which is only 8 time units above this simple lower bound. This indicates that machine conflicts and operation precedence constraints add significant scheduling complexity beyond raw processing workload.

### 2.4 Complexity Analysis

The solution space is very large even for FT06. If each of the 6 machines can process the 6 jobs in any order, a rough machine-ordering search space is:

$$
(6!)^6 = 720^6 \approx 1.39 \times 10^{17}
$$

For an operation-based chromosome where 36 positions contain six copies of each job, the number of possible chromosomes is approximately:

$$
\frac{36!}{(6!)^6} \approx 2.67 \times 10^{24}
$$

This explains why exhaustive enumeration is not practical and why GA and MIP search procedures are required.

### 2.5 Preprocessing and Assumptions

The FT06 data was converted into two representations:

- A processing-time matrix for GA decoding and MIP duration parameters.
- A machine-route matrix for assigning each operation to its required machine.

Machine identifiers were represented using 1-based numbering for readability in the report and converted internally where needed for Python implementation. The main assumptions are:

- Processing times are deterministic and known.
- Each job visits each required machine once.
- Each operation is non-preemptive.
- Machines are continuously available from time 0.
- No setup times, due dates, maintenance windows, worker constraints, or machine breakdowns are included.

## 3. Genetic Algorithm Method and Results

### 3.1 Method Overview

The Genetic Algorithm searches for a good operation sequence. The chromosome itself does not contain start times. Instead, a decoder reads the chromosome from left to right and schedules each operation at the earliest feasible time.

The GA follows these steps:

1. Create an initial population of valid chromosomes.
2. Decode each chromosome into a feasible schedule.
3. Evaluate each schedule using makespan.
4. Select parents using tournament selection.
5. Apply Job Order Crossover.
6. Apply swap mutation.
7. Preserve elite chromosomes.
8. Repeat until the generation limit or early stopping condition is met.

### 3.2 Chromosome Representation

The GA uses an **operation-based chromosome** of length 36. Each job number appears exactly 6 times because each job has 6 operations. For example:

```text
[2, 3, 3, 3, 6, 1, 4, 1, 6, 2, 2, 5, ...]
```

When a job number appears for the first time, it represents that job's first unscheduled operation. The second appearance represents its second operation, and so on. This representation preserves job operation order through the decoder, so chromosomes can be converted into feasible schedules.

### 3.3 Schedule Decoding

For each gene in the chromosome, the decoder schedules the next unscheduled operation of that job. The start time is:

$$
S_{jk} = \max\{A_j, A_{\mu_{jk}}\}
$$

where $A_j$ is the current availability time of job $j$, and $A_{\mu_{jk}}$ is the current availability time of the machine required by operation $O_{jk}$.

This ensures that:

- The previous operation of the same job has finished.
- The required machine is available.
- No two operations overlap on the same machine.
- Every job follows its required technological route.

### 3.4 Fitness Function

The objective is to minimize makespan. Fitness can be represented as:

$$
\operatorname{Fitness}(x) = \frac{1}{C_{\max}(x)}
$$

In the implementation, chromosomes are ranked directly by makespan. Lower makespan is better.

### 3.5 Selection, Crossover, Mutation, and Elitism

| GA Component | Design Used |
| --- | --- |
| Population initialization | Random shuffle of six copies of each job number |
| Selection | Tournament selection with tournament size 3 |
| Crossover | Job Order Crossover (JOX) |
| Mutation | Swap mutation |
| Elitism | Best 4 chromosomes copied to the next generation |
| Stopping | Maximum 700 generations or 250 generations without improvement |

Tournament selection is simple and maintains selection pressure toward lower makespan. JOX preserves the relative structure of selected jobs from one parent while filling the remaining jobs from the other parent. Swap mutation exchanges two chromosome positions and helps maintain diversity.

### 3.6 Hyperparameter Tuning

The tuning process considered several values for population size, crossover probability, mutation probability, generation count, and elite size.

| Hyperparameter | Tested Values | Final Value |
| --- | --- | ---: |
| Population size | 50, 100, 150 | 150 |
| Crossover probability | 0.80, 0.90, 0.95 | 0.90 |
| Mutation probability | 0.10, 0.20, 0.25, 0.30 | 0.25 |
| Maximum generations | 300, 500, 700 | 700 |
| Elite size | 2, 4 | 4 |
| Tournament size | 3 | 3 |
| Early stopping patience | 150, 250 | 250 |

The selected configuration balances solution quality and runtime. A larger population improves search diversity, while a moderate mutation probability helps avoid premature convergence.

### 3.7 Results

The final GA configuration was executed on FT06 using seed 0. The generated run produced the following result:

| Metric | Value |
| --- | ---: |
| Best makespan | 55 |
| Known optimum | 55 |
| Optimality gap | 0.00% |
| Feasibility | Feasible |
| Runtime in local run | 13.00 seconds |
| Generations completed | 270 |
| First generation reaching makespan 55 | 20 |

The best 1-based chromosome obtained was:

```text
[2, 3, 3, 3, 6, 1, 4, 1, 6, 2, 2, 5, 4, 4, 5, 6, 3, 5,
 6, 1, 4, 2, 5, 4, 3, 2, 1, 1, 3, 4, 6, 5, 6, 1, 2, 5]
```

The optimality gap is:

$$
\operatorname{Gap} =
\frac{C_{\max}^{\mathrm{GA}}-C_{\max}^{*}}{C_{\max}^{*}} \times 100
= \frac{55-55}{55} \times 100 = 0\%
$$

This confirms that the GA found an optimal schedule for FT06 in this run. The GA itself does not mathematically prove optimality, but the result matches the known FT06 optimum and the MIP optimum.

### 3.8 Convergence Analysis

![GA convergence on FT06](question3_report_assets/ga_convergence.png)

The convergence plot shows a steep improvement during the early generations. The initial best makespan was 62 and the initial average makespan was approximately 89.53. The best makespan reached 55 by generation 20. After that, the best curve remained flat at the known optimum because no better schedule exists for FT06. The average makespan continued to fluctuate around the high-50s because crossover and mutation still introduced weaker chromosomes, while elitism preserved the best solution.

This behavior indicates that the selected representation and operators were effective for this small benchmark instance. The stopping rule ended the run at generation 270 after 250 generations without further improvement.

### 3.9 Best Schedule

![Best GA schedule Gantt chart](question3_report_assets/ga_gantt.png)

The Gantt chart shows the best GA schedule with makespan 55. Each bar is one operation, grouped by machine. The chart visually confirms the two main feasibility properties: no machine processes overlapping operations, and all jobs follow their required operation sequence.

The schedule also shows that several machines remain busy close to the end of the horizon, especially M4, M5, and M6. The final operations finish at time 55, which matches $C_{\max}=55$.

The full decoded GA schedule is available in:

```text
question3_report_assets/ga_best_schedule.csv
```

### 3.10 Computational Efficiency and Scalability

The configured upper-bound evaluation effort is:

$$
150 \times 700 \times 36 = 3,780,000
$$

operation placements in a full run. Because early stopping ended this run at generation 270, the actual evaluation effort was approximately:

$$
150 \times 271 \times 36 = 1,463,400
$$

operation placements including the initial population. This is computationally manageable for FT06. The decoding step is linear in the number of operations, $O(nm)$, because each chromosome is scanned once.

For larger JSSP instances such as FT10, FT20, or Lawrence benchmarks, GA can still be used by increasing population size, generation count, and parallelizing chromosome evaluation. However, GA results depend on parameter tuning, random seed, representation, and operator design. GA also cannot independently prove that a found solution is globally optimal.

## 4. Mixed Integer Programming Method and Results

### 4.1 Method Overview

The same FT06 instance was solved using a MIP model. The formulation uses start-time variables for each operation and binary sequencing variables for pairs of operations that require the same machine. The model was implemented with **PuLP** and solved with the open-source **CBC** solver.

The MIP model directly minimizes $C_{\max}$, subject to precedence, machine non-overlap, and makespan constraints.

### 4.2 Compact MIP Formulation

Let $\mathcal{O}$ be the set of all operations. For each operation $a \in \mathcal{O}$:

- $S_a$ is the start time.
- $p_a$ is the processing time.
- $\mu_a$ is the required machine.

For each unordered pair of operations $\{a,b\}\subset \mathcal{O}$ requiring the same machine, with a fixed ordering $a<b$, define:

$$
x_{ab} =
\begin{cases}
1, & \text{if operation } a \text{ is scheduled before } b \\
0, & \text{otherwise}
\end{cases}
$$

The objective is:

$$
\min C_{\max}
$$

Precedence constraints:

$$
S_{j,k+1} \geq S_{jk} + p_{jk},\qquad j\in J,\; k=1,\dots,m-1
$$

Machine non-overlap constraints:

$$
S_a + p_a \leq S_b + H(1 - x_{ab}),\qquad \{a,b\}\subset \mathcal{O},\; a<b,\; \mu_a=\mu_b
$$

$$
S_b + p_b \leq S_a + H x_{ab},\qquad \{a,b\}\subset \mathcal{O},\; a<b,\; \mu_a=\mu_b
$$

Makespan constraints:

$$
C_{\max} \geq S_a + p_a,\qquad a\in\mathcal{O}
$$

Variable domains:

$$
S_a \geq 0\quad(a\in\mathcal{O}),\qquad C_{\max} \geq 0,\qquad
x_{ab} \in \{0,1\}\quad(\{a,b\}\subset \mathcal{O},\; a<b,\; \mu_a=\mu_b)
$$

No equality constraints are required for this deterministic FT06 formulation. Feasibility is represented through precedence, machine non-overlap, makespan, non-negativity, and binary sequencing constraints.

### 4.3 MIP Model Size

For FT06, the MIP model contains:

| Model Component | Count |
| --- | ---: |
| Operations | 36 |
| Start-time variables | 36 |
| Makespan variable | 1 |
| Binary sequencing variables | 90 |
| Total variables | 127 |
| Precedence constraints | 30 |
| Machine non-overlap constraints | 180 |
| Makespan constraints | 36 |
| Total constraints | 246 |
| Big-M horizon $H$ | 197 |

The 90 binary variables arise because each of the 6 machines has 6 operations assigned to it. Each machine therefore has:

$$
\binom{6}{2} = 15
$$

operation pairs, giving:

$$
6 \times 15 = 90
$$

binary sequencing variables.

### 4.4 MIP Results

The MIP model was solved using PuLP/CBC. The local solver run produced:

| Metric | Value |
| --- | ---: |
| Solver status | Optimal |
| Objective value / makespan | 55 |
| Known optimum | 55 |
| Optimality gap | 0.00% |
| Runtime in local run | 19.05 seconds |
| Feasibility | Satisfied |

The MIP result has:

$$
C_{\max} = 55
$$

Since this equals the known FT06 optimum, the MIP model proves optimality for the instance.

### 4.5 Optimal MIP Schedule

![Optimal MIP schedule Gantt chart](question3_report_assets/mip_gantt.png)

The MIP Gantt chart also finishes at time 55. It differs from the GA Gantt chart in some operation placements, but both are valid optimal schedules. This is expected because JSSP benchmark instances can have multiple optimal schedules with the same makespan.

The full MIP schedule is available in:

```text
question3_report_assets/mip_optimal_schedule.csv
```

### 4.6 Feasibility and Constraint Satisfaction

The MIP schedule satisfies all required feasibility conditions:

| Requirement | Result |
| --- | --- |
| Job precedence | Satisfied |
| Machine capacity | Satisfied |
| Non-overlapping operations | Satisfied |
| Non-negative start times | Satisfied |
| Makespan covers all operations | Satisfied |
| Binary sequencing decisions | Satisfied |

Because the solver status is optimal, the schedule is not only feasible but also proven optimal for the given formulation and instance.

### 4.7 MIP Scalability

For a general JSSP with $n$ jobs and $m$ machines, assuming each job visits each machine once:

| Instance Size | Operations | Approximate Binary Sequencing Variables |
| --- | ---: | ---: |
| 6 jobs x 6 machines | 36 | 90 |
| 10 jobs x 10 machines | 100 | 450 |
| 20 jobs x 5 machines | 100 | 950 |
| 20 jobs x 15 machines | 300 | 2,850 |

The number of binary variables grows as:

$$
m \times \binom{n}{2} = O(mn^2)
$$

As the number of binary variables grows, the branch-and-bound or branch-and-cut tree becomes larger. The solver may still find feasible schedules, but proving optimality can become much harder.

The Big-M value also affects computational performance. A very large $H$ weakens the linear relaxation and can slow down the solver, while a value that is too small can remove feasible schedules. In this report, $H=197$, the sum of all processing times, is a valid upper bound for FT06.

## 5. Comparative Findings and Practical Interpretation

### 5.1 Quantitative Comparison

Both GA and MIP produced an optimal makespan of 55 for FT06.

| Method | Best Makespan | Known Optimum | Optimality Gap | Feasible | Runtime |
| --- | ---: | ---: | ---: | --- | ---: |
| Genetic Algorithm | 55 | 55 | 0.00% | Yes | 13.00 s |
| MIP | 55 | 55 | 0.00% | Yes | 19.05 s |

![GA and MIP comparison](question3_report_assets/method_comparison.png)

The runtime values are local execution times and will vary by hardware, solver version, and environment. The important result is that both methods reached the same objective value. The key difference is that MIP proves optimality, while GA finds the optimum empirically and must be compared against a benchmark or exact method to confirm the gap.

### 5.2 Solution Quality

The solution quality of both methods is excellent for FT06 because both reached:

$$
C_{\max} = 55
$$

The GA result shows that the operation-based representation, decoder, tournament selection, JOX crossover, swap mutation, and elitism worked effectively for this benchmark. The MIP result validates the GA result by proving that no better schedule exists for FT06.

However, this equality should not be generalized to all job-shop instances. For larger or more constrained problems, GA may return a near-optimal schedule faster, while MIP may require a long time to prove optimality.

### 5.3 Computational Performance

GA has a predictable evaluation structure. Its computational effort mainly depends on:

$$
\text{Population size} \times \text{Generations} \times \text{Chromosome length}
$$

This makes runtime easier to control through hyperparameters. However, more generations or a larger population may be needed for harder instances.

MIP runtime is less predictable because it depends on the solver search tree, model formulation, Big-M strength, branching decisions, cuts, and whether the solver can close the optimality gap. For FT06, CBC solved the model to optimality in the local run, but larger instances may become much harder.

### 5.4 Scalability

| Problem Size | GA Behavior | MIP Behavior |
| --- | --- | --- |
| Small instance, e.g. FT06 | Can reach optimum with good tuning | Can solve and prove optimality |
| Medium instance | Can produce good schedules | Runtime may increase significantly |
| Large instance | Still usable with larger population or parallelism | May struggle to prove optimality |
| Dynamic or frequently changing instance | Flexible and easy to rerun | Full re-optimization may be expensive |

GA scales better in practical terms because chromosomes can be evaluated independently, making parallel evaluation possible. MIP is stronger for exactness but becomes harder as binary sequencing variables increase.

### 5.5 Ability to Handle Complex Constraints

Both methods can be extended to handle more realistic constraints, but they do so differently.

| Constraint Type | GA | MIP |
| --- | --- | --- |
| Setup times | Add to decoder and fitness function | Add to constraints |
| Due dates and tardiness | Add penalty terms to fitness | Add tardiness variables and constraints |
| Machine maintenance | Modify decoder availability windows | Add machine availability constraints |
| Job priorities | Weighted fitness penalties | Weighted objective terms |
| Multi-objective scheduling | Easy to combine in weighted fitness | Possible but model becomes larger |
| Soft constraints | Natural through penalty functions | Requires penalty variables |

GA is often more flexible for soft and complex constraints. MIP is stronger when constraints must be enforced exactly and the model remains computationally solvable.

### 5.6 Advantages and Limitations

| Criterion | GA | MIP | Better Method |
| --- | --- | --- | --- |
| Solution quality on FT06 | Optimal makespan found | Optimal makespan found | Equal |
| Optimality proof | No independent proof | Provides proof | MIP |
| Runtime control | Controlled by hyperparameters | Depends on solver search | GA |
| Scalability | Better for large cases | Weaker for large cases | GA |
| Constraint strictness | Decoder and repair logic needed | Enforced exactly | MIP |
| Flexibility | High | Medium | GA |
| Interpretability | Medium | High | MIP |
| Real-world adaptability | High | Medium to high | GA |

### 5.7 Recommended Use Cases for GA

GA is the more appropriate choice when:

- The instance is large and exact optimization is too slow.
- A near-optimal solution is acceptable.
- The schedule must be regenerated frequently.
- Soft constraints, penalties, or multiple objectives are important.
- Parallel evaluation is available.
- The production system includes complex real-world rules that are difficult to formulate exactly.

### 5.8 Recommended Use Cases for MIP

MIP is the more appropriate choice when:

- A provably optimal solution is required.
- The instance is small or moderate.
- Strict feasibility and auditability are important.
- Decision-makers need confidence that no better schedule exists.
- The mathematical formulation remains computationally manageable.

### 5.9 Practical Limitations

FT06 is useful for validating both methods because it has a known optimum and is small enough for exact solution. However, it is still a simplified benchmark. It does not include due dates, stochastic processing times, setup times, worker shifts, material availability, machine failures, or urgent job arrivals. Therefore, the comparison is valuable for methodological validation, but it does not fully represent industrial scheduling complexity.

The MIP result is more rigorous for FT06 because it proves optimality. The GA result is more relevant to scalability and flexibility, especially for larger or more dynamic settings. Neither method is universally better; the best choice depends on problem size, time availability, required proof, and business constraints.

### 5.10 Future Work and Methodological Improvements

The analysis could be strengthened through:

1. Test larger benchmarks such as FT10, FT20, LA01, and LA20.
2. Run the GA over many random seeds and report best, average, worst, standard deviation, and average runtime.
3. Add local search to create a hybrid GA.
4. Use a GA-MIP hybrid, where GA finds strong feasible solutions and MIP improves or validates them.
5. Improve the MIP formulation using tighter Big-M values, valid inequalities, or alternative formulations.
6. Parallelize GA fitness evaluation.
7. Add real-world constraints such as setup times, due dates, maintenance windows, and machine breakdowns.

## 6. Conclusions and Recommendations

This analysis solved the FT06 Job-Shop Scheduling Problem using both a Genetic Algorithm and a Mixed Integer Programming formulation. The objective was to minimize makespan while satisfying job precedence and machine-capacity constraints.

Both approaches achieved the known optimal makespan of 55 with a 0% optimality gap. The GA reached the optimum through evolutionary search using an operation-based chromosome, schedule decoder, tournament selection, JOX crossover, swap mutation, elitism, and early stopping. The MIP model reached the same objective value and proved optimality using explicit sequencing variables and machine non-overlap constraints.

Overall, MIP is the stronger method for small instances where proof of optimality is required, while GA is the more flexible and scalable method for larger or more complex scheduling environments. A hybrid GA-MIP approach would be a strong future direction because it could combine GA's search flexibility with MIP's exact validation and improvement capabilities.

## References

- Adams, J., Balas, E., and Zawack, D. (1988). The shifting bottleneck procedure for job shop scheduling.
- Cebi, F., Atac, B., and Sahingoz, O. K. (2020). Review of Job-Shop Scheduling Problem solution algorithms.
- Cheng, R., Gen, M., and Tsujimura, Y. (1996/1999). Genetic algorithm representation and hybrid strategies for job-shop scheduling.
- Garey, M. R., Johnson, D. S., and Sethi, R. (1976). The complexity of flowshop and jobshop scheduling.
- Goncalves, J. F., Mendes, J. J. M., and Resende, M. G. C. (2005). A hybrid genetic algorithm for the job shop scheduling problem.
- King, A. J. and Hildebrand, R. (2024). Comparative approaches for job-shop scheduling using integer programming and related methods.
- Ku, W. Y. and Beck, J. C. (2016). Mixed integer programming models for job-shop scheduling.
- PuLP documentation: https://coin-or.github.io/pulp/
- CBC documentation: https://coin-or.github.io/Cbc/intro.html
- FT06 benchmark reference: https://www.jobshoppuzzle.com/benchmarks.html

## Appendix: Reproducibility Artefacts

The analysis outputs are stored with the report so that the schedules and figures can be checked independently.

| Artefact | Description |
| --- | --- |
| `question3_generate_assets.py` | Script used to generate the GA/MIP schedules, metrics, and report figures. |
| `question3_report_assets/ga_convergence.png` | GA convergence chart showing best and average makespan across generations. |
| `question3_report_assets/ga_gantt.png` | Gantt chart for the best GA schedule. |
| `question3_report_assets/mip_gantt.png` | Gantt chart for the optimal MIP schedule. |
| `question3_report_assets/method_comparison.png` | Comparison chart for GA and MIP makespan/runtime results. |
| `question3_report_assets/ga_best_schedule.csv` | Decoded operation-level GA schedule. |
| `question3_report_assets/mip_optimal_schedule.csv` | Operation-level MIP schedule. |
| `question3_report_assets/summary.json` | Summary metrics for the GA and MIP runs. |
