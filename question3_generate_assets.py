from __future__ import annotations

import csv
import html
import itertools
import json
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# Fisher-Thompson FT06 benchmark (machines are stored using 1-based labels).
PROCESSING_TIMES = (
    (1, 3, 6, 7, 3, 6),
    (8, 5, 10, 10, 10, 4),
    (5, 4, 8, 9, 1, 7),
    (5, 5, 5, 3, 8, 9),
    (9, 3, 5, 4, 3, 1),
    (3, 3, 9, 10, 4, 1),
)

MACHINE_ROUTES = (
    (3, 1, 2, 4, 6, 5),
    (2, 3, 5, 6, 1, 4),
    (3, 4, 6, 1, 2, 5),
    (2, 1, 3, 4, 5, 6),
    (3, 2, 5, 6, 1, 4),
    (2, 4, 6, 1, 5, 3),
)

ROOT_DIR = Path(__file__).resolve().parent
ASSET_DIR = ROOT_DIR / "question3_report_assets"
N_JOBS = len(PROCESSING_TIMES)
N_MACHINES = len(PROCESSING_TIMES[0])
N_OPERATIONS = N_JOBS * N_MACHINES
KNOWN_OPTIMUM = 55
BIG_M = sum(sum(row) for row in PROCESSING_TIMES)

# The taught tuning material distinguishes the search space, validation metric,
# and final evaluation. A bounded random search is used here because a complete
# 3^4 grid would require 81 configurations before seed repetitions.
TUNING_SEARCH_SEED = 2025
TUNING_CANDIDATE_COUNT = 12
TUNING_SEEDS = (11, 29, 47)
EVALUATION_SEEDS = (101, 103, 107, 109, 113)
TUNING_SPACE = {
    "population_size": (50, 100, 150),
    "crossover_probability": (0.80, 0.90, 0.95),
    "mutation_probability": (0.10, 0.20, 0.30),
    "generations": (100, 300, 500),
}


@dataclass(frozen=True)
class GAConfig:
    population_size: int
    generations: int
    crossover_probability: float
    mutation_probability: float
    elite_size: int = 4
    tournament_size: int = 3


def operations() -> list[dict[str, int]]:
    return [
        {
            "Job": job + 1,
            "Operation": operation + 1,
            "Machine": MACHINE_ROUTES[job][operation],
            "Duration": PROCESSING_TIMES[job][operation],
        }
        for job in range(N_JOBS)
        for operation in range(N_MACHINES)
    ]


OPERATIONS = operations()


def validate_chromosome(chromosome: list[int]) -> None:
    if len(chromosome) != N_OPERATIONS:
        raise ValueError(f"A chromosome must contain {N_OPERATIONS} genes")
    counts = {job: chromosome.count(job) for job in range(N_JOBS)}
    if any(count != N_MACHINES for count in counts.values()):
        raise ValueError(f"Invalid job multiplicities: {counts}")


def decode_chromosome(chromosome: list[int]) -> tuple[int, list[dict[str, int]]]:
    """Decode a valid operation-based chromosome into a feasible schedule."""
    validate_chromosome(chromosome)
    job_next_operation = [0] * N_JOBS
    job_available = [0] * N_JOBS
    machine_available = [0] * N_MACHINES
    schedule: list[dict[str, int]] = []

    for job in chromosome:
        operation = job_next_operation[job]
        machine = MACHINE_ROUTES[job][operation] - 1
        duration = PROCESSING_TIMES[job][operation]
        start = max(job_available[job], machine_available[machine])
        end = start + duration

        job_next_operation[job] += 1
        job_available[job] = end
        machine_available[machine] = end
        schedule.append(
            {
                "Job": job + 1,
                "Operation": operation + 1,
                "Machine": machine + 1,
                "Start": start,
                "End": end,
                "Duration": duration,
            }
        )

    return max(job_available), schedule


def initial_chromosome(rng: random.Random) -> list[int]:
    chromosome = [job for job in range(N_JOBS) for _ in range(N_MACHINES)]
    rng.shuffle(chromosome)
    return chromosome


def tournament_selection(
    population: list[list[int]],
    makespans: list[int],
    size: int,
    rng: random.Random,
) -> list[int]:
    candidate_indices = rng.sample(range(len(population)), size)
    winner = min(candidate_indices, key=lambda index: makespans[index])
    return population[winner][:]


def job_order_crossover(
    parent_1: list[int],
    parent_2: list[int],
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    """Job Order Crossover preserves the required count of every job."""
    subset_size = rng.randint(1, N_JOBS - 1)
    selected_jobs = set(rng.sample(range(N_JOBS), subset_size))

    def make_child(primary: list[int], secondary: list[int]) -> list[int]:
        child: list[int | None] = [None] * len(primary)
        for index, gene in enumerate(primary):
            if gene in selected_jobs:
                child[index] = gene
        filler = iter(gene for gene in secondary if gene not in selected_jobs)
        for index, gene in enumerate(child):
            if gene is None:
                child[index] = next(filler)
        result = [int(gene) for gene in child]
        validate_chromosome(result)
        return result

    return make_child(parent_1, parent_2), make_child(parent_2, parent_1)


def swap_mutation(chromosome: list[int], rng: random.Random) -> None:
    first, second = rng.sample(range(len(chromosome)), 2)
    chromosome[first], chromosome[second] = chromosome[second], chromosome[first]


def evaluate_population(
    population: list[list[int]],
    cache: dict[tuple[int, ...], int],
) -> list[int]:
    makespans: list[int] = []
    for chromosome in population:
        key = tuple(chromosome)
        if key not in cache:
            cache[key] = decode_chromosome(chromosome)[0]
        makespans.append(cache[key])
    return makespans


def run_ga(config: GAConfig, seed: int, keep_history: bool = True) -> dict[str, Any]:
    """Run one reproducible GA replication for a supplied configuration."""
    if config.elite_size >= config.population_size:
        raise ValueError("elite_size must be smaller than population_size")
    if config.tournament_size > config.population_size:
        raise ValueError("tournament_size must not exceed population_size")

    rng = random.Random(seed)
    started = time.perf_counter()
    cache: dict[tuple[int, ...], int] = {}
    population = [initial_chromosome(rng) for _ in range(config.population_size)]
    makespans = evaluate_population(population, cache)

    best_index = min(range(config.population_size), key=lambda index: makespans[index])
    best_makespan = makespans[best_index]
    best_chromosome = population[best_index][:]
    first_optimum_generation = 0 if best_makespan <= KNOWN_OPTIMUM else None
    generations_completed = 0

    history: list[dict[str, float | int]] = []

    def record_history(generation: int) -> None:
        if keep_history:
            history.append(
                {
                    "Generation": generation,
                    "Best Makespan": best_makespan,
                    "Average Makespan": statistics.mean(makespans),
                    "Best Fitness": 1.0 / best_makespan,
                    "Average Fitness": statistics.mean(1.0 / value for value in makespans),
                }
            )

    record_history(0)

    for generation in range(1, config.generations + 1):
        ranked_indices = sorted(
            range(config.population_size), key=lambda index: makespans[index]
        )
        next_population = [
            population[index][:] for index in ranked_indices[: config.elite_size]
        ]

        while len(next_population) < config.population_size:
            parent_1 = tournament_selection(
                population, makespans, config.tournament_size, rng
            )
            parent_2 = tournament_selection(
                population, makespans, config.tournament_size, rng
            )
            if rng.random() < config.crossover_probability:
                child_1, child_2 = job_order_crossover(parent_1, parent_2, rng)
            else:
                child_1, child_2 = parent_1, parent_2

            if rng.random() < config.mutation_probability:
                swap_mutation(child_1, rng)
            if rng.random() < config.mutation_probability:
                swap_mutation(child_2, rng)

            next_population.append(child_1)
            if len(next_population) < config.population_size:
                next_population.append(child_2)

        population = next_population
        makespans = evaluate_population(population, cache)
        generation_best_index = min(
            range(config.population_size), key=lambda index: makespans[index]
        )
        generation_best = makespans[generation_best_index]

        if generation_best < best_makespan:
            best_makespan = generation_best
            best_chromosome = population[generation_best_index][:]
            if best_makespan <= KNOWN_OPTIMUM and first_optimum_generation is None:
                first_optimum_generation = generation

        generations_completed = generation
        record_history(generation)

    makespan, schedule = decode_chromosome(best_chromosome)
    validation = validate_schedule(schedule, reported_makespan=makespan)
    if not validation["valid"]:
        raise RuntimeError(f"GA produced an invalid schedule: {validation['issues']}")

    return {
        "seed": seed,
        "config": asdict(config),
        "best_makespan": makespan,
        "best_fitness": 1.0 / makespan,
        "known_optimum": KNOWN_OPTIMUM,
        "known_optimum_gap_percent": 100 * (makespan - KNOWN_OPTIMUM) / KNOWN_OPTIMUM,
        "runtime_seconds": time.perf_counter() - started,
        "generations_completed": generations_completed,
        "first_optimum_generation": first_optimum_generation,
        "chromosome_1_based": [gene + 1 for gene in best_chromosome],
        "history": history,
        "schedule": schedule,
        "schedule_validation": validation,
    }


def summarise_ga_runs(runs: list[dict[str, Any]]) -> dict[str, float | int]:
    makespans = [float(run["best_makespan"]) for run in runs]
    runtimes = [float(run["runtime_seconds"]) for run in runs]
    generations = [float(run["generations_completed"]) for run in runs]
    return {
        "runs": len(runs),
        "best_makespan": min(makespans),
        "mean_makespan": statistics.mean(makespans),
        "worst_makespan": max(makespans),
        "std_makespan": statistics.pstdev(makespans),
        "optimum_hit_rate_percent": 100
        * sum(value == KNOWN_OPTIMUM for value in makespans)
        / len(makespans),
        "mean_runtime_seconds": statistics.mean(runtimes),
        "mean_generations_completed": statistics.mean(generations),
    }


def tuning_candidates() -> list[GAConfig]:
    combinations = list(
        itertools.product(
            TUNING_SPACE["population_size"],
            TUNING_SPACE["crossover_probability"],
            TUNING_SPACE["mutation_probability"],
            TUNING_SPACE["generations"],
        )
    )
    sampled = random.Random(TUNING_SEARCH_SEED).sample(
        combinations, TUNING_CANDIDATE_COUNT
    )
    for position, values in enumerate(zip(*sampled)):
        expected = set(tuple(TUNING_SPACE.values())[position])
        if set(values) != expected:
            raise RuntimeError("The bounded tuning sample does not cover every tested value")
    return [
        GAConfig(
            population_size=population_size,
            generations=generations,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
        )
        for population_size, crossover_probability, mutation_probability, generations in sampled
    ]


def tune_ga() -> tuple[list[dict[str, Any]], GAConfig]:
    rows: list[dict[str, Any]] = []
    configs = tuning_candidates()
    for candidate_number, config in enumerate(configs, start=1):
        print(
            f"Tuning GA configuration {candidate_number}/{len(configs)}",
            file=sys.stderr,
        )
        runs = [run_ga(config, seed, keep_history=False) for seed in TUNING_SEEDS]
        summary = summarise_ga_runs(runs)
        rows.append(
            {
                "Candidate": candidate_number,
                "Population Size": config.population_size,
                "Crossover Probability": config.crossover_probability,
                "Mutation Probability": config.mutation_probability,
                "Max Generations": config.generations,
                "Mean Makespan": summary["mean_makespan"],
                "Best Makespan": summary["best_makespan"],
                "Worst Makespan": summary["worst_makespan"],
                "Std Makespan": summary["std_makespan"],
                "Optimum Hit Rate Percent": summary["optimum_hit_rate_percent"],
                "Mean Runtime Seconds": summary["mean_runtime_seconds"],
                "Mean Generations Completed": summary["mean_generations_completed"],
                "Selected": "No",
            }
        )

    selected_index = min(
        range(len(rows)),
        key=lambda index: (
            float(rows[index]["Mean Makespan"]),
            float(rows[index]["Worst Makespan"]),
            float(rows[index]["Std Makespan"]),
            configs[index].population_size * configs[index].generations,
            int(rows[index]["Candidate"]),
        ),
    )
    rows[selected_index]["Selected"] = "Yes"
    return rows, configs[selected_index]


def validate_schedule(
    schedule: list[dict[str, Any]],
    reported_makespan: float | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    issues: list[str] = []
    expected = {(job + 1, operation + 1) for job in range(N_JOBS) for operation in range(N_MACHINES)}
    actual = [(int(row["Job"]), int(row["Operation"])) for row in schedule]

    if len(actual) != N_OPERATIONS:
        issues.append(f"expected {N_OPERATIONS} operations, found {len(actual)}")
    if len(set(actual)) != len(actual):
        issues.append("duplicate operations found")
    missing = expected - set(actual)
    if missing:
        issues.append(f"missing operations: {sorted(missing)}")

    for row in schedule:
        job = int(row["Job"]) - 1
        operation = int(row["Operation"]) - 1
        if not (0 <= job < N_JOBS and 0 <= operation < N_MACHINES):
            continue
        expected_machine = MACHINE_ROUTES[job][operation]
        expected_duration = PROCESSING_TIMES[job][operation]
        start = float(row["Start"])
        end = float(row["End"])
        duration = float(row["Duration"])
        if int(row["Machine"]) != expected_machine:
            issues.append(f"J{job + 1}-O{operation + 1} has the wrong machine")
        if abs(duration - expected_duration) > tolerance:
            issues.append(f"J{job + 1}-O{operation + 1} has the wrong duration")
        if start < -tolerance:
            issues.append(f"J{job + 1}-O{operation + 1} starts before zero")
        if abs(end - start - duration) > tolerance:
            issues.append(f"J{job + 1}-O{operation + 1} has inconsistent times")

    for job in range(1, N_JOBS + 1):
        rows = sorted(
            (row for row in schedule if int(row["Job"]) == job),
            key=lambda row: int(row["Operation"]),
        )
        for previous, current in zip(rows, rows[1:]):
            if float(current["Start"]) + tolerance < float(previous["End"]):
                issues.append(f"job precedence violated for J{job}")

    for machine in range(1, N_MACHINES + 1):
        rows = sorted(
            (row for row in schedule if int(row["Machine"]) == machine),
            key=lambda row: (float(row["Start"]), float(row["End"])),
        )
        for previous, current in zip(rows, rows[1:]):
            if float(current["Start"]) + tolerance < float(previous["End"]):
                issues.append(f"machine overlap found on M{machine}")

    calculated_makespan = max((float(row["End"]) for row in schedule), default=0.0)
    if reported_makespan is not None and abs(calculated_makespan - reported_makespan) > tolerance:
        issues.append("reported makespan differs from the schedule")

    return {
        "valid": not issues,
        "operations_checked": len(schedule),
        "calculated_makespan": calculated_makespan,
        "issues": issues,
    }


def normalise_time(value: float) -> int | float:
    rounded = round(value)
    return int(rounded) if abs(value - rounded) <= 1e-6 else round(value, 6)


def solve_mip(time_limit_seconds: int = 120) -> dict[str, Any]:
    try:
        import pulp
    except ImportError as exc:
        return {
            "available": False,
            "status": "PuLP unavailable",
            "message": str(exc),
            "schedule": [],
        }

    started = time.perf_counter()
    problem = pulp.LpProblem("FT06_Job_Shop_Scheduling", pulp.LpMinimize)
    op_keys = [(row["Job"] - 1, row["Operation"] - 1) for row in OPERATIONS]

    start_vars = pulp.LpVariable.dicts("S", op_keys, lowBound=0, cat="Continuous")
    completion_vars = pulp.LpVariable.dicts("C", op_keys, lowBound=0, cat="Continuous")
    cmax = pulp.LpVariable("Cmax", lowBound=0, cat="Continuous")

    machine_to_ops: dict[int, list[tuple[int, int]]] = {}
    for row in OPERATIONS:
        machine_to_ops.setdefault(row["Machine"] - 1, []).append(
            (row["Job"] - 1, row["Operation"] - 1)
        )

    sequencing_vars: dict[tuple[tuple[int, int], tuple[int, int]], Any] = {}
    for machine_ops in machine_to_ops.values():
        for index, op_a in enumerate(machine_ops):
            for op_b in machine_ops[index + 1 :]:
                sequencing_vars[(op_a, op_b)] = pulp.LpVariable(
                    f"x_J{op_a[0] + 1}_O{op_a[1] + 1}_before_"
                    f"J{op_b[0] + 1}_O{op_b[1] + 1}",
                    cat="Binary",
                )

    problem += cmax, "Minimize_makespan"

    # Equality constraints define every completion time.
    for job, operation in op_keys:
        problem += (
            completion_vars[(job, operation)]
            == start_vars[(job, operation)] + PROCESSING_TIMES[job][operation],
            f"completion_J{job + 1}_O{operation + 1}",
        )

    # Inequality constraints enforce technological precedence.
    for job in range(N_JOBS):
        for operation in range(N_MACHINES - 1):
            problem += (
                completion_vars[(job, operation)]
                <= start_vars[(job, operation + 1)],
                f"precedence_J{job + 1}_O{operation + 1}",
            )

    for job, operation in op_keys:
        problem += (
            completion_vars[(job, operation)] <= cmax,
            f"makespan_J{job + 1}_O{operation + 1}",
        )

    # Binary disjunctions prevent operations on the same machine from overlapping.
    for (op_a, op_b), x_var in sequencing_vars.items():
        problem += completion_vars[op_a] <= start_vars[op_b] + BIG_M * (1 - x_var)
        problem += completion_vars[op_b] <= start_vars[op_a] + BIG_M * x_var

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds)
    status_code = problem.solve(solver)
    runtime_seconds = time.perf_counter() - started
    status = pulp.LpStatus.get(status_code, str(status_code))
    objective_value = pulp.value(cmax)

    schedule: list[dict[str, Any]] = []
    if objective_value is not None and status not in {"Infeasible", "Unbounded"}:
        for job, operation in op_keys:
            start_value = pulp.value(start_vars[(job, operation)])
            completion_value = pulp.value(completion_vars[(job, operation)])
            if start_value is None or completion_value is None:
                continue
            schedule.append(
                {
                    "Job": job + 1,
                    "Operation": operation + 1,
                    "Machine": MACHINE_ROUTES[job][operation],
                    "Start": float(start_value),
                    "End": float(completion_value),
                    "Duration": PROCESSING_TIMES[job][operation],
                }
            )

    schedule.sort(
        key=lambda row: (row["Machine"], row["Start"], row["Job"], row["Operation"])
    )
    validation = validate_schedule(schedule, objective_value) if schedule else {
        "valid": False,
        "operations_checked": 0,
        "calculated_makespan": None,
        "issues": ["no schedule returned"],
    }

    return {
        "available": True,
        "solver": "CBC via PuLP",
        "pulp_version": pulp.__version__,
        "status": status,
        "status_code": status_code,
        "proven_optimal": status == "Optimal",
        "objective_value": objective_value,
        "known_optimum": KNOWN_OPTIMUM,
        "known_optimum_gap_percent": (
            100 * (float(objective_value) - KNOWN_OPTIMUM) / KNOWN_OPTIMUM
            if objective_value is not None
            else None
        ),
        "runtime_seconds": runtime_seconds,
        "schedule": schedule,
        "schedule_validation": validation,
        "variables": len(problem.variables()),
        "start_variables": len(start_vars),
        "completion_variables": len(completion_vars),
        "binary_variables": len(sequencing_vars),
        "constraints": len(problem.constraints),
        "equality_constraints": N_OPERATIONS,
        "precedence_constraints": N_JOBS * (N_MACHINES - 1),
        "machine_non_overlap_constraints": 2 * len(sequencing_vars),
        "makespan_constraints": N_OPERATIONS,
        "big_m": BIG_M,
        "message": "",
    }


def svg_line_chart(
    history: list[dict[str, float | int]], seed: int, path: Path
) -> None:
    width, height = 960, 540
    left, right, top, bottom = 82, 28, 44, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    generations = [int(row["Generation"]) for row in history]
    best = [float(row["Best Fitness"]) for row in history]
    average = [float(row["Average Fitness"]) for row in history]
    x_min, x_max = min(generations), max(generations)
    optimum_fitness = 1.0 / KNOWN_OPTIMUM
    y_min = min(min(average), min(best)) * 0.98
    y_max = max(optimum_fitness, max(average), max(best)) * 1.02

    def x_scale(value: float) -> float:
        return left if x_max == x_min else left + (value - x_min) / (x_max - x_min) * plot_w

    def y_scale(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    def points(values: list[float]) -> str:
        return " ".join(
            f"{x_scale(generation):.2f},{y_scale(value):.2f}"
            for generation, value in zip(generations, values)
        )

    grid: list[str] = []
    x_step = 20 if x_max <= 120 else 50
    for tick in range(0, x_max + 1, x_step):
        x = x_scale(tick)
        grid.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height - bottom}" stroke="#e5e7eb"/>')
        grid.append(f'<text x="{x:.2f}" y="{height - bottom + 24}" font-size="12" text-anchor="middle" fill="#374151">{tick}</text>')
    for tick_number in range(6):
        tick = y_min + tick_number * (y_max - y_min) / 5
        y = y_scale(tick)
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        grid.append(f'<text x="{left - 12}" y="{y + 4:.2f}" font-size="12" text-anchor="end" fill="#374151">{tick:.4f}</text>')

    optimum_y = y_scale(optimum_fitness)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="26" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#111827">GA fitness convergence on FT06 (evaluation seed {seed})</text>
  <g font-family="Arial, sans-serif">
    {''.join(grid)}
    <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111827" stroke-width="1.5"/>
    <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827" stroke-width="1.5"/>
    <line x1="{left}" y1="{optimum_y:.2f}" x2="{width - right}" y2="{optimum_y:.2f}" stroke="#6b7280" stroke-width="1.5" stroke-dasharray="7 6"/>
    <polyline points="{points(average)}" fill="none" stroke="#d97706" stroke-width="2.2"/>
    <polyline points="{points(best)}" fill="none" stroke="#0f766e" stroke-width="3"/>
    <text x="{width / 2}" y="{height - 20}" font-size="14" text-anchor="middle" fill="#111827">Generation</text>
    <text x="22" y="{height / 2}" font-size="14" text-anchor="middle" fill="#111827" transform="rotate(-90 22 {height / 2})">Fitness = 1 / makespan (higher is better)</text>
    <rect x="{width - 272}" y="50" width="220" height="82" fill="#ffffff" stroke="#d1d5db"/>
    <line x1="{width - 252}" y1="76" x2="{width - 208}" y2="76" stroke="#0f766e" stroke-width="3"/>
    <text x="{width - 196}" y="80" font-size="13" fill="#111827">Best fitness</text>
    <line x1="{width - 252}" y1="100" x2="{width - 208}" y2="100" stroke="#d97706" stroke-width="2.2"/>
    <text x="{width - 196}" y="104" font-size="13" fill="#111827">Average fitness</text>
    <line x1="{width - 252}" y1="124" x2="{width - 208}" y2="124" stroke="#6b7280" stroke-width="1.5" stroke-dasharray="7 6"/>
    <text x="{width - 196}" y="128" font-size="13" fill="#111827">Optimum fitness 1/55</text>
  </g>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def svg_gantt(schedule: list[dict[str, Any]], title: str, path: Path) -> None:
    width = 1060
    row_h, top, left, right, bottom = 46, 58, 92, 34, 56
    height = top + row_h * N_MACHINES + bottom
    max_time = max(float(item["End"]) for item in schedule)
    plot_w = width - left - right
    colors = {
        1: "#2563eb", 2: "#dc2626", 3: "#16a34a",
        4: "#9333ea", 5: "#d97706", 6: "#0891b2",
    }

    def x_scale(value: float) -> float:
        return left + value / max_time * plot_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="28" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        '<g font-family="Arial, sans-serif">',
    ]
    for machine in range(1, N_MACHINES + 1):
        y = top + (machine - 1) * row_h
        parts.append(f'<text x="{left - 20}" y="{y + 28}" font-size="14" text-anchor="end" fill="#111827">M{machine}</text>')
        parts.append(f'<line x1="{left}" y1="{y + row_h - 5}" x2="{width - right}" y2="{y + row_h - 5}" stroke="#e5e7eb"/>')

    for tick in range(0, int(max_time) + 1, 5):
        x = x_scale(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{top - 10}" x2="{x:.2f}" y2="{height - bottom + 8}" stroke="#f3f4f6"/>')
        parts.append(f'<text x="{x:.2f}" y="{height - 24}" font-size="12" text-anchor="middle" fill="#374151">{tick}</text>')

    for item in schedule:
        y = top + (int(item["Machine"]) - 1) * row_h + 8
        x = x_scale(float(item["Start"]))
        bar_width = max(3, x_scale(float(item["End"])) - x)
        label = f'J{item["Job"]}-O{item["Operation"]}'
        parts.append(f'<rect x="{x:.2f}" y="{y}" width="{bar_width:.2f}" height="28" rx="4" fill="{colors[int(item["Job"])]}" opacity="0.88"/>')
        if bar_width >= 42:
            parts.append(f'<text x="{x + bar_width / 2:.2f}" y="{y + 18}" font-size="11" text-anchor="middle" fill="#ffffff">{label}</text>')

    parts.append(f'<text x="{width / 2}" y="{height - 5}" font-size="14" text-anchor="middle" fill="#111827">Time units</text>')
    parts.append("</g></svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ASSET_DIR.mkdir(exist_ok=True)

    tuning_rows, selected_config = tune_ga()
    write_csv(ASSET_DIR / "ga_tuning_results.csv", tuning_rows)

    evaluation_runs = [
        run_ga(selected_config, seed, keep_history=True) for seed in EVALUATION_SEEDS
    ]
    evaluation_summary = summarise_ga_runs(evaluation_runs)
    evaluation_rows = [
        {
            "Seed": run["seed"],
            "Best Makespan": run["best_makespan"],
            "Known Optimum Gap Percent": run["known_optimum_gap_percent"],
            "Runtime Seconds": run["runtime_seconds"],
            "Generations Completed": run["generations_completed"],
            "First Optimum Generation": (
                run["first_optimum_generation"]
                if run["first_optimum_generation"] is not None
                else ""
            ),
            "Schedule Valid": run["schedule_validation"]["valid"],
        }
        for run in evaluation_runs
    ]
    write_csv(ASSET_DIR / "ga_evaluation_results.csv", evaluation_rows)
    best_ga_run = min(
        evaluation_runs, key=lambda run: (run["best_makespan"], run["seed"])
    )
    write_csv(ASSET_DIR / "ga_best_schedule.csv", best_ga_run["schedule"])
    svg_line_chart(
        best_ga_run["history"],
        int(best_ga_run["seed"]),
        ASSET_DIR / "ga_convergence.svg",
    )
    svg_gantt(
        best_ga_run["schedule"],
        f"Best evaluated GA schedule, makespan = {best_ga_run['best_makespan']}",
        ASSET_DIR / "ga_gantt.svg",
    )

    mip_result = solve_mip()
    if not mip_result.get("available"):
        raise RuntimeError(
            "PuLP is required. Run with .venv-q3/bin/python or install PuLP."
        )
    if not mip_result["proven_optimal"]:
        raise RuntimeError(f"CBC did not prove optimality: {mip_result['status']}")
    if not mip_result["schedule_validation"]["valid"]:
        raise RuntimeError(
            f"MIP schedule validation failed: {mip_result['schedule_validation']['issues']}"
        )

    write_csv(ASSET_DIR / "mip_optimal_schedule.csv", mip_result["schedule"])
    svg_gantt(
        mip_result["schedule"],
        f"MIP schedule ({mip_result['status']}), makespan = {normalise_time(float(mip_result['objective_value']))}",
        ASSET_DIR / "mip_gantt.svg",
    )

    summary = {
        "problem": {
            "instance": "FT06",
            "jobs": N_JOBS,
            "machines": N_MACHINES,
            "operations": N_OPERATIONS,
            "known_optimum": KNOWN_OPTIMUM,
            "big_m": BIG_M,
        },
        "tuning": {
            "method": "bounded random search",
            "search_seed": TUNING_SEARCH_SEED,
            "candidate_count": TUNING_CANDIDATE_COUNT,
            "tuning_seeds": list(TUNING_SEEDS),
            "selection_order": [
                "mean makespan",
                "worst makespan",
                "makespan standard deviation",
                "population_size * max_generations",
            ],
            "selected_config": asdict(selected_config),
        },
        "ga_evaluation": {
            "seeds": list(EVALUATION_SEEDS),
            "summary": evaluation_summary,
            "best_run": {
                "seed": best_ga_run["seed"],
                "best_makespan": best_ga_run["best_makespan"],
                "known_optimum_gap_percent": best_ga_run[
                    "known_optimum_gap_percent"
                ],
                "generations_completed": best_ga_run["generations_completed"],
                "first_optimum_generation": best_ga_run[
                    "first_optimum_generation"
                ],
                "chromosome_1_based": best_ga_run["chromosome_1_based"],
                "schedule_validation": best_ga_run["schedule_validation"],
            },
        },
        "mip": {key: value for key, value in mip_result.items() if key != "schedule"},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    (ASSET_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
