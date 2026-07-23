"""Fixed-budget FT06/FT10 scalability check for Question 3.

The larger FT10 data are copied from OR-Library's ``jobshop1.txt`` file:
https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt

This script deliberately reuses the GA representation, operators, decoder,
validator, and MIP formulation from ``question3_generate_assets.py``. It does
not retune the GA on FT10. Both instances receive the same GA configuration,
seeds, and 30-second single-thread CBC limit so that the comparison measures a
fixed computational budget rather than the best result obtainable per problem.
"""

from __future__ import annotations

import csv
import json
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import question3_generate_assets as q3


ROOT_DIR = Path(__file__).resolve().parent
ASSET_DIR = ROOT_DIR / "question3_report_assets"
GA_SEEDS = (101, 103, 107, 109, 113)
GA_CONFIG = q3.GAConfig(
    population_size=100,
    generations=100,
    crossover_probability=0.95,
    mutation_probability=0.20,
    elite_size=4,
    tournament_size=3,
)
CBC_TIME_LIMIT_SECONDS = 30


@dataclass(frozen=True)
class Instance:
    name: str
    processing_times: tuple[tuple[int, ...], ...]
    machine_routes: tuple[tuple[int, ...], ...]
    known_optimum: int


FT06 = Instance(
    name="FT06",
    processing_times=q3.PROCESSING_TIMES,
    machine_routes=q3.MACHINE_ROUTES,
    known_optimum=55,
)

# OR-Library stores machine identifiers from 0 to 9. They are converted to the
# 1-based convention used by the main Question 3 implementation.
FT10_PAIRS = (
    ((0, 29), (1, 78), (2, 9), (3, 36), (4, 49), (5, 11), (6, 62), (7, 56), (8, 44), (9, 21)),
    ((0, 43), (2, 90), (4, 75), (9, 11), (3, 69), (1, 28), (6, 46), (5, 46), (7, 72), (8, 30)),
    ((1, 91), (0, 85), (3, 39), (2, 74), (8, 90), (5, 10), (7, 12), (6, 89), (9, 45), (4, 33)),
    ((1, 81), (2, 95), (0, 71), (4, 99), (6, 9), (8, 52), (7, 85), (3, 98), (9, 22), (5, 43)),
    ((2, 14), (0, 6), (1, 22), (5, 61), (3, 26), (4, 69), (8, 21), (7, 49), (9, 72), (6, 53)),
    ((2, 84), (1, 2), (5, 52), (3, 95), (8, 48), (9, 72), (0, 47), (6, 65), (4, 6), (7, 25)),
    ((1, 46), (0, 37), (3, 61), (2, 13), (6, 32), (5, 21), (9, 32), (8, 89), (7, 30), (4, 55)),
    ((2, 31), (0, 86), (1, 46), (5, 74), (4, 32), (6, 88), (8, 19), (9, 48), (7, 36), (3, 79)),
    ((0, 76), (1, 69), (3, 76), (5, 51), (2, 85), (9, 11), (6, 40), (7, 89), (4, 26), (8, 74)),
    ((1, 85), (0, 13), (2, 61), (6, 7), (8, 64), (9, 76), (5, 47), (3, 52), (4, 90), (7, 45)),
)

FT10 = Instance(
    name="FT10",
    processing_times=tuple(
        tuple(duration for _, duration in row) for row in FT10_PAIRS
    ),
    machine_routes=tuple(
        tuple(machine + 1 for machine, _ in row) for row in FT10_PAIRS
    ),
    known_optimum=930,
)


def configure_instance(instance: Instance) -> None:
    """Point the shared implementation at one square benchmark instance."""
    jobs = len(instance.processing_times)
    machines = len(instance.processing_times[0])
    if any(len(row) != machines for row in instance.processing_times):
        raise ValueError(f"{instance.name} has inconsistent processing rows")
    if len(instance.machine_routes) != jobs or any(
        len(row) != machines for row in instance.machine_routes
    ):
        raise ValueError(f"{instance.name} has inconsistent route rows")
    expected_machines = set(range(1, machines + 1))
    if any(set(row) != expected_machines for row in instance.machine_routes):
        raise ValueError(f"Each {instance.name} job must visit every machine once")

    q3.PROCESSING_TIMES = instance.processing_times
    q3.MACHINE_ROUTES = instance.machine_routes
    q3.N_JOBS = jobs
    q3.N_MACHINES = machines
    q3.N_OPERATIONS = jobs * machines
    q3.KNOWN_OPTIMUM = instance.known_optimum
    q3.BIG_M = sum(sum(row) for row in instance.processing_times)
    q3.OPERATIONS = q3.operations()


def extract_number(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return float(match.group(1)) if match else None


def parse_cbc_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    result_match = re.search(r"^Result - (.+)$", text, flags=re.MULTILINE)
    termination = result_match.group(1).strip() if result_match else "Not recorded"
    objective = extract_number(r"^Objective value:\s+([-+0-9.eE]+)", text)
    lower_bound = extract_number(r"^Lower bound:\s+([-+0-9.eE]+)", text)
    raw_gap = extract_number(r"^Gap:\s+([-+0-9.eE]+)", text)
    enumerated_nodes = extract_number(r"^Enumerated nodes:\s+([-+0-9.eE]+)", text)
    wall_seconds = extract_number(
        r"^Time \(Wallclock seconds\):\s+([-+0-9.eE]+)", text
    )
    proven_optimal = "Optimal solution found" in termination
    if proven_optimal and objective is not None:
        lower_bound = objective
        raw_gap = 0.0
    return {
        "termination": termination,
        "proven_optimal_from_log": proven_optimal,
        "objective_value_from_log": objective,
        "best_bound": lower_bound,
        "solver_gap_percent": 100 * raw_gap if raw_gap is not None else None,
        "enumerated_nodes": int(enumerated_nodes) if enumerated_nodes is not None else None,
        "wallclock_seconds_from_log": wall_seconds,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_instance(instance: Instance) -> dict[str, Any]:
    configure_instance(instance)
    ga_runs = [q3.run_ga(GA_CONFIG, seed, keep_history=False) for seed in GA_SEEDS]
    ga_summary = q3.summarise_ga_runs(ga_runs)
    best_ga_run = min(ga_runs, key=lambda row: (row["best_makespan"], row["seed"]))
    write_csv(
        ASSET_DIR / f"{instance.name.lower()}_scalability_ga_best_schedule.csv",
        best_ga_run["schedule"],
    )

    log_path = ASSET_DIR / f"{instance.name.lower()}_scalability_cbc.log"
    mip = q3.solve_mip(
        CBC_TIME_LIMIT_SECONDS,
        log_path=log_path,
        problem_name=f"{instance.name}_Scalability_Check",
        threads=1,
    )
    log_evidence = parse_cbc_log(log_path)
    if mip["proven_optimal"] != log_evidence["proven_optimal_from_log"]:
        raise RuntimeError("Mapped and parsed CBC optimality evidence disagree")
    if mip["schedule"]:
        write_csv(
            ASSET_DIR / f"{instance.name.lower()}_scalability_mip_schedule.csv",
            mip["schedule"],
        )

    return {
        "instance": instance.name,
        "jobs": q3.N_JOBS,
        "machines": q3.N_MACHINES,
        "operations": q3.N_OPERATIONS,
        "known_optimum": instance.known_optimum,
        "big_m": q3.BIG_M,
        "ga": {
            "config": GA_CONFIG.__dict__,
            "seeds": list(GA_SEEDS),
            "summary": ga_summary,
            "best_schedule_validation": best_ga_run["schedule_validation"],
        },
        "mip": {
            "time_limit_seconds": CBC_TIME_LIMIT_SECONDS,
            "threads": 1,
            "pulp_status": mip["status"],
            "pulp_status_code": mip["status_code"],
            "proven_optimal": mip["proven_optimal"],
            "objective_value": mip["objective_value"],
            "runtime_seconds": mip["runtime_seconds"],
            "schedule_validation": mip["schedule_validation"],
            "variables": mip["variables"],
            "binary_variables": mip["binary_variables"],
            "constraints": mip["constraints"],
            "cbc_version": mip["cbc_version"],
            "pulp_version": mip["pulp_version"],
            **log_evidence,
        },
    }


def main() -> None:
    ASSET_DIR.mkdir(exist_ok=True)
    instances = [run_instance(instance) for instance in (FT06, FT10)]
    summary = {
        "design": {
            "purpose": "fixed-budget empirical scalability check",
            "instances": ["FT06", "FT10"],
            "source": "OR-Library jobshop1",
            "source_url": "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt",
            "ga_retuned_for_ft10": False,
        },
        "instances": instances,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pulp": instances[0]["mip"]["pulp_version"],
            "cbc": instances[0]["mip"]["cbc_version"],
        },
    }
    (ASSET_DIR / "scalability_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    comparison_rows: list[dict[str, Any]] = []
    for result in instances:
        ga = result["ga"]["summary"]
        mip = result["mip"]
        comparison_rows.append(
            {
                "Instance": result["instance"],
                "Jobs": result["jobs"],
                "Machines": result["machines"],
                "Operations": result["operations"],
                "Known Optimum": result["known_optimum"],
                "GA Best Makespan": ga["best_makespan"],
                "GA Mean Makespan": ga["mean_makespan"],
                "GA Worst Makespan": ga["worst_makespan"],
                "GA Mean Runtime Seconds": ga["mean_runtime_seconds"],
                "MIP Objective": mip["objective_value"],
                "MIP Best Bound": mip["best_bound"],
                "MIP Solver Gap Percent": mip["solver_gap_percent"],
                "MIP Termination": mip["termination"],
                "MIP Runtime Seconds": mip["runtime_seconds"],
                "MIP Binary Variables": mip["binary_variables"],
                "MIP Constraints": mip["constraints"],
                "GA Schedule Valid": result["ga"]["best_schedule_validation"]["valid"],
                "MIP Schedule Valid": mip["schedule_validation"]["valid"],
            }
        )
    write_csv(ASSET_DIR / "scalability_results.csv", comparison_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
