#!/usr/bin/env python3

# What this script does:
#   Locks the fixed younger preprocessing subset used in this project so the
#   workflow reproduces the same younger branch each time.
# How to run it:
#   Run from the repo root with:
#   python code/primary/04_select_younger_preprocessing_subset.py
# Main outputs:
#   data/processed/screening/ds005752_mri_participants_age_20_25_preprocessing_primary.tsv
#   data/processed/screening/ds005752_mri_participants_age_20_25_preprocessing_backup.tsv
#   data/processed/screening/ds005752_mri_participants_age_20_25_preprocessing_plan.tsv

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Candidate:
    # This just stores the subject-level details I need when I balance the younger subset.
    row: dict[str, str]
    subject_id: str
    age: int
    sex: str
    release_group: str
    already_downloaded: int
    already_preprocessed: int


@dataclass(frozen=True)
class Plan:
    # A plan is one possible subset together with the counts I care about matching.
    candidates: tuple[Candidate, ...]
    female_count: int
    release_1_count: int
    release_2_count: int
    downloaded_count: int
    preprocessed_count: int

    @property
    def subject_ids(self) -> tuple[str, ...]:
        return tuple(candidate.subject_id for candidate in self.candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Choose the younger participants to preprocess using a fixed, "
            "dissertation-defensible age, sex, and release balancing rule."
        )
    )
    parser.add_argument(
        "--input",
        default="data/processed/screening/ds005752_mri_participants_age_20_25_remote_anat_forward.tsv",
        help="Remote-screened younger candidate TSV.",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/ds005752",
        help="Raw BIDS dataset directory used to detect already downloaded subjects.",
    )
    parser.add_argument(
        "--derivatives-dir",
        default="data/derivatives/fmriprep",
        help="fMRIPrep derivatives directory used to detect already preprocessed subjects.",
    )
    parser.add_argument(
        "--primary-size",
        type=int,
        default=25,
        help="Number of younger subjects to place in the primary preprocessing set.",
    )
    parser.add_argument(
        "--backup-size",
        type=int,
        default=10,
        help="Number of backup younger subjects to set aside if replacements are needed.",
    )
    parser.add_argument(
        "--female-target",
        type=int,
        default=17,
        help="Target number of female subjects in the primary younger set.",
    )
    parser.add_argument(
        "--male-target",
        type=int,
        default=8,
        help="Target number of male subjects in the primary younger set.",
    )
    parser.add_argument(
        "--release-1-target",
        type=int,
        default=17,
        help="Target number of release_1 subjects in the primary younger set.",
    )
    parser.add_argument(
        "--release-2-target",
        type=int,
        default=8,
        help="Target number of release_2 subjects in the primary younger set.",
    )
    parser.add_argument(
        "--age-targets",
        default="20:1,21:4,22:7,23:6,24:4,25:3",
        help=(
            "Comma-separated age targets for the primary younger set in the form "
            "age:count,age:count."
        ),
    )
    parser.add_argument(
        "--primary-output",
        default="data/processed/screening/ds005752_mri_participants_age_20_25_preprocessing_primary.tsv",
        help="Output TSV for the primary younger preprocessing set.",
    )
    parser.add_argument(
        "--backup-output",
        default="data/processed/screening/ds005752_mri_participants_age_20_25_preprocessing_backup.tsv",
        help="Output TSV for the backup younger preprocessing set.",
    )
    parser.add_argument(
        "--plan-output",
        default="data/processed/screening/ds005752_mri_participants_age_20_25_preprocessing_plan.tsv",
        help="Output TSV ranking all younger candidates as primary, backup, or not selected.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_age_targets(text: str) -> dict[int, int]:
    age_targets: dict[int, int] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        age_text, count_text = item.split(":", maxsplit=1)
        age_targets[int(age_text)] = int(count_text)
    if not age_targets:
        raise ValueError("At least one age target is required.")
    return age_targets


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return reader.fieldnames, list(reader)


def release_group(row: dict[str, str]) -> str:
    release_1 = row.get("release_1", "").strip()
    release_2 = row.get("release_2", "").strip()
    if release_1 == "1" and release_2 == "0":
        return "release_1"
    if release_1 == "0" and release_2 == "1":
        return "release_2"
    return "other"


def build_candidates(
    rows: list[dict[str, str]],
    raw_dir: Path,
    derivatives_dir: Path,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for row in rows:
        subject_id = row["participant_id"].strip()
        age = int(row["age"])
        sex = row["sex"].strip().lower()
        if sex not in {"female", "male"}:
            raise ValueError(f"Unexpected sex value for {subject_id}: {row['sex']!r}")
        candidates.append(
            Candidate(
                row=row,
                subject_id=subject_id,
                age=age,
                sex=sex,
                release_group=release_group(row),
                already_downloaded=int((raw_dir / subject_id).exists()),
                already_preprocessed=int((derivatives_dir / f"{subject_id}.html").exists()),
            )
        )
    return sorted(candidates, key=lambda candidate: (candidate.age, candidate.subject_id))


def make_plan(candidates: tuple[Candidate, ...]) -> Plan:
    return Plan(
        candidates=candidates,
        female_count=sum(candidate.sex == "female" for candidate in candidates),
        release_1_count=sum(candidate.release_group == "release_1" for candidate in candidates),
        release_2_count=sum(candidate.release_group == "release_2" for candidate in candidates),
        downloaded_count=sum(candidate.already_downloaded for candidate in candidates),
        preprocessed_count=sum(candidate.already_preprocessed for candidate in candidates),
    )


def is_better_plan(left: Plan, right: Plan | None) -> bool:
    if right is None:
        return True
    left_key = (left.preprocessed_count, left.downloaded_count)
    right_key = (right.preprocessed_count, right.downloaded_count)
    if left_key != right_key:
        return left_key > right_key
    return left.subject_ids < right.subject_ids


def combine_plans(left: Plan, right: Plan) -> Plan:
    return Plan(
        candidates=left.candidates + right.candidates,
        female_count=left.female_count + right.female_count,
        release_1_count=left.release_1_count + right.release_1_count,
        release_2_count=left.release_2_count + right.release_2_count,
        downloaded_count=left.downloaded_count + right.downloaded_count,
        preprocessed_count=left.preprocessed_count + right.preprocessed_count,
    )


def validate_primary_targets(
    candidates: list[Candidate],
    age_targets: dict[int, int],
    primary_size: int,
    female_target: int,
    male_target: int,
    release_1_target: int,
    release_2_target: int,
) -> None:
    if sum(age_targets.values()) != primary_size:
        raise ValueError("Age targets must sum to the primary selection size.")
    if female_target + male_target != primary_size:
        raise ValueError("Female and male targets must sum to the primary selection size.")
    if release_1_target + release_2_target != primary_size:
        raise ValueError("Release targets must sum to the primary selection size.")

    available_by_age: dict[int, int] = {}
    for candidate in candidates:
        available_by_age[candidate.age] = available_by_age.get(candidate.age, 0) + 1
    for age, target in sorted(age_targets.items()):
        available = available_by_age.get(age, 0)
        if available < target:
            raise ValueError(
                f"Age {age} has only {available} candidates, below the target of {target}."
            )

    available_female = sum(candidate.sex == "female" for candidate in candidates)
    available_male = sum(candidate.sex == "male" for candidate in candidates)
    if available_female < female_target:
        raise ValueError(
            f"Only {available_female} female candidates are available, below the target of "
            f"{female_target}."
        )
    if available_male < male_target:
        raise ValueError(
            f"Only {available_male} male candidates are available, below the target of "
            f"{male_target}."
        )

    available_release_1 = sum(candidate.release_group == "release_1" for candidate in candidates)
    available_release_2 = sum(candidate.release_group == "release_2" for candidate in candidates)
    if available_release_1 < release_1_target:
        raise ValueError(
            f"Only {available_release_1} release_1 candidates are available, below the target "
            f"of {release_1_target}."
        )
    if available_release_2 < release_2_target:
        raise ValueError(
            f"Only {available_release_2} release_2 candidates are available, below the target "
            f"of {release_2_target}."
        )


def enumerate_age_choices(
    candidates: list[Candidate],
    age_targets: dict[int, int],
) -> dict[int, list[Plan]]:
    age_choices: dict[int, list[Plan]] = {}
    for age, target in sorted(age_targets.items()):
        age_candidates = [candidate for candidate in candidates if candidate.age == age]
        best_by_counts: dict[tuple[int, int, int], Plan] = {}
        for subset in combinations(age_candidates, target):
            plan = make_plan(tuple(subset))
            key = (plan.female_count, plan.release_1_count, plan.release_2_count)
            current = best_by_counts.get(key)
            if is_better_plan(plan, current):
                best_by_counts[key] = plan
        age_choices[age] = list(best_by_counts.values())
    return age_choices


def select_subset(
    candidates: list[Candidate],
    age_targets: dict[int, int],
    female_target: int,
    release_1_target: int,
    release_2_target: int,
) -> Plan | None:
    # This dynamic-programming step finds a subset that hits the requested balance as closely as possible.
    empty_plan = make_plan(tuple())
    dp: dict[tuple[int, int, int], Plan] = {(0, 0, 0): empty_plan}
    age_choices = enumerate_age_choices(candidates, age_targets)

    for age in sorted(age_targets):
        next_dp: dict[tuple[int, int, int], Plan] = {}
        for current_key, current_plan in dp.items():
            for age_plan in age_choices[age]:
                female_count = current_key[0] + age_plan.female_count
                release_1_count = current_key[1] + age_plan.release_1_count
                release_2_count = current_key[2] + age_plan.release_2_count
                if female_count > female_target:
                    continue
                if release_1_count > release_1_target:
                    continue
                if release_2_count > release_2_target:
                    continue
                combined = combine_plans(current_plan, age_plan)
                key = (female_count, release_1_count, release_2_count)
                current_best = next_dp.get(key)
                if is_better_plan(combined, current_best):
                    next_dp[key] = combined
        dp = next_dp

    return dp.get((female_target, release_1_target, release_2_target))


def scale_targets(targets: dict[int, int], scaled_total: int) -> dict[int, int]:
    base_total = sum(targets.values())
    if base_total == 0:
        return {key: 0 for key in targets}

    scaled = {key: (value * scaled_total) / base_total for key, value in targets.items()}
    floored = {key: math.floor(value) for key, value in scaled.items()}
    remaining = scaled_total - sum(floored.values())
    ranked_keys = sorted(
        targets,
        key=lambda key: (scaled[key] - floored[key], targets[key], key),
        reverse=True,
    )
    for key in ranked_keys[:remaining]:
        floored[key] += 1
    return floored


def sort_remaining_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.already_preprocessed,
            -candidate.already_downloaded,
            candidate.age,
            candidate.subject_id,
        ),
    )


def summarize_counts(candidates: tuple[Candidate, ...]) -> tuple[dict[int, int], dict[str, int], dict[str, int]]:
    age_counts: dict[int, int] = {}
    sex_counts = {"female": 0, "male": 0}
    release_counts = {"release_1": 0, "release_2": 0, "other": 0}

    for candidate in candidates:
        age_counts[candidate.age] = age_counts.get(candidate.age, 0) + 1
        sex_counts[candidate.sex] += 1
        release_counts[candidate.release_group] = release_counts.get(candidate.release_group, 0) + 1
    return age_counts, sex_counts, release_counts


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_output_rows(
    candidates: list[Candidate],
    primary_ids: set[str],
    backup_ids: set[str],
) -> list[dict[str, object]]:
    primary_candidates = sorted(
        [candidate for candidate in candidates if candidate.subject_id in primary_ids],
        key=lambda candidate: (candidate.age, candidate.subject_id),
    )
    backup_candidates = sorted(
        [candidate for candidate in candidates if candidate.subject_id in backup_ids],
        key=lambda candidate: (candidate.age, candidate.subject_id),
    )
    remaining_candidates = sort_remaining_candidates(
        [
            candidate
            for candidate in candidates
            if candidate.subject_id not in primary_ids and candidate.subject_id not in backup_ids
        ]
    )

    ordered_candidates = primary_candidates + backup_candidates + remaining_candidates
    rows: list[dict[str, object]] = []

    for rank, candidate in enumerate(ordered_candidates, start=1):
        row = dict(candidate.row)
        if candidate.subject_id in primary_ids:
            tier = "primary"
        elif candidate.subject_id in backup_ids:
            tier = "backup"
        else:
            tier = "not_selected"
        row.update(
            {
                "release_group": candidate.release_group,
                "already_downloaded": candidate.already_downloaded,
                "already_preprocessed": candidate.already_preprocessed,
                "selection_tier": tier,
                "selection_rank": rank,
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    raw_dir = resolve_project_path(args.raw_dir)
    derivatives_dir = resolve_project_path(args.derivatives_dir)
    primary_output = resolve_project_path(args.primary_output)
    backup_output = resolve_project_path(args.backup_output)
    plan_output = resolve_project_path(args.plan_output)

    age_targets = parse_age_targets(args.age_targets)
    fieldnames, rows = load_rows(input_path)
    candidates = build_candidates(rows, raw_dir, derivatives_dir)

    validate_primary_targets(
        candidates=candidates,
        age_targets=age_targets,
        primary_size=args.primary_size,
        female_target=args.female_target,
        male_target=args.male_target,
        release_1_target=args.release_1_target,
        release_2_target=args.release_2_target,
    )

    primary_plan = select_subset(
        candidates=candidates,
        age_targets=age_targets,
        female_target=args.female_target,
        release_1_target=args.release_1_target,
        release_2_target=args.release_2_target,
    )
    if primary_plan is None:
        raise ValueError("No younger subset satisfies the requested primary age, sex, and release targets.")

    primary_ids = {candidate.subject_id for candidate in primary_plan.candidates}
    remaining_candidates = [
        candidate for candidate in candidates if candidate.subject_id not in primary_ids
    ]

    backup_candidates: tuple[Candidate, ...] = tuple()
    if args.backup_size > 0 and remaining_candidates:
        scaled_age_targets = scale_targets(age_targets, args.backup_size)
        scaled_female_target = round((args.female_target / args.primary_size) * args.backup_size)
        scaled_release_1_target = round((args.release_1_target / args.primary_size) * args.backup_size)
        scaled_release_2_target = args.backup_size - scaled_release_1_target

        backup_plan = select_subset(
            candidates=remaining_candidates,
            age_targets=scaled_age_targets,
            female_target=scaled_female_target,
            release_1_target=scaled_release_1_target,
            release_2_target=scaled_release_2_target,
        )
        if backup_plan is not None:
            backup_candidates = backup_plan.candidates
        else:
            # If no exact balanced backup exists, I just keep the best remaining candidates as reserves.
            backup_candidates = tuple(sort_remaining_candidates(remaining_candidates)[: args.backup_size])

    backup_ids = {candidate.subject_id for candidate in backup_candidates}
    all_rows = build_output_rows(candidates, primary_ids=primary_ids, backup_ids=backup_ids)
    output_fieldnames = fieldnames + [
        "release_group",
        "already_downloaded",
        "already_preprocessed",
        "selection_tier",
        "selection_rank",
    ]

    primary_rows = [row for row in all_rows if row["selection_tier"] == "primary"]
    backup_rows = [row for row in all_rows if row["selection_tier"] == "backup"]

    write_rows(primary_output, output_fieldnames, primary_rows)
    write_rows(backup_output, output_fieldnames, backup_rows)
    write_rows(plan_output, output_fieldnames, all_rows)

    primary_age_counts, primary_sex_counts, primary_release_counts = summarize_counts(
        primary_plan.candidates
    )
    print(f"Wrote {len(primary_rows)} primary younger subjects to {primary_output}")
    print(f"Wrote {len(backup_rows)} backup younger subjects to {backup_output}")
    print(f"Wrote ranked younger selection plan to {plan_output}")
    print(f"Primary age counts: {primary_age_counts}")
    print(f"Primary sex counts: {primary_sex_counts}")
    print(f"Primary release counts: {primary_release_counts}")


if __name__ == "__main__":
    main()
