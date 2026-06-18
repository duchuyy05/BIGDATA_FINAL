from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path


DEFAULT_SAMPLE_PATIENTS = [
    "p000001",
    "p000003",
    "p000007",
    "p000009",
    "p000011",
    "p000018",
    "p000028",
    "p000042",
]


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def int_from_env(*names: str, default: int) -> int:
    for name in names:
        raw = os.getenv(name)
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
    return default


def split_env_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def patient_id(value: str) -> str:
    return Path(value.strip()).stem


def resolve_data_root() -> Path:
    configured = os.getenv("DATA_ROOT")
    if configured:
        return Path(configured)
    if Path("/data").exists():
        return Path("/data")
    return Path("data")


def resolve_data_path(raw_value: str | None, data_root: Path, default_name: str) -> Path:
    if not raw_value:
        return data_root / default_name

    normalized = raw_value.replace("\\", "/")
    if normalized == "/data":
        return data_root
    if normalized.startswith("/data/") and not Path("/data").exists():
        return data_root / normalized.removeprefix("/data/")
    return Path(raw_value)


def ensure_inside(child: Path, parent: Path) -> None:
    child_resolved = child.resolve()
    parent_resolved = parent.resolve()
    try:
        child_resolved.relative_to(parent_resolved)
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside data root: {child_resolved}") from exc


def reset_directory(target: Path, data_root: Path) -> None:
    ensure_inside(target, data_root)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def source_dirs(data_root: Path) -> list[Path]:
    configured = split_env_list(os.getenv("DATA_SOURCE_DIRS"))
    if configured:
        return [Path(item) for item in configured]
    return [data_root / "Data-Set-A", data_root / "Data-Set-B"]


def build_file_index(data_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for directory in source_dirs(data_root):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.psv")):
            index.setdefault(path.stem, path)
    return index


def select_sample_files(index: dict[str, Path]) -> list[Path]:
    configured = [patient_id(item) for item in split_env_list(os.getenv("DEMO_PATIENT_IDS"))]
    sample_ids = configured or DEFAULT_SAMPLE_PATIENTS
    missing = [pid for pid in sample_ids if pid not in index]
    if missing:
        print(f"Warning: missing sample patient files: {', '.join(missing)}")
    return [index[pid] for pid in sample_ids if pid in index]


def select_x_patient_files(data_root: Path, x_patients: int) -> list[Path]:
    selected: list[Path] = []
    for directory in source_dirs(data_root):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.psv")):
            selected.append(path)
            if len(selected) >= x_patients:
                return selected
    return selected


def patient_summary(path: Path) -> dict[str, object]:
    rows = 0
    sepsis_hours = 0
    lab_observations = 0
    important_columns = {
        "HR",
        "O2Sat",
        "Temp",
        "SBP",
        "MAP",
        "Resp",
        "Lactate",
        "WBC",
        "Platelets",
        "Creatinine",
        "BUN",
        "Glucose",
    }

    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        for row in reader:
            rows += 1
            if str(row.get("SepsisLabel", "")).strip() == "1":
                sepsis_hours += 1
            for column in important_columns:
                value = str(row.get(column, "")).strip()
                if value and value != "NaN":
                    lab_observations += 1

    return {
        "patient_id": path.stem,
        "source_file": str(path).replace("\\", "/"),
        "rows": rows,
        "has_sepsis": sepsis_hours > 0,
        "sepsis_hours": sepsis_hours,
        "observed_values": lab_observations,
    }


def copy_selection(files: list[Path], target_dir: Path, data_root: Path) -> list[dict[str, object]]:
    reset_directory(target_dir, data_root)
    summaries = []
    for path in files:
        target = target_dir / path.name
        shutil.copy2(path, target)
        summaries.append(patient_summary(target))
    return summaries


def write_manifest(target_dir: Path, mode: str, patients: list[dict[str, object]], x_patients: int) -> None:
    manifest = {
        "mode": mode,
        "use_x_patients": mode == "x_patients",
        "x_patients": x_patients,
        "delay_seconds": int_from_env("DELAY", default=2),
        "count": len(patients),
        "patients": patients,
    }
    (target_dir / "patients.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    load_dotenv()
    data_root = resolve_data_root()
    active_dir = resolve_data_path(os.getenv("ACTIVE_DATA_DIR"), data_root, "active")
    sample_dir = resolve_data_path(os.getenv("SAMPLE_DATA_DIR"), data_root, "sample")
    use_x_patients = truthy(os.getenv("USE_X_PATIENTS"))
    x_patients = int_from_env("X_PATIENTS", "MAX_PATIENTS", default=100)

    if use_x_patients:
        selected_files = select_x_patient_files(data_root, x_patients)
        mode = "x_patients"
    else:
        index = build_file_index(data_root)
        selected_files = select_sample_files(index)
        mode = "sample"
        sample_patients = copy_selection(selected_files, sample_dir, data_root)
        write_manifest(sample_dir, mode, sample_patients, x_patients)

    active_patients = copy_selection(selected_files, active_dir, data_root)
    write_manifest(active_dir, mode, active_patients, x_patients)

    ids = ", ".join(patient["patient_id"] for patient in active_patients)
    print(f"Prepared {len(active_patients)} patient files in {active_dir} using mode={mode}")
    print(f"Active patient IDs: {ids}")


if __name__ == "__main__":
    main()
