# -*- coding: utf-8 -*-
"""
Week 38 timetable optimization for VoN, University of Iceland.

This module is the cleaned "final model" built from the A–I3 validation chain.

Validated design choices:
- Week 38 only.
- Exams excluded.
- Exact activities and durations preserved.
- Fixed rooms; joint room-events move together.
- No overlap between different scheduling units using the same room.
- First-year protection: S1 somewhere AND no S2/S3 anywhere -> locked to baseline.
- Existing good lunch breaks for year 2/3 cohorts are protected:
  at least one of slots {5,6,7} must remain free from mandatory (S) courses.
- No new Saturday teaching; existing Saturday teaching stays fixed.
- No new overlaps between distinct sessions of the same course.
  Existing baseline parallel overlaps remain allowed.
- Teacher data is intentionally NOT used as a hard constraint because validation
  showed it behaves like course-level teacher-team metadata rather than reliable
  activity-level responsibility data.
- Objective: minimize weighted student course clashes using "actual" values from
  Árekstur. A course-pair contributes its weight once if ANY activity from one
  course overlaps ANY activity from the other.
- Baseline is passed as a MIP start and independently revalidated after solve.

The script is intentionally self-contained so the Colab notebook can stay short.

Typical use from Colab:
    from timetable_final import run_week38_optimization
    result = run_week38_optimization(
        data_dir="/content/drive/MyDrive/timetable_project",
        time_limit=600,
        mip_gap=0.05,
    )

Files expected in data_dir:
    Stundatöflugögn - Bókanir.csv
    Stundatöflugögn - Kjörsvið.csv
    Stundatöflugögn - Árekstur.csv

The other project CSV files may remain in the folder but are not required by the
validated final formulation because fixed room usage is already contained in Bókanir.

No Gurobi credentials are stored in this file.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import math
import re
import time
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import gurobipy as gp
from gurobipy import GRB


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOOKINGS_FILE = "Stundatöflugögn - Bókanir.csv"
ELECTIVES_FILE = "Stundatöflugögn - Kjörsvið.csv"
CLASH_FILE = "Stundatöflugögn - Árekstur.csv"

TARGET_WEEK = 38
LUNCH_SLOTS = {5, 6, 7}

DAY_MAP = {
    "mán": 0,
    "man": 0,
    "mánudagur": 0,
    "manudagur": 0,
    "mon": 0,
    "þri": 1,
    "thri": 1,
    "þriðjudagur": 1,
    "thridjudagur": 1,
    "tue": 1,
    "mið": 2,
    "mid": 2,
    "miðvikudagur": 2,
    "midvikudagur": 2,
    "wed": 2,
    "fim": 3,
    "fimmtudagur": 3,
    "thu": 3,
    "fös": 4,
    "fos": 4,
    "föstudagur": 4,
    "fostudagur": 4,
    "fri": 4,
    "lau": 5,
    "laugardagur": 5,
    "sat": 5,
}
DAY_NAMES = {
    0: "Mánudagur",
    1: "Þriðjudagur",
    2: "Miðvikudagur",
    3: "Fimmtudagur",
    4: "Föstudagur",
    5: "Laugardagur",
}
SATURDAY = 5
N_DAYS = 6

# Validated distinction:
# activities may OCCUPY slots up to the data-derived MAX_OCCUPIED_SLOT (=12),
# but new teaching may only START in the real timetable start slots 0..10.
MAX_START_SLOT = 10


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _norm_name(text: str) -> str:
    text = _strip_accents(str(text)).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def _clean_scalar(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    return s


def _clean_id(value) -> Optional[str]:
    s = _clean_scalar(value)
    if s is None:
        return None
    # Excel-like numeric IDs often arrive as 1234.0
    if re.fullmatch(r"-?\d+\.0", s):
        s = s[:-2]
    return s.strip()


def _as_int(value, field_name: str) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError(f"Missing {field_name}")
    try:
        f = float(value)
    except Exception as exc:
        raise ValueError(f"Invalid {field_name}: {value!r}") from exc
    if abs(f - round(f)) > 1e-9:
        raise ValueError(f"{field_name} must be integer, got {value!r}")
    return int(round(f))


def _find_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    *,
    required: bool = True,
) -> Optional[str]:
    norm_to_actual = {_norm_name(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_name(cand)
        if key in norm_to_actual:
            return norm_to_actual[key]
    if required:
        raise KeyError(
            f"Could not find required column. Tried {list(candidates)}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def _read_csv_flexible(path: Path, header="infer") -> pd.DataFrame:
    attempts = [
        dict(encoding="utf-8-sig", sep=","),
        dict(encoding="utf-8", sep=","),
        dict(encoding="latin1", sep=","),
        dict(encoding="utf-8-sig", sep=";"),
        dict(encoding="latin1", sep=";"),
    ]
    last_exc = None
    for opts in attempts:
        try:
            df = pd.read_csv(path, header=header, **opts)
            if df.shape[1] > 1:
                return df
        except Exception as exc:
            last_exc = exc
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Could not read {path}")


def parse_slot(slot_value) -> Tuple[int, int]:
    s = _clean_scalar(slot_value)
    if s is None:
        raise ValueError("Missing slot")
    s = s.strip().lower()
    # Usually "mán-6"
    m = re.match(r"^\s*([^-–—\s]+)\s*[-–—]\s*(\d+)\s*$", s)
    if not m:
        # fallback "mán 6"
        m = re.match(r"^\s*(\S+)\s+(\d+)\s*$", s)
    if not m:
        raise ValueError(f"Could not parse slot {slot_value!r}")
    day_txt = m.group(1).lower()
    start = int(m.group(2))
    if day_txt not in DAY_MAP:
        # try accent-stripped day key
        alt = _strip_accents(day_txt).lower()
        if alt not in DAY_MAP:
            raise ValueError(f"Unknown day in slot {slot_value!r}")
        day_txt = alt
    return DAY_MAP[day_txt], start


def _week_cell_contains_target(value, target_week: int) -> bool:
    s = _clean_scalar(value)
    if s is None:
        return False
    s_low = s.lower().strip()

    if s_low in {"true", "yes", "já", "ja", "x"}:
        return True

    # Direct numeric 38 / 38.0
    try:
        if int(float(s_low)) == target_week and float(s_low).is_integer():
            return True
    except Exception:
        pass

    # Parse ranges/lists: "36-40", "36,37,38", "38;39", "38-42 odd" etc.
    nums = [int(x) for x in re.findall(r"\d+", s_low)]
    if not nums:
        return False

    # Explicit appearance
    if target_week in nums:
        return True

    # Any simple range whose endpoints enclose target
    range_matches = re.findall(r"(\d+)\s*[-–—]\s*(\d+)", s_low)
    for a, b in range_matches:
        a, b = int(a), int(b)
        if min(a, b) <= target_week <= max(a, b):
            return True

    return False


# ---------------------------------------------------------------------------
# A — bookings / validated week 38 activities
# ---------------------------------------------------------------------------

def load_week38_activities(bookings_path: Path, target_week: int = TARGET_WEEK):
    raw = _read_csv_flexible(bookings_path)

    activity_col = _find_column(
        raw, ["activity_code", "activity code", "activity", "activitycode", "bókunarnúmer", "bokunarnumer"]
    )
    course_col = _find_column(
        raw, ["course_id", "course id", "course", "course_short_id", "námskeið", "namskeid", "námskeiðsnúmer"]
    )
    course_name_col = _find_column(
        raw, ["course_name", "course name", "name", "heiti", "námskeiðsheiti", "namskeidsheiti"],
        required=False,
    )
    type_col = _find_column(
        raw, ["type", "activity_type", "activity type", "tegund"], required=False
    )
    slot_col = _find_column(raw, ["slot", "timeslot", "time slot", "tími", "timi"])
    duration_col = _find_column(
        raw, ["n_timeslot", "n timeslot", "duration", "timeslots", "fjöldi tíma", "fjoldi tima"]
    )
    room_col = _find_column(
        raw, ["room_id", "room id", "room", "rooms", "stofa", "stofur"], required=False
    )
    teacher_col = _find_column(
        raw, ["teachers", "teacher", "teacher_values", "kennari", "kennarar"], required=False
    )

    # Exam detection
    exam_col = _find_column(raw, ["exam", "is_exam", "próf", "prof"], required=False)

    # Week detection can be a generic week(s) column or a dedicated "38" column.
    week_col = _find_column(
        raw,
        ["week", "weeks", "vika", "vikur", "week_number", "week numbers", "kennsluvikur"],
        required=False,
    )
    dedicated_week_col = None
    for c in raw.columns:
        nc = _norm_name(c)
        if nc in {str(target_week), f"v{target_week}", f"vika{target_week}", f"week{target_week}"}:
            dedicated_week_col = c
            break

    df = raw.copy()

    # Exclude exams if explicit marker exists.
    if exam_col is not None:
        def is_exam(v):
            s = (_clean_scalar(v) or "").lower()
            return s in {"1", "true", "yes", "já", "ja", "x", "exam", "próf", "prof"}
        df = df[~df[exam_col].map(is_exam)].copy()
    elif type_col is not None:
        # Conservative fallback: only filter explicit exam-like types.
        def exam_type(v):
            s = (_clean_scalar(v) or "").lower()
            return "próf" in s or "prof" in _strip_accents(s).lower() or "exam" in s
        df = df[~df[type_col].map(exam_type)].copy()

    # Filter target week when a week indicator exists.
    if dedicated_week_col is not None:
        def dedicated_active(v):
            s = (_clean_scalar(v) or "").lower()
            if s in {"", "0", "false", "no", "nei", "nan", "none"}:
                return False
            return True
        df = df[df[dedicated_week_col].map(dedicated_active)].copy()
    elif week_col is not None:
        df = df[df[week_col].map(lambda v: _week_cell_contains_target(v, target_week))].copy()
    else:
        raise KeyError(
            "Could not locate a week indicator in Bókanir. "
            "The final model needs to identify rows active in week 38."
        )

    parsed_rows = []
    for ridx, row in df.iterrows():
        activity_code = _clean_id(row[activity_col])
        course_id = _clean_id(row[course_col])
        if activity_code is None or course_id is None:
            continue

        day, start = parse_slot(row[slot_col])
        duration = _as_int(row[duration_col], "n_timeslot")
        if duration <= 0:
            raise ValueError(f"Activity {activity_code}: duration must be positive")

        room = _clean_id(row[room_col]) if room_col is not None else None
        teacher = _clean_scalar(row[teacher_col]) if teacher_col is not None else None

        parsed_rows.append(
            {
                "source_index": ridx,
                "activity_code": activity_code,
                "course_id": course_id,
                "course_name": (_clean_scalar(row[course_name_col]) if course_name_col else "") or "",
                "type": (_clean_scalar(row[type_col]) if type_col else "") or "",
                "slot": _clean_scalar(row[slot_col]),
                "baseline_day": day,
                "baseline_start": start,
                "n_timeslot": duration,
                "room_id": room,
                "teacher_value": teacher,
            }
        )

    week_raw = pd.DataFrame(parsed_rows)
    if week_raw.empty:
        raise ValueError("No active week-38 activities were found.")

    # Consolidate duplicate activity_code rows. Duplicates are allowed when they
    # represent multiple rooms or exact duplicate source rows.
    clean_rows = []
    inconsistent = []

    # Only scheduling-critical fields must be identical across duplicate
    # activity rows. Text fields such as course_name/type are descriptive and
    # can contain harmless source-formatting differences. This matches the
    # validated A-step logic: duplicate rows may represent multi-room teaching
    # or repeated source rows, but they must refer to the same course and exact
    # baseline placement/duration.
    core_cols = [
        "course_id",
        "baseline_day",
        "baseline_start",
        "n_timeslot",
    ]

    for activity_code, grp in week_raw.groupby("activity_code", sort=False):
        consistency = {}
        for col in core_cols:
            vals = tuple(pd.unique(grp[col]))
            consistency[col] = vals

        if any(len(v) > 1 for v in consistency.values()):
            inconsistent.append((activity_code, consistency))
            continue

        first = grp.iloc[0]

        rooms = tuple(
            sorted(
                {
                    r for r in grp["room_id"].tolist()
                    if r is not None and str(r).strip()
                }
            )
        )
        teachers = tuple(
            sorted(
                {
                    t for t in grp["teacher_value"].tolist()
                    if t is not None and str(t).strip()
                }
            )
        )

        clean_rows.append(
            {
                "activity_code": activity_code,
                "course_id": first["course_id"],
                "course_name": first["course_name"],
                "type": first["type"],
                "slot": first["slot"],
                "baseline_day": int(first["baseline_day"]),
                "baseline_start": int(first["baseline_start"]),
                "n_timeslot": int(first["n_timeslot"]),
                "room_ids": rooms,
                "teacher_values": teachers,
                "raw_rows": len(grp),
                "core_data_consistent": True,
            }
        )

    if inconsistent:
        raise ValueError(
            "Inconsistent duplicate activity codes found in Bókanir. "
            f"First examples: {inconsistent[:5]}"
        )

    activities = pd.DataFrame(clean_rows).reset_index(drop=True)
    activities.index.name = "activity_idx"

    max_occupied_slot = int(
        max(
            activities["baseline_start"] + activities["n_timeslot"] - 1
        )
    )

    max_baseline_start = int(activities["baseline_start"].max())
    if max_baseline_start > MAX_START_SLOT:
        bad = activities.loc[
            activities["baseline_start"] > MAX_START_SLOT,
            ["activity_code", "course_id", "slot", "baseline_start", "n_timeslot"],
        ]
        raise AssertionError(
            f"Baseline contains start slots above {MAX_START_SLOT}. "
            f"Examples: {bad.head(10).to_dict('records')}"
        )

    return raw, week_raw, activities, max_occupied_slot


# ---------------------------------------------------------------------------
# E/H/F — Kjörsvið classifications
# ---------------------------------------------------------------------------

def load_course_classifications(kjorsvid_path: Path):
    raw = _read_csv_flexible(kjorsvid_path, header=None)
    if raw.shape[0] < 3 or raw.shape[1] < 3:
        raise ValueError("Kjörsvið file has unexpected shape.")

    header_row = raw.iloc[1].tolist()
    headers = []
    seen = defaultdict(int)
    for j, h in enumerate(header_row):
        s = _clean_scalar(h) or f"col_{j}"
        seen[s] += 1
        if seen[s] > 1:
            s = f"{s}__{seen[s]}"
        headers.append(s)

    data = raw.iloc[2:].copy()
    data.columns = headers

    course_col = headers[0]
    name_col = headers[1]
    programme_cols = headers[2:]

    course_details = {}

    for _, row in data.iterrows():
        course_id = _clean_id(row[course_col])
        if course_id is None:
            continue

        entries = []
        for prog in programme_cols:
            cell = _clean_scalar(row[prog])
            if cell is None:
                continue

            # A cell may contain multiple labels, e.g. "V1,V2,V3".
            for cls, yr in re.findall(r"\b([SBV])\s*([123])\b", cell.upper()):
                entries.append(
                    {
                        "programme": prog,
                        "class": cls,
                        "year": int(yr),
                        "raw": cell,
                    }
                )

        course_details[course_id] = {
            "course_name": _clean_scalar(row[name_col]) or "",
            "entries": entries,
        }

    return raw, data, course_details


# ---------------------------------------------------------------------------
# B — Árekstur weights and baseline clash metric
# ---------------------------------------------------------------------------

def load_clash_weights(clash_path: Path, week38_courses: Set[str]):
    raw = _read_csv_flexible(clash_path, header=None)

    if raw.shape[0] < 3 or raw.shape[1] <= 4:
        raise ValueError("Árekstur file has unexpected shape.")

    # Validated B-structure:
    # row 0 = section/category labels
    # row 1 = actual column headers
    #          col 0: course_short_id
    #          col 1: course_name
    #          col 2: expected_students
    #          col 3: actual_students
    #          col 4+ = course IDs
    # row 2+ = course rows
    #
    # Clash cells are "expected,actual"; the SECOND value is the weight.
    col_courses = {}
    for j in range(4, raw.shape[1]):
        c = _clean_id(raw.iloc[1, j])
        if c:
            col_courses[j] = c

    if not col_courses:
        raise ValueError("Could not read course headers from Árekstur row 1.")

    row_course_col = 0

    header0 = (_clean_scalar(raw.iloc[1, row_course_col]) or "").strip().lower()
    if "course" not in header0:
        raise ValueError(
            f"Unexpected Árekstur structure: row 1 col 0 is {raw.iloc[1, 0]!r}"
        )

    pair_weights = {}

    for i in range(2, raw.shape[0]):
        row_course = _clean_id(raw.iloc[i, row_course_col])
        if not row_course:
            continue

        for j, col_course in col_courses.items():
            if row_course == col_course:
                continue

            cell = _clean_scalar(raw.iloc[i, j])
            if cell is None:
                continue

            parts = [p.strip() for p in cell.split(",")]
            if len(parts) < 2:
                continue

            try:
                actual = float(parts[1])
            except Exception:
                continue

            if actual <= 0:
                continue

            pair = tuple(sorted((row_course, col_course)))

            # Matrix is symmetric and can contain duplicate course headers.
            # Validated B logic deduplicates a course-pair and keeps max actual.
            previous = pair_weights.get(pair)
            if previous is None or actual > previous:
                pair_weights[pair] = actual

    # Keep only pairs where both courses are active in week 38.
    relevant = {
        pair: weight
        for pair, weight in pair_weights.items()
        if pair[0] in week38_courses and pair[1] in week38_courses
    }

    relevant = {
        pair: (int(w) if abs(w - round(w)) < 1e-9 else float(w))
        for pair, w in relevant.items()
    }

    return raw, relevant


# ---------------------------------------------------------------------------
# Scheduling units from validated joint room-events
# ---------------------------------------------------------------------------

def build_activity_data(activities: pd.DataFrame):
    activity_data = {}
    for idx, row in activities.iterrows():
        activity_data[idx] = {
            "activity_code": str(row["activity_code"]),
            "course_id": str(row["course_id"]),
            "course_name": str(row["course_name"]),
            "type": str(row["type"]),
            "baseline_day": int(row["baseline_day"]),
            "baseline_start": int(row["baseline_start"]),
            "duration": int(row["n_timeslot"]),
            "rooms": tuple(row["room_ids"]),
            "teachers": tuple(row["teacher_values"]),
        }
    return activity_data


def build_joint_groups(activity_data: Dict[int, dict]):
    """
    Validated D logic:
    Activities belong to one joint room-event component when they:
      - share at least one room, AND
      - have the same baseline day/start.
    Connected components define joint scheduling units.
    """
    parent = {idx: idx for idx in activity_data}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    room_start_map = defaultdict(list)
    for idx, a in activity_data.items():
        for room in a["rooms"]:
            room_start_map[(room, a["baseline_day"], a["baseline_start"])].append(idx)

    for ids in room_start_map.values():
        if len(ids) >= 2:
            first = ids[0]
            for other in ids[1:]:
                union(first, other)

    comps = defaultdict(list)
    for idx in activity_data:
        comps[find(idx)].append(idx)

    joint_groups = [
        sorted(members)
        for members in comps.values()
        if len(members) >= 2
    ]
    return joint_groups


def build_units(activity_data: Dict[int, dict], joint_groups: List[List[int]]):
    grouped = set()
    units = {}
    activity_to_unit = {}

    u = 0
    for members in joint_groups:
        members = tuple(sorted(members))
        grouped.update(members)

        baseline_positions = {
            (activity_data[idx]["baseline_day"], activity_data[idx]["baseline_start"])
            for idx in members
        }
        if len(baseline_positions) != 1:
            raise ValueError(f"Joint group {members} has inconsistent baseline starts.")
        bd, bs = next(iter(baseline_positions))

        room_duration = {}
        for idx in members:
            a = activity_data[idx]
            for room in a["rooms"]:
                room_duration[room] = max(room_duration.get(room, 0), a["duration"])

        max_duration = max(activity_data[idx]["duration"] for idx in members)

        units[u] = {
            "kind": "joint",
            "members": members,
            "baseline_day": bd,
            "baseline_start": bs,
            "room_duration": room_duration,
            "max_duration": max_duration,
        }
        for idx in members:
            activity_to_unit[idx] = u
        u += 1

    for idx in sorted(activity_data):
        if idx in grouped:
            continue
        a = activity_data[idx]
        room_duration = {room: a["duration"] for room in a["rooms"]}
        units[u] = {
            "kind": "singleton",
            "members": (idx,),
            "baseline_day": a["baseline_day"],
            "baseline_start": a["baseline_start"],
            "room_duration": room_duration,
            "max_duration": a["duration"],
        }
        activity_to_unit[idx] = u
        u += 1

    return units, activity_to_unit


# ---------------------------------------------------------------------------
# Baseline helpers / independent metric
# ---------------------------------------------------------------------------

def _interval_overlap(d1, s1, dur1, d2, s2, dur2):
    if d1 != d2:
        return False
    return s1 < s2 + dur2 and s2 < s1 + dur1


def calculate_course_clashes(
    activity_data: Dict[int, dict],
    activity_to_unit: Dict[int, int],
    positions: Dict[int, Tuple[int, int]],
    clash_weights: Dict[Tuple[str, str], float],
):
    course_acts = defaultdict(list)
    for idx, a in activity_data.items():
        u = activity_to_unit[idx]
        d, s = positions[u]
        course_acts[a["course_id"]].append((d, s, a["duration"], idx))

    clash_pairs = set()
    for pair in clash_weights:
        c1, c2 = pair
        found = False
        for d1, s1, dur1, _ in course_acts.get(c1, []):
            for d2, s2, dur2, _ in course_acts.get(c2, []):
                if _interval_overlap(d1, s1, dur1, d2, s2, dur2):
                    found = True
                    break
            if found:
                break
        if found:
            clash_pairs.add(pair)

    score = sum(clash_weights[pair] for pair in clash_pairs)
    return score, clash_pairs


# ---------------------------------------------------------------------------
# Cohorts / lunch / first-year protection
# ---------------------------------------------------------------------------

def build_cohort_mandatory_courses(course_details, week38_courses: Set[str]):
    cohort_courses = defaultdict(set)
    for course_id, info in course_details.items():
        if course_id not in week38_courses:
            continue
        for entry in info["entries"]:
            if entry["class"] == "S" and entry["year"] in {1, 2, 3}:
                cohort_courses[(entry["programme"], entry["year"])].add(course_id)
    return cohort_courses


def build_course_unit_duration(activity_data, activity_to_unit):
    course_unit_duration = defaultdict(dict)
    for idx, a in activity_data.items():
        c = a["course_id"]
        u = activity_to_unit[idx]
        course_unit_duration[c][u] = max(
            course_unit_duration[c].get(u, 0),
            a["duration"],
        )
    return course_unit_duration


def build_cohort_unit_duration(
    cohort_courses,
    course_unit_duration,
):
    out = defaultdict(dict)
    for cohort, courses in cohort_courses.items():
        for c in courses:
            for u, dur in course_unit_duration.get(c, {}).items():
                out[cohort][u] = max(out[cohort].get(u, 0), dur)
    return out


def baseline_lunch_table(cohort_unit_duration, units):
    rows = []
    for (programme, year), u_durs in cohort_unit_duration.items():
        for day in range(5):  # weekdays only
            occupied = set()
            for u, dur in u_durs.items():
                if units[u]["baseline_day"] != day:
                    continue
                s = units[u]["baseline_start"]
                for t in LUNCH_SLOTS:
                    if s <= t < s + dur:
                        occupied.add(t)
            free = sorted(LUNCH_SLOTS - occupied)
            rows.append(
                {
                    "programme": programme,
                    "year": int(year),
                    "day": day,
                    "lunch_ok": len(free) >= 1,
                    "free_lunch_slots": tuple(free),
                }
            )
    return pd.DataFrame(rows)


def protected_year1_courses(course_details, week38_courses: Set[str]):
    s_years = {}
    for c in week38_courses:
        years = {
            int(e["year"])
            for e in course_details.get(c, {}).get("entries", [])
            if e["class"] == "S" and e["year"] in {1, 2, 3}
        }
        s_years[c] = years

    protected = {
        c for c, years in s_years.items()
        if 1 in years and 2 not in years and 3 not in years
    }
    return protected, s_years


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    model: gp.Model
    baseline_score: float
    optimized_score: float
    baseline_clash_pairs: Set[Tuple[str, str]]
    optimized_clash_pairs: Set[Tuple[str, str]]
    improvement: float
    improvement_pct: float
    positions: Dict[int, Tuple[int, int]]
    optimized_schedule: pd.DataFrame
    moved_units: pd.DataFrame
    optimized_clashes: pd.DataFrame
    summary: dict


def run_week38_optimization(
    data_dir: str | Path,
    *,
    time_limit: int = 600,
    mip_gap: float = 0.05,
    output_flag: int = 1,
    expected_baseline_score: Optional[float] = 3822,
    expected_activity_count: Optional[int] = 504,
    expected_unit_count: Optional[int] = 448,
    write_outputs: bool = True,
    output_prefix: str = "week38_optimized",
) -> OptimizationResult:
    """
    Build, solve and independently validate the final week-38 timetable model.

    Returns OptimizationResult and optionally writes CSV outputs to data_dir.
    """
    t_build = time.time()
    data_dir = Path(data_dir)

    bookings_path = data_dir / BOOKINGS_FILE
    kjorsvid_path = data_dir / ELECTIVES_FILE
    clash_path = data_dir / CLASH_FILE

    for p in (bookings_path, kjorsvid_path, clash_path):
        if not p.exists():
            raise FileNotFoundError(p)

    # A
    raw_bookings, week_raw, activities, MAX_OCCUPIED_SLOT = load_week38_activities(
        bookings_path, TARGET_WEEK
    )
    activity_data = build_activity_data(activities)
    week38_courses = {a["course_id"] for a in activity_data.values()}

    # E
    _, _, course_details = load_course_classifications(kjorsvid_path)

    # B
    _, clash_weights = load_clash_weights(clash_path, week38_courses)

    # D
    joint_groups = build_joint_groups(activity_data)
    units, activity_to_unit = build_units(activity_data, joint_groups)

    if expected_activity_count is not None and len(activity_data) != expected_activity_count:
        raise AssertionError(
            f"Expected {expected_activity_count} week-38 activities, got {len(activity_data)}"
        )
    if expected_unit_count is not None and len(units) != expected_unit_count:
        raise AssertionError(
            f"Expected {expected_unit_count} scheduling units, got {len(units)}"
        )

    baseline_positions = {
        u: (unit["baseline_day"], unit["baseline_start"])
        for u, unit in units.items()
    }

    baseline_score, baseline_clash_pairs = calculate_course_clashes(
        activity_data, activity_to_unit, baseline_positions, clash_weights
    )

    if expected_baseline_score is not None and abs(baseline_score - expected_baseline_score) > 1e-9:
        raise AssertionError(
            f"Baseline mismatch: expected {expected_baseline_score}, got {baseline_score}. "
            "Stop here rather than optimize against a changed metric."
        )

    # F / H
    cohort_courses = build_cohort_mandatory_courses(course_details, week38_courses)
    course_unit_duration = build_course_unit_duration(activity_data, activity_to_unit)
    cohort_unit_duration = build_cohort_unit_duration(cohort_courses, course_unit_duration)
    lunch_df = baseline_lunch_table(cohort_unit_duration, units)

    protected_lunch_rows = lunch_df[
        lunch_df["year"].isin([2, 3]) & lunch_df["lunch_ok"]
    ].copy()

    protected_courses, s_years = protected_year1_courses(course_details, week38_courses)
    protected_activity_idxs = {
        idx for idx, a in activity_data.items()
        if a["course_id"] in protected_courses
    }
    protected_units = {
        activity_to_unit[idx] for idx in protected_activity_idxs
    }

    # -------------------------------------------------------
    # Build model
    # -------------------------------------------------------
    model = gp.Model("VoN_week38_final_optimization")

    # C: exactly one placement per scheduling unit
    y = {}
    for u, unit in units.items():
        latest_start = min(
            MAX_START_SLOT,
            MAX_OCCUPIED_SLOT - unit["max_duration"] + 1,
        )
        for d in range(N_DAYS):
            for s in range(latest_start + 1):
                y[(u, d, s)] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"y_{u}_{d}_{s}",
                )

    for u in units:
        model.addConstr(
            gp.quicksum(var for (uu, d, s), var in y.items() if uu == u) == 1,
            name=f"assign_{u}",
        )

    # D: room occupancy by scheduling unit
    room_to_units = defaultdict(dict)
    for u, unit in units.items():
        for room, dur in unit["room_duration"].items():
            room_to_units[room][u] = dur

    for room, u_durs in room_to_units.items():
        for d in range(N_DAYS):
            for t in range(MAX_OCCUPIED_SLOT + 1):
                terms = []
                for u, room_dur in u_durs.items():
                    latest_start = min(
                        MAX_START_SLOT,
                        MAX_OCCUPIED_SLOT - units[u]["max_duration"] + 1,
                    )
                    for s in range(latest_start + 1):
                        if s <= t < s + room_dur:
                            terms.append(y[(u, d, s)])
                if terms:
                    model.addConstr(
                        gp.quicksum(terms) <= 1,
                        name=f"room_{room}_{d}_{t}",
                    )

    # F: protect existing good year-2/3 lunch breaks.
    lunch_free = {}
    for ridx, row in protected_lunch_rows.iterrows():
        programme = row["programme"]
        year = int(row["year"])
        day = int(row["day"])
        cohort = (programme, year)

        for t in sorted(LUNCH_SLOTS):
            z = model.addVar(
                vtype=GRB.BINARY,
                name=f"lunchfree_{ridx}_{t}",
            )
            lunch_free[(programme, year, day, t)] = z

            occ_terms = []
            for u, dur in cohort_unit_duration[cohort].items():
                latest_start = min(
                    MAX_START_SLOT,
                    MAX_OCCUPIED_SLOT - units[u]["max_duration"] + 1,
                )
                for s in range(latest_start + 1):
                    if s <= t < s + dur:
                        occ_terms.append(y[(u, day, s)])

            if occ_terms:
                M = max(1, len(cohort_unit_duration[cohort]))
                model.addConstr(
                    gp.quicksum(occ_terms) <= M * (1 - z),
                    name=f"lunch_empty_{ridx}_{t}",
                )
            else:
                model.addConstr(z == 1, name=f"lunch_empty_{ridx}_{t}")

        model.addConstr(
            gp.quicksum(
                lunch_free[(programme, year, day, t)]
                for t in sorted(LUNCH_SLOTS)
            ) >= 1,
            name=f"lunch_protect_{ridx}",
        )

    # H: first-year lock
    for u in sorted(protected_units):
        bd, bs = baseline_positions[u]
        model.addConstr(y[(u, bd, bs)] == 1, name=f"year1_lock_{u}")

    # Final validated guardrail: no new Saturday teaching.
    for u, unit in units.items():
        bd, bs = baseline_positions[u]
        if bd == SATURDAY:
            model.addConstr(y[(u, bd, bs)] == 1, name=f"keep_saturday_{u}")
        else:
            sat_vars = [var for (uu, d, s), var in y.items() if uu == u and d == SATURDAY]
            if sat_vars:
                model.addConstr(
                    gp.quicksum(sat_vars) == 0,
                    name=f"no_new_saturday_{u}",
                )

    # Final validated guardrail: no NEW same-course overlaps.
    same_course_protected_pairs = []
    for course, u_durs in course_unit_duration.items():
        course_units = sorted(u_durs)
        for u1, u2 in combinations(course_units, 2):
            dur1 = u_durs[u1]
            dur2 = u_durs[u2]
            bd1, bs1 = baseline_positions[u1]
            bd2, bs2 = baseline_positions[u2]
            baseline_overlap = _interval_overlap(bd1, bs1, dur1, bd2, bs2, dur2)
            if not baseline_overlap:
                same_course_protected_pairs.append((course, u1, u2))

    for course, u1, u2 in same_course_protected_pairs:
        dur1 = course_unit_duration[course][u1]
        dur2 = course_unit_duration[course][u2]
        latest1 = min(
            MAX_START_SLOT,
            MAX_OCCUPIED_SLOT - units[u1]["max_duration"] + 1,
        )
        latest2 = min(
            MAX_START_SLOT,
            MAX_OCCUPIED_SLOT - units[u2]["max_duration"] + 1,
        )

        for d in range(N_DAYS):
            for t in range(MAX_OCCUPIED_SLOT + 1):
                terms = []
                for s in range(latest1 + 1):
                    if s <= t < s + dur1:
                        terms.append(y[(u1, d, s)])
                for s in range(latest2 + 1):
                    if s <= t < s + dur2:
                        terms.append(y[(u2, d, s)])
                if len(terms) > 1:
                    model.addConstr(
                        gp.quicksum(terms) <= 1,
                        name=f"samecourse_{course}_{u1}_{u2}_{d}_{t}",
                    )

    # Efficient I2/I3 objective formulation:
    # exact course occupancy OR -> clash binary.
    weighted_courses = set()
    for c1, c2 in clash_weights:
        weighted_courses.add(c1)
        weighted_courses.add(c2)

    course_occ = {}
    for course in sorted(weighted_courses):
        if course not in course_unit_duration:
            continue
        for d in range(N_DAYS):
            for t in range(MAX_OCCUPIED_SLOT + 1):
                possible_vars = []
                for u, dur in course_unit_duration[course].items():
                    latest_start = min(
                        MAX_START_SLOT,
                        MAX_OCCUPIED_SLOT - units[u]["max_duration"] + 1,
                    )
                    for s in range(latest_start + 1):
                        if s <= t < s + dur:
                            possible_vars.append(y[(u, d, s)])

                # Deduplicate variable references.
                # Variables may not yet have a queryable VarName until
                # model.update() has been called. Deduplicate by Python object
                # identity instead of reading a Gurobi attribute here.
                unique_vars = {id(v): v for v in possible_vars}
                possible_vars = list(unique_vars.values())
                if not possible_vars:
                    continue

                occ = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"occ_{course}_{d}_{t}",
                )
                course_occ[(course, d, t)] = occ
                model.addGenConstrOr(
                    occ,
                    possible_vars,
                    name=f"occ_or_{course}_{d}_{t}",
                )

    clash_var = {}
    for pair in clash_weights:
        c1, c2 = pair
        clash_var[pair] = model.addVar(
            vtype=GRB.BINARY,
            name=f"clash_{c1}_{c2}",
        )

    for pair, z in clash_var.items():
        c1, c2 = pair
        for d in range(N_DAYS):
            for t in range(MAX_OCCUPIED_SLOT + 1):
                k1 = (c1, d, t)
                k2 = (c2, d, t)
                if k1 in course_occ and k2 in course_occ:
                    model.addConstr(
                        z >= course_occ[k1] + course_occ[k2] - 1,
                        name=f"clashlink_{c1}_{c2}_{d}_{t}",
                    )

    model.setObjective(
        gp.quicksum(clash_weights[pair] * z for pair, z in clash_var.items()),
        GRB.MINIMIZE,
    )

    # -------------------------------------------------------
    # Baseline MIP start
    # -------------------------------------------------------
    for (u, d, s), var in y.items():
        var.Start = 1.0 if (d, s) == baseline_positions[u] else 0.0

    for (course, d, t), occ in course_occ.items():
        occupied = False
        for u, dur in course_unit_duration[course].items():
            bd, bs = baseline_positions[u]
            if bd == d and bs <= t < bs + dur:
                occupied = True
                break
        occ.Start = 1.0 if occupied else 0.0

    for pair, z in clash_var.items():
        z.Start = 1.0 if pair in baseline_clash_pairs else 0.0

    for _, row in protected_lunch_rows.iterrows():
        programme = row["programme"]
        year = int(row["year"])
        day = int(row["day"])
        baseline_free = set(row["free_lunch_slots"])
        for t in LUNCH_SLOTS:
            lunch_free[(programme, year, day, t)].Start = 1.0 if t in baseline_free else 0.0

    model.update()

    build_seconds = time.time() - t_build

    print("=" * 78)
    print("FINAL WEEK-38 TIMETABLE MODEL")
    print("=" * 78)
    print(f"Activities                  : {len(activity_data)}")
    print(f"Courses                     : {len(week38_courses)}")
    print(f"Joint room-event groups     : {len(joint_groups)}")
    print(f"Scheduling units            : {len(units)}")
    print(f"Protected S1-only courses   : {len(protected_courses)}")
    print(f"Protected scheduling units  : {len(protected_units)}")
    print(f"Protected lunch cohort-days : {len(protected_lunch_rows)}")
    print(f"Relevant weighted pairs     : {len(clash_weights)}")
    print(f"Baseline clash pairs        : {len(baseline_clash_pairs)}")
    print(f"Baseline clash score        : {baseline_score}")
    print(f"Allowed start slots         : 0..{MAX_START_SLOT}")
    print(f"Max occupied slot           : {MAX_OCCUPIED_SLOT}")
    print(f"Variables                   : {model.NumVars}")
    print(f"Linear constraints          : {model.NumConstrs}")
    print(f"General constraints         : {model.NumGenConstrs}")
    print(f"Build time                  : {build_seconds:.2f}s")
    print("=" * 78)

    # -------------------------------------------------------
    # Solve
    # -------------------------------------------------------
    model.Params.OutputFlag = output_flag
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap

    solve_t0 = time.time()
    model.optimize()
    solve_seconds = time.time() - solve_t0

    if model.SolCount <= 0:
        raise RuntimeError("Gurobi found no feasible solution.")

    optimized_score_gurobi = float(model.ObjVal)
    if optimized_score_gurobi > baseline_score + 1e-9:
        raise AssertionError(
            f"Optimizer returned worse score {optimized_score_gurobi} > baseline {baseline_score}"
        )

    positions = {}
    for u in units:
        chosen = [
            (d, s)
            for (uu, d, s), var in y.items()
            if uu == u and var.X > 0.5
        ]
        if len(chosen) != 1:
            raise AssertionError(f"Unit {u} has {len(chosen)} chosen positions.")
        positions[u] = chosen[0]

    # -------------------------------------------------------
    # Independent I3-style validation
    # -------------------------------------------------------
    independent_score, independent_pairs = calculate_course_clashes(
        activity_data, activity_to_unit, positions, clash_weights
    )
    gurobi_pairs = {pair for pair, z in clash_var.items() if z.X > 0.5}

    if abs(independent_score - optimized_score_gurobi) > 1e-9:
        raise AssertionError(
            f"Independent score {independent_score} != Gurobi {optimized_score_gurobi}"
        )
    if independent_pairs != gurobi_pairs:
        raise AssertionError("Independent clash-pair set differs from Gurobi clash-pair set.")

    # Reconstruct activity schedule
    optimized_rows = []
    for idx, a in activity_data.items():
        u = activity_to_unit[idx]
        d, s = positions[u]
        optimized_rows.append(
            {
                "activity_idx": idx,
                "activity_code": a["activity_code"],
                "course_id": a["course_id"],
                "course_name": a["course_name"],
                "type": a["type"],
                "unit": u,
                "day": d,
                "day_name": DAY_NAMES[d],
                "start": s,
                "duration": a["duration"],
                "end_exclusive": s + a["duration"],
                "rooms": "|".join(a["rooms"]),
                "baseline_day": units[u]["baseline_day"],
                "baseline_start": units[u]["baseline_start"],
                "moved": positions[u] != baseline_positions[u],
            }
        )
    optimized_schedule = pd.DataFrame(optimized_rows)

    # Rooms
    room_usage = defaultdict(list)
    for row in optimized_rows:
        rooms = [r for r in str(row["rooms"]).split("|") if r]
        for room in rooms:
            room_usage[room].append(row)

    room_conflicts = []
    for room, rows in room_usage.items():
        for a, b in combinations(rows, 2):
            if a["unit"] == b["unit"]:
                continue
            if _interval_overlap(
                a["day"], a["start"], a["duration"],
                b["day"], b["start"], b["duration"],
            ):
                room_conflicts.append((room, a["activity_code"], b["activity_code"]))
    if room_conflicts:
        raise AssertionError(f"Unexpected room conflicts: {room_conflicts[:10]}")

    # Joint events
    for u, unit in units.items():
        pos = {
            (row["day"], row["start"])
            for row in optimized_rows
            if row["unit"] == u
        }
        if len(pos) != 1:
            raise AssertionError(f"Joint/scheduling unit {u} split across starts.")

    # Protected S1 units
    protected_moved = [
        u for u in protected_units
        if positions[u] != baseline_positions[u]
    ]
    if protected_moved:
        raise AssertionError(f"Protected first-year units moved: {protected_moved[:10]}")

    # Lunch
    lunch_failures = []
    for _, row in protected_lunch_rows.iterrows():
        cohort = (row["programme"], int(row["year"]))
        day = int(row["day"])
        occupied = set()
        for u, dur in cohort_unit_duration[cohort].items():
            d, s = positions[u]
            if d != day:
                continue
            for t in LUNCH_SLOTS:
                if s <= t < s + dur:
                    occupied.add(t)
        if not (LUNCH_SLOTS - occupied):
            lunch_failures.append((cohort, day))
    if lunch_failures:
        raise AssertionError(f"Protected lunch failures: {lunch_failures[:10]}")

    # Same-course guardrail
    new_same_course = []
    for course, u1, u2 in same_course_protected_pairs:
        d1, s1 = positions[u1]
        d2, s2 = positions[u2]
        dur1 = course_unit_duration[course][u1]
        dur2 = course_unit_duration[course][u2]
        if _interval_overlap(d1, s1, dur1, d2, s2, dur2):
            new_same_course.append((course, u1, u2))
    if new_same_course:
        raise AssertionError(f"New same-course overlaps: {new_same_course[:10]}")

    # Start-slot validity
    invalid_start_units = [
        u for u, (d, s) in positions.items()
        if s > MAX_START_SLOT
    ]
    if invalid_start_units:
        raise AssertionError(
            f"Optimized units start after slot {MAX_START_SLOT}: "
            f"{invalid_start_units[:10]}"
        )

    # Saturday
    new_sat = []
    moved_sat = []
    for u in units:
        bd, bs = baseline_positions[u]
        d, s = positions[u]
        if bd != SATURDAY and d == SATURDAY:
            new_sat.append(u)
        if bd == SATURDAY and (d, s) != (bd, bs):
            moved_sat.append(u)
    if new_sat or moved_sat:
        raise AssertionError(
            f"Saturday validation failed. New={new_sat[:10]}, moved={moved_sat[:10]}"
        )

    # Moved units
    moved_rows = []
    for u, unit in units.items():
        if positions[u] == baseline_positions[u]:
            continue
        moved_rows.append(
            {
                "unit": u,
                "kind": unit["kind"],
                "baseline_day": baseline_positions[u][0],
                "baseline_start": baseline_positions[u][1],
                "optimized_day": positions[u][0],
                "optimized_start": positions[u][1],
                "courses": "|".join(
                    sorted({activity_data[idx]["course_id"] for idx in unit["members"]})
                ),
                "activities": "|".join(
                    activity_data[idx]["activity_code"] for idx in unit["members"]
                ),
            }
        )
    moved_units = pd.DataFrame(moved_rows)

    optimized_clashes = pd.DataFrame(
        [
            {
                "course_1": pair[0],
                "course_2": pair[1],
                "weight": clash_weights[pair],
            }
            for pair in sorted(independent_pairs)
        ]
    )

    improvement = baseline_score - independent_score
    improvement_pct = 100.0 * improvement / baseline_score if baseline_score else 0.0

    summary = {
        "baseline_score": baseline_score,
        "optimized_score": independent_score,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "baseline_clash_pairs": len(baseline_clash_pairs),
        "optimized_clash_pairs": len(independent_pairs),
        "activities": len(activity_data),
        "courses": len(week38_courses),
        "joint_groups": len(joint_groups),
        "units": len(units),
        "moved_units": len(moved_units),
        "protected_courses": len(protected_courses),
        "protected_units": len(protected_units),
        "protected_lunch_days": len(protected_lunch_rows),
        "room_conflicts": 0,
        "new_same_course_overlaps": 0,
        "new_saturday_units": 0,
        "invalid_start_units": 0,
        "solve_seconds": solve_seconds,
        "status": model.Status,
        "best_bound": float(model.ObjBound),
        "mip_gap": float(model.MIPGap),
    }

    print()
    print("=" * 78)
    print("FINAL RESULT — INDEPENDENTLY VALIDATED")
    print("=" * 78)
    print(f"Baseline score             : {baseline_score}")
    print(f"Optimized score            : {independent_score}")
    print(f"Improvement                : {improvement} ({improvement_pct:.2f}%)")
    print(f"Baseline clash pairs       : {len(baseline_clash_pairs)}")
    print(f"Optimized clash pairs      : {len(independent_pairs)}")
    print(f"Scheduling units moved     : {len(moved_units)} / {len(units)}")
    print(f"Protected S1 units moved   : 0")
    print(f"Protected lunch failures   : 0")
    print(f"Room conflicts             : 0")
    print(f"New same-course overlaps   : 0")
    print(f"New Saturday units         : 0")
    print(f"Invalid start slots > 10   : 0")
    print(f"Solve time                 : {solve_seconds:.1f}s")
    print(f"Best bound                 : {model.ObjBound:.2f}")
    print(f"MIP gap                    : {100.0 * model.MIPGap:.2f}%")
    print("✅ Final model solution passed independent validation")
    print("=" * 78)

    if write_outputs:
        optimized_schedule.to_csv(
            data_dir / f"{output_prefix}_schedule.csv",
            index=False,
            encoding="utf-8-sig",
        )
        moved_units.to_csv(
            data_dir / f"{output_prefix}_moved_units.csv",
            index=False,
            encoding="utf-8-sig",
        )
        optimized_clashes.to_csv(
            data_dir / f"{output_prefix}_clashes.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame([summary]).to_csv(
            data_dir / f"{output_prefix}_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return OptimizationResult(
        model=model,
        baseline_score=baseline_score,
        optimized_score=independent_score,
        baseline_clash_pairs=baseline_clash_pairs,
        optimized_clash_pairs=independent_pairs,
        improvement=improvement,
        improvement_pct=improvement_pct,
        positions=positions,
        optimized_schedule=optimized_schedule,
        moved_units=moved_units,
        optimized_clashes=optimized_clashes,
        summary=summary,
    )


if __name__ == "__main__":
    # Default Colab/Drive location used throughout the project.
    default_dir = Path("/content/drive/MyDrive/timetable_project")
    if not default_dir.exists():
        raise SystemExit(
            f"Default project folder not found: {default_dir}\n"
            "Import run_week38_optimization() and pass your own data_dir."
        )

    run_week38_optimization(default_dir)
