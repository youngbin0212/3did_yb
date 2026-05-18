#!/usr/bin/env python3
"""
Gesture Workspace — PRACTICE build
============================================
Identical to asl_gesture12.py except TASK_DEFS is replaced with two tiny
practice tasks (1 slot each). Goal: run a short session and confirm the
metrics logger writes non-empty CSVs covering mode_change / action /
placement / task_end(completed=True) for both C-mode and X+drag flows.

On exit, three CSVs are written to ./logs/:
  events_<id>.csv   one row per raw event (mode change, action, placement, ...)
  summary_<id>.csv  one row per task (duration, error rate, action counts)
  session_<id>.csv  session-wide totals incl. unintentional-action rate

Key 'u' marks the most recent action as unintentional (ground truth for
Gesture Recognition Accuracy / Unintentional Trigger Rate).

New layout per ui_image1.png / ui_image2.png:

  ┌─────────┬─────────┬───────────────────┬─────────────┬───────┐
  │ Stage   │ Instr.  │     Self View     │ Instruction │ Stat  │
  │ Sidebar │ View    │ (webcam)          │ Panel       │       │
  │ 1..N    ├─────────┼───────────────────┴─────────────┤       │
  │         │         │       System Feedback           │ Prog. │
  │         │ Refer-  ├───────────────────┬─────────────┤ Indi- │
  │         │ ence    │                   │             │ cator │
  │         │         │  Action           │  Task       │       │
  │         │         │  Workspace        │  Workspace  │       │
  └─────────┴─────────┴───────────────────┴─────────────┴───────┘

Left-hand ASL (mode):
  open palm   -> drag mode  (default)
  C           -> copy / paste
  F           -> highlight same kind
  X           -> delete
  Z (index)   -> undo last delete

Right-hand: index fingertip = pointer; pinch (thumb+index) = click/grab.

Install:  pip install opencv-python mediapipe numpy
Run:      python asl_gesture5.py
"""

import atexit, copy, csv, os, random, time, urllib.request
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ──────────────────────────────────────────────
#  Model
# ──────────────────────────────────────────────

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")

# ──────────────────────────────────────────────
#  Layout (1400 × 800)
# ──────────────────────────────────────────────

W, H = 1400, 800
M    = 10        # outer margin
G    = 8         # gap between panels

# Column widths
SS_W   = 110
COL2_W = 200
COL4_W = 380
COL5_W = 130
COL3_W = W - (M + SS_W + G + COL2_W + G + G + COL4_W + G + COL5_W + M)  # = 528

# Column X positions
SS_X   = M
COL2_X = SS_X + SS_W + G
COL3_X = COL2_X + COL2_W + G
COL4_X = COL3_X + COL3_W + G
COL5_X = COL4_X + COL4_W + G

# Row Y positions
TOP_Y  = M
TOP_H  = 170
FB_Y   = TOP_Y + TOP_H + G          # 188
FB_H   = 44
MAIN_Y = FB_Y + FB_H + G            # 240
MAIN_H = H - MAIN_Y - M             # 550

# Stage sidebar
SS_Y, SS_H = M, H - 2 * M

# Instructor view (col2 top)
IV_X, IV_Y, IV_W, IV_H = COL2_X, TOP_Y, COL2_W, TOP_H

# Reference (col2, below; spans through main row)
RF_X, RF_Y, RF_W = COL2_X, FB_Y, COL2_W
RF_H = H - RF_Y - M

# Self view (col3 top, centered)
SV_W = 260
SV_X = COL3_X + (COL3_W - SV_W) // 2
SV_Y, SV_H = TOP_Y, TOP_H

# System feedback (spans col3 + col4)
SF_X = COL3_X
SF_Y = FB_Y
SF_W = COL3_W + G + COL4_W
SF_H = FB_H

# Action workspace (col3 main)
AW_X, AW_Y, AW_W, AW_H = COL3_X, MAIN_Y, COL3_W, MAIN_H

# Instruction panel (col4 top)
IP_X, IP_Y, IP_W, IP_H = COL4_X, TOP_Y, COL4_W, TOP_H

# Task workspace (col4 main)
TW_X, TW_Y, TW_W, TW_H = COL4_X, MAIN_Y, COL4_W, MAIN_H

# Status (col5 top, smaller) + Progress (col5 below)
SST_X, SST_Y, SST_W, SST_H = COL5_X, TOP_Y, COL5_W, 110
PI_X, PI_Y, PI_W = COL5_X, SST_Y + SST_H + G, COL5_W
PI_H = H - PI_Y - M

# Lego defaults (per-shape sizes are recomputed per-task in _configure_slots)
LEGO_W, LEGO_H = 90, 56
SH_W, SH_H = LEGO_W, LEGO_H

# Drop zone — slot grid is sized per-task
DZ_HDR = 32
SLOT_PAD_DEFAULT = 6
SLOT_W_MAX, SLOT_H_MAX = 110, 88  # caps so bricks stay reasonable across tasks
SLOT_W_MIN, SLOT_H_MIN = 40, 34   # floors so bricks stay pinch-able in dense tasks

# Task workspace scatter — pool layout is computed per-task at runtime
# (see App._build_scatter_positions) because layout sizes vary widely.
TW_INNER_PAD = 14
TW_HEADER_H  = 32

# ──────────────────────────────────────────────
#  Colors (BGR)
# ──────────────────────────────────────────────

BG       = (28, 30, 36)
PANEL_BG = (245, 245, 242)
PANEL_BD = (170, 170, 165)
PANEL_HD = (108, 108, 100)
SIDEBAR  = (155, 155, 150)
ACTIVE   = (190, 220, 250)
FB_BG    = (38, 42, 50)
FB_BD    = (95, 100, 115)

SHAPE_COLORS = {
    "red":    ( 50, 100, 220),
    "green":  ( 70, 185,  90),
    "blue":   (200, 115,  45),
    "yellow": ( 35, 200, 210),
    "purple": (175,  55, 160),
    "grey":   (165, 165, 165),
    # Find-task distractors — visually close to the target colours so the
    # user really has to look (rather than spotting them by hue alone).
    "lime":    ( 60, 220, 175),   # yellow-green (≈ yellow AND ≈ green)
    "orange":  ( 35, 140, 235),   # orange        (≈ red)
    "pink":    (155, 140, 235),   # salmon pink   (≈ red)
    "crimson": ( 60,  55, 180),   # dark red      (≈ red)
    "olive":   ( 40, 130, 150),   # dull yellow   (≈ yellow)
    "teal":    (130, 145,  55),   # green-blue    (≈ green)
    "mint":    (140, 215, 140),   # light green   (≈ green)
}

# Colors used as random pool fillers when a task doesn't specify its own
# `pool_colors`. We exclude the find-distractor colors so they only ever
# show up in tasks that explicitly opt into them.
DEFAULT_POOL_COLORS = ["red", "green", "blue", "yellow", "purple", "grey"]

# ──────────────────────────────────────────────
#  Tasks  (target lego arrangements; reference images in tasks/)
# ──────────────────────────────────────────────

#  Practice mode — short tasks tuned to exercise as many logging paths as
#  possible within ~30s each. Together they should produce mode_change,
#  action C / F / X / drag (and Z if the user undoes), placement
#  (correct=True and optionally False), and task_end(completed=True) rows.
TASK_DEFS = [
    # Practice 1 — Copy & Paste, 1x3 row of red.
    # Intended flow: open -> C -> copy a red pool brick -> paste three
    # times onto the empty row. Exercises repeat actions in a single mode
    # and multiple placement(correct=True) events.
    {
        "name": "Practice 1 - Copy & Paste",
        "file": "practice1.png",          # missing image is fine; reference panel goes blank
        "cols": 3, "rows": 1,
        "layout": [
            (0, 0, "red"), (1, 0, "red"), (2, 0, "red"),
        ],
        "pool_colors": ["red"],
    },
    # Practice 2 — Find + Delete + Drag, 2x2 blue with two red intruders.
    # Intended flow: open -> F (highlight all red) -> X (delete both reds)
    # -> open/drag (fill empty slots with blue from the pool). Exercises
    # action F, repeated action X, multiple drags, and several mode_change
    # transitions. The pool also includes red so a wrong-colour drop
    # triggers a misplacement event (useful for error_rate testing).
    # Optional: during the X phase, do one delete then a Z (undo) before
    # continuing to capture an action Z event as well.
    {
        "name": "Practice 2 - Find + Delete + Drag",
        "file": "practice2.png",
        "cols": 2, "rows": 2,
        "layout": [
            (0, 0, "blue"), (1, 0, "blue"),
            (0, 1, "blue"), (1, 1, "blue"),
        ],
        "pre_placed": [
            (0, 0, "blue"), (1, 0, "red"),
            (0, 1, "red"),  (1, 1, "blue"),
        ],
        "pool_colors": ["blue", "red"],
    },
]

# ──────────────────────────────────────────────
#  Lego rendering
# ──────────────────────────────────────────────

def _shade(c, factor):
    return tuple(max(0, min(255, int(v * factor))) for v in c)

def draw_lego(frame, cx, cy, w, h, color, alpha_body=0.92,
              border=None, border_w=1, highlight=False, n_studs=2):
    """Draw a stylized lego brick centered at (cx, cy)."""
    x1 = int(cx - w / 2);  y1 = int(cy - h / 2)
    x2 = int(cx + w / 2);  y2 = int(cy + h / 2)
    stud_h  = max(7, h // 8)
    stud_r  = max(5, min(8, h // 8))
    body_y1 = y1 + stud_h

    # body fill (alpha-blended)
    if alpha_body < 1.0:
        ov = frame.copy()
        cv2.rectangle(ov, (x1, body_y1), (x2, y2), color, -1)
        cv2.addWeighted(ov, alpha_body, frame, 1 - alpha_body, 0, frame)
    else:
        cv2.rectangle(frame, (x1, body_y1), (x2, y2), color, -1)

    # subtle bottom shadow band
    sh = _shade(color, 0.78)
    cv2.rectangle(frame, (x1, y2 - 5), (x2, y2), sh, -1)

    # studs
    body_w = x2 - x1
    for i in range(n_studs):
        sx = x1 + (i + 1) * body_w // (n_studs + 1)
        sy = body_y1
        cv2.circle(frame, (sx, sy), stud_r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (sx, sy), stud_r, _shade(color, 0.55), 1, cv2.LINE_AA)
        # specular highlight
        cv2.circle(frame, (sx - stud_r // 3, sy - stud_r // 3),
                   max(1, stud_r // 3), _shade(color, 1.35), -1, cv2.LINE_AA)

    # outline
    bc = border if border is not None else _shade(color, 0.50)
    cv2.rectangle(frame, (x1, body_y1), (x2, y2), bc, border_w, cv2.LINE_AA)

    if highlight:
        # bright outline around full bounding box
        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2),
                      (255, 255, 255), 2, cv2.LINE_AA)


def draw_mini_lego(frame, x1, y1, w, h, color, border_w=1):
    """Compact lego for sidebar / reference previews."""
    x2, y2 = x1 + w, y1 + h
    stud_h = max(3, h // 4)
    body_y1 = y1 + stud_h
    cv2.rectangle(frame, (x1, body_y1), (x2, y2), color, -1)
    cv2.rectangle(frame, (x1, y2 - 2), (x2, y2), _shade(color, 0.75), -1)
    # 2 studs
    for i in range(2):
        sx = x1 + (i + 1) * w // 3
        cv2.circle(frame, (sx, body_y1), max(2, h // 6), color, -1, cv2.LINE_AA)
        cv2.circle(frame, (sx, body_y1), max(2, h // 6),
                   _shade(color, 0.55), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, body_y1), (x2, y2),
                  _shade(color, 0.50), border_w, cv2.LINE_AA)

# ──────────────────────────────────────────────
#  Data class
# ──────────────────────────────────────────────

@dataclass
class Shape:
    id:    int
    px:    float
    py:    float
    w:     int
    h:     int
    label: str
    kind:  str
    alive: bool  = True
    slot:  object = None

# ──────────────────────────────────────────────
#  Hand classifier  (unchanged from v3)
# ──────────────────────────────────────────────

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

def _ext(lm, tip, pip, thresh=0.025):
    return lm[tip].y < lm[pip].y - thresh

def classify_left(lm) -> str:
    idx_ext  = _ext(lm,  8,  6)
    mid_ext  = _ext(lm, 12, 10)
    ring_ext = _ext(lm, 16, 14)
    pnk_ext  = _ext(lm, 20, 18)

    idx_curl  = lm[8].y  > lm[5].y  + 0.01
    mid_curl  = lm[12].y > lm[9].y  + 0.01
    ring_curl = lm[16].y > lm[13].y + 0.01
    pnk_curl  = lm[20].y > lm[17].y + 0.01

    d_ti = float(np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y))

    if idx_ext and mid_ext and ring_ext and pnk_ext:
        return "open"
    if idx_curl and mid_curl and ring_curl and pnk_curl:
        return "fist"
    if idx_ext and not mid_ext and not ring_ext and not pnk_ext:
        return "Z"
    if d_ti < 0.07 and mid_ext and ring_ext and pnk_ext:
        return "F"
    others_fully_curled = mid_curl and ring_curl and pnk_curl
    if not mid_ext and not ring_ext and not pnk_ext and others_fully_curled:
        return "X"
    all_bent = (not idx_ext and not mid_ext and not ring_ext and not pnk_ext)
    not_fist = not (idx_curl and mid_curl and ring_curl and pnk_curl)
    if all_bent and not_fist and not others_fully_curled:
        return "C"
    return "unknown"

def pinch_dist(lm) -> float:
    """3-D distance between thumb tip (4) and index tip (8).

    Adding z (depth) on top of x,y is what makes pinch detection robust
    to camera angle: when the user's hand is rotated so the thumb passes
    behind or in front of the index, the two fingertips can project to
    nearly the same 2-D point — a pure 2-D distance would treat that as
    a pinch even though the fingers are not actually touching. z fills
    that gap because mediapipe reports depth relative to the wrist on
    roughly the same normalized scale as x.
    """
    dx = lm[4].x - lm[8].x
    dy = lm[4].y - lm[8].y
    dz = lm[4].z - lm[8].z
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)

# ──────────────────────────────────────────────
#  Metrics logger
# ──────────────────────────────────────────────

class MetricsLogger:
    """Lightweight CSV/console logger for usability metrics.

    Captures:
      - Task Completion Time (per task)
      - Error Rate (misplaced bricks per task)
      - Response Latency  (pinch -> action; gesture-first -> mode commit)
      - Action / Mode log (raw events for Unintentional Trigger analysis)
      - Right-hand detection rate (proxy for tracking loss)

    Press 'u' in the running app to mark the most recent action as
    unintentional. Recognition Accuracy / Unintentional Trigger Rate are
    derived from these marks at write time.
    """

    OUT_SUBDIR = "logs"

    def __init__(self):
        self.session_start = time.time()
        self.session_id    = time.strftime("%Y%m%d_%H%M%S")
        self.events:       list[dict] = []
        self.task_records: list[dict] = []
        self.current_task: Optional[dict] = None
        self._frames_total      = 0
        self._frames_right_hand = 0
        self._last_action_idx: Optional[int] = None
        self._written = False

    def _now(self) -> float:
        return time.time() - self.session_start

    def log_event(self, event_type: str, **kwargs) -> int:
        e = {
            "t":          round(self._now(), 3),
            "task_idx":   self.current_task["task_idx"] if self.current_task else -1,
            "task_name":  self.current_task["name"]     if self.current_task else "",
            "event_type": event_type,
        }
        e.update(kwargs)
        self.events.append(e)
        return len(self.events) - 1

    # ── task lifecycle ─────────────────────────
    def start_task(self, task_idx: int, name: str, total_slots: int):
        if self.current_task is not None:
            self.end_task(completed=False)
        self.current_task = {
            "task_idx":           task_idx,
            "name":               name,
            "start_t":            round(self._now(), 3),
            "end_t":              None,
            "duration_s":         None,
            "total_slots":        total_slots,
            "misplacements":      0,
            "correct_placements": 0,
            "mode_changes":       0,
            "action_C":           0,
            "action_F":           0,
            "action_X":           0,
            "action_Z":           0,
            "unintentional":      0,
            "completed":          False,
            "error_rate":         0.0,
        }
        self.log_event("task_start", total_slots=total_slots)

    def end_task(self, completed: bool):
        if self.current_task is None:
            return
        t = self.current_task
        t["end_t"]      = round(self._now(), 3)
        t["duration_s"] = round(t["end_t"] - t["start_t"], 3)
        t["completed"]  = completed
        attempts = t["misplacements"] + t["correct_placements"]
        t["error_rate"] = round(t["misplacements"] / attempts, 3) if attempts else 0.0
        self.task_records.append(t)
        self.log_event("task_end",
                       completed=completed,
                       duration_s=t["duration_s"],
                       misplacements=t["misplacements"],
                       correct_placements=t["correct_placements"],
                       error_rate=t["error_rate"])
        print(f"[metrics] Task {t['task_idx']+1} '{t['name']}' "
              f"{'COMPLETED' if completed else 'abandoned'}: "
              f"{t['duration_s']}s, misplaced={t['misplacements']}, "
              f"correct={t['correct_placements']}, err={t['error_rate']:.2%}, "
              f"C/F/X/Z="
              f"{t['action_C']}/{t['action_F']}/{t['action_X']}/{t['action_Z']}")
        self.current_task     = None
        self._last_action_idx = None

    # ── placement (error rate) ─────────────────
    def log_placement(self, expected: str, actual: str, source: str):
        correct = (expected == actual)
        if self.current_task is not None:
            if correct:
                self.current_task["correct_placements"] += 1
            else:
                self.current_task["misplacements"] += 1
        self.log_event("placement",
                       expected=expected, actual=actual,
                       source=source, correct=correct)

    # ── mode commit / action ───────────────────
    def log_mode_change(self, from_mode: str, to_mode: str, commit_latency_s: float):
        if self.current_task is not None:
            self.current_task["mode_changes"] += 1
        self.log_event("mode_change",
                       from_mode=from_mode, to_mode=to_mode,
                       commit_latency_s=round(commit_latency_s, 3))

    def log_action(self, action: str, response_latency_s: float, **extra) -> int:
        if self.current_task is not None:
            key = f"action_{action}"
            if key in self.current_task:
                self.current_task[key] += 1
        idx = self.log_event("action",
                             action=action,
                             response_latency_s=round(response_latency_s, 3),
                             **extra)
        self._last_action_idx = idx
        return idx

    def mark_last_unintentional(self) -> bool:
        if self._last_action_idx is None:
            return False
        e = self.events[self._last_action_idx]
        if e.get("unintentional"):
            return False
        e["unintentional"] = True
        if self.current_task is not None:
            self.current_task["unintentional"] += 1
        self.log_event("mark_unintentional",
                       action=e.get("action"),
                       ref_event_t=e["t"])
        return True

    # ── frame stats (tracking proxy) ───────────
    def log_frame(self, right_hand_detected: bool):
        self._frames_total += 1
        if right_hand_detected:
            self._frames_right_hand += 1

    # ── persist ────────────────────────────────
    def write_to_disk(self):
        if self._written:
            return
        self._written = True
        if self.current_task is not None:
            self.end_task(completed=False)

        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               self.OUT_SUBDIR)
        os.makedirs(out_dir, exist_ok=True)
        sid = self.session_id

        ev_path = os.path.join(out_dir, f"events_{sid}.csv")
        if self.events:
            keys, seen = [], set()
            for k in ("t", "task_idx", "task_name", "event_type"):
                keys.append(k); seen.add(k)
            for e in self.events:
                for k in e.keys():
                    if k not in seen:
                        keys.append(k); seen.add(k)
            with open(ev_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for e in self.events:
                    w.writerow(e)

        sm_path = os.path.join(out_dir, f"summary_{sid}.csv")
        if self.task_records:
            keys, seen = [], set()
            for r in self.task_records:
                for k in r.keys():
                    if k not in seen:
                        keys.append(k); seen.add(k)
            with open(sm_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in self.task_records:
                    w.writerow(r)

        ss_path = os.path.join(out_dir, f"session_{sid}.csv")
        total_actions  = sum(1 for e in self.events if e["event_type"] == "action")
        unintentional  = sum(1 for e in self.events if e.get("unintentional"))
        completed_n    = sum(1 for r in self.task_records if r["completed"])
        loss_rate      = 1 - (self._frames_right_hand / max(1, self._frames_total))
        with open(ss_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["session_id",              sid])
            w.writerow(["session_duration_s",      round(self._now(), 3)])
            w.writerow(["tasks_recorded",          len(self.task_records)])
            w.writerow(["tasks_completed",         completed_n])
            w.writerow(["total_actions",           total_actions])
            w.writerow(["unintentional_actions",   unintentional])
            w.writerow(["unintentional_rate",
                        round(unintentional / max(1, total_actions), 3)])
            w.writerow(["recognition_accuracy_proxy",
                        round(1 - unintentional / max(1, total_actions), 3)])
            w.writerow(["frames_total",                self._frames_total])
            w.writerow(["frames_right_hand_detected",  self._frames_right_hand])
            w.writerow(["right_hand_loss_rate",        round(loss_rate, 3)])

        print(f"[metrics] wrote {ev_path}")
        if self.task_records:
            print(f"[metrics] wrote {sm_path}")
        print(f"[metrics] wrote {ss_path}")

# ──────────────────────────────────────────────
#  App
# ──────────────────────────────────────────────

class App:
    STABLE    = 10
    # Pinch hysteresis: engage when fingers come close, release only when
    # they clearly separate. Stops the "flicker" you get with a single
    # threshold and lets the user hold a grab without trembling.
    PINCH_ON  = 0.055
    PINCH_OFF = 0.075
    HIT_PAD   = 28        # extra px around a brick for forgiving targeting
    HISTORY_MAX = 10       # number of undo steps Z mode can roll back
    GUIDE_REVEAL_S = 5.0   # how long the gesture-guide panel stays visible
                           # after a pinch (hidden by default the rest of
                           # the time so the panel doesn't crowd the UI).
    # Cursor is a weighted blend of index tip (8) and thumb tip (4).
    # 50/50 midpoint is geometrically the "pinch point" (stable through
    # the pinch motion), but users instinctively aim with the INDEX tip,
    # so a pure midpoint feels biased toward the thumb side. We weight
    # toward the index to put the cursor near where the user aims while
    # still benefiting from the midpoint's stability under pinch.
    PTR_Y_OFFSET = 44    # shift cursor up (positive = up since y is image-down)
    PTR_X_OFFSET = -18   # shift cursor left (negative = left). Compensates
                         # for the slight right-bias of the thumb-index
                         # midpoint vs. where the user feels they're aiming.
    INDEX_WEIGHT = 0.50  # cursor = INDEX_WEIGHT*index + (1-INDEX_WEIGHT)*thumb (0.5 = midpoint)
    # Slot snap: brick snaps if it overlaps a slot by at least this fraction
    # of the BRICK's area — i.e. you don't have to land it dead-center.
    SNAP_THRESHOLD = 0.35
    NOTIF_S   = 1.6
    CORRECT_S = 2.5
    MODE_BGR  = {
        "C":    (180, 120,  50),
        "F":    (175,  55, 195),
        "X":    ( 40,  40, 210),
        "Z":    ( 95, 190,  75),
        "open": (145, 145, 145),
    }
    MODE_DESC = {
        "C": "Copy / Paste",
        "F": "Highlight same kind",
        "X": "Delete",
        "Z": "Undo",
        "open": "Drag mode",
    }

    def __init__(self):
        self.task_idx       = 0
        self.shapes:        list[Shape] = []
        self.clipboard:     Optional[Shape] = None
        # Full-state undo history. Each entry is a snapshot of all
        # shapes' (alive, slot, px, py), the clipboard reference and
        # _next_id. Z mode pops one entry per pinch to roll the world
        # back. Bounded by HISTORY_MAX so the buffer doesn't grow.
        self._history:      list[dict] = []
        self.mode           = "open"
        self._buf:          list[str] = []
        self._last_left     = "unknown"
        self.highlighted:   set[int] = set()
        self.hovered:       Optional[Shape] = None
        self.dragging:      Optional[Shape] = None
        self._drag_offset   = (0.0, 0.0)
        self.r_ptr:         Optional[tuple] = None
        self._pinch_pt:     Optional[tuple] = None
        self.r_pinch        = False
        self._prev_pin      = False
        self._notif         = ""
        self._notif_t       = 0.0
        self._raw_frame     = None
        self._next_id       = 0
        self._correct_t     = 0.0
        self._stable_count  = 0

        # Metrics — created before _load_task so the first task is logged too
        self.metrics = MetricsLogger()
        atexit.register(self.metrics.write_to_disk)
        self._pinch_start_t:   Optional[float] = None
        self._gesture_first_t: Optional[float] = None
        self._pending_gesture: Optional[str]   = None
        # Instruction Panel auto-show: hidden by default, revealed for
        # GUIDE_REVEAL_S seconds whenever the user does any pinch.
        self._guide_until_t:   float = 0.0

        # Hand-photo cards for the Instruction Panel.
        self._gesture_imgs = self._load_gesture_images()

        # Load 4 tasks (target structures + reference images).
        self.tasks = self._load_tasks()

        # slot dimensions are recomputed per task
        self._configure_slots()
        self._load_task()

    # ── gesture image loader ─────────────────

    def _load_gesture_images(self):
        """Load ui_image/gesture_{c,f,x,z}.png. Returns dict keyed by letter,
        or {} if files are missing (panel falls back to plain letters)."""
        here = os.path.dirname(os.path.abspath(__file__))
        search_dirs = [
            os.path.join(here, "ui_image"),
            here,
        ]
        out = {}
        for letter in ("C", "F", "X", "Z"):
            fn = f"gesture_{letter.lower()}.png"
            for d in search_dirs:
                p = os.path.join(d, fn)
                if os.path.exists(p):
                    img = cv2.imread(p)
                    if img is not None:
                        out[letter] = img
                        break
        missing = [k for k in ("C", "F", "X", "Z") if k not in out]
        if missing:
            print(f"[info] Instruction Panel: missing gesture images for {missing} "
                  f"-> those will fall back to a letter.")
        else:
            print(f"Instruction Panel: loaded gesture images for C/F/X/Z.")
        return out

    # ── task management ──────────────────────

    def _load_tasks(self):
        """Load reference images for each TASK_DEFS entry."""
        here = os.path.dirname(os.path.abspath(__file__))
        tasks_dir = os.path.join(here, "tasks")
        result = []
        for d in TASK_DEFS:
            t = dict(d)
            p = os.path.join(tasks_dir, d["file"])
            t["image"] = cv2.imread(p) if os.path.exists(p) else None
            if t["image"] is None:
                print(f"[info] Task image missing: {p}")
            else:
                print(f"Task image loaded: {p}")
            result.append(t)
        return result

    def _configure_slots(self):
        """Compute slot/brick dimensions for the current task."""
        t = self.tasks[self.task_idx % len(self.tasks)]
        cols, rows = t["cols"], t["rows"]
        avail_w = AW_W - 16
        avail_h = AW_H - 60
        slot_pad = SLOT_PAD_DEFAULT

        slot_w = max(SLOT_W_MIN, (avail_w - (cols + 1) * slot_pad) // cols)
        slot_h = max(SLOT_H_MIN, (avail_h - DZ_HDR - (rows + 1) * slot_pad) // rows)
        slot_w = min(slot_w, SLOT_W_MAX)
        slot_h = min(slot_h, SLOT_H_MAX)

        self.slot_w   = slot_w
        self.slot_h   = slot_h
        self.slot_pad = slot_pad
        self.grid_cols = cols
        self.grid_rows = rows

        self.dz_w = cols * slot_w + (cols + 1) * slot_pad
        self.dz_h = rows * slot_h + (rows + 1) * slot_pad + DZ_HDR
        self.dz_x = AW_X + (AW_W - self.dz_w) // 2
        self.dz_y = AW_Y + (AW_H - self.dz_h) // 2 + 14

        # Brick fits inside slot with a small inset; floor keeps it
        # pinch-friendly even when the layout is dense.
        self.brick_w = max(36, int(slot_w * 0.90))
        self.brick_h = max(30, int(slot_h * 0.88))

    def _slot_rect(self, col, row):
        x = self.dz_x + self.slot_pad + col * (self.slot_w + self.slot_pad)
        y = self.dz_y + DZ_HDR + self.slot_pad + row * (self.slot_h + self.slot_pad)
        return x, y, x + self.slot_w, y + self.slot_h

    def _slot_center(self, col, row):
        x1, y1, x2, y2 = self._slot_rect(col, row)
        return (x1 + x2) // 2, (y1 + y2) // 2

    def _find_pool_free_pos(self) -> tuple:
        """Return a (px, py) inside the task workspace that doesn't overlap
        any currently-alive pool brick. Used when restoring a deleted brick
        via undo so it doesn't pile on top of an in-slot brick."""
        bw, bh = self.brick_w, self.brick_h
        # Try a generous grid first; pick the first cell with no overlap.
        for cx, cy in self._build_scatter_positions(24):
            px = TW_X + cx
            py = TW_Y + cy
            clash = False
            for s in self.shapes:
                if not s.alive or s.slot is not None:
                    continue
                if abs(s.px - px) < bw * 0.9 and abs(s.py - py) < bh * 0.9:
                    clash = True
                    break
            if not clash:
                return px, py
        # Fallback: random spot inside TW (shouldn't normally hit this).
        px = TW_X + random.randint(TW_INNER_PAD + bw // 2,
                                   max(TW_INNER_PAD + bw // 2,
                                       TW_W - TW_INNER_PAD - bw // 2))
        py = TW_Y + random.randint(TW_HEADER_H + bh // 2,
                                   max(TW_HEADER_H + bh // 2,
                                       TW_H - bh // 2 - 6))
        return px, py

    # ── undo history (full-state snapshots) ────
    def _snapshot(self) -> dict:
        """Capture enough state to restore the world to this moment.
        Used by Z mode to roll back drag / paste / delete actions."""
        return {
            "shapes": [(s.id, s.alive, s.slot, s.px, s.py) for s in self.shapes],
            "clipboard_id": self.clipboard.id if self.clipboard else None,
            "next_id": self._next_id,
        }

    def _push_history(self):
        """Save the current state, dropping the oldest if over the cap."""
        self._history.append(self._snapshot())
        if len(self._history) > self.HISTORY_MAX:
            self._history.pop(0)

    def _restore_snapshot(self, snap: dict):
        """Roll the world back to a previously captured state. Any shapes
        created after the snapshot (e.g. a paste) are dropped; existing
        shapes have their alive/slot/position restored."""
        valid_ids = {sid for sid, *_ in snap["shapes"]}
        # Drop shapes that didn't exist at snapshot time.
        self.shapes = [s for s in self.shapes if s.id in valid_ids]
        by_id = {s.id: s for s in self.shapes}
        for sid, alive, slot, px, py in snap["shapes"]:
            s = by_id.get(sid)
            if s is None:
                continue
            s.alive = alive
            s.slot  = slot
            s.px    = px
            s.py    = py
        self.clipboard = by_id.get(snap["clipboard_id"])
        self._next_id  = snap["next_id"]
        # Drag state and find-highlight don't survive a rollback.
        self.dragging   = None
        self.highlighted.clear()

    def _build_scatter_positions(self, n: int):
        """Lay out n brick centers in a grid that fills the task workspace.
        Grid columns/rows are chosen to roughly match the workspace aspect
        ratio so cells stay square-ish for any pool size."""
        if n <= 0:
            return []
        avail_w = TW_W - 2 * TW_INNER_PAD
        avail_h = TW_H - TW_HEADER_H - 12
        aspect  = avail_w / max(1, avail_h)
        cols = max(1, int(round((n * aspect) ** 0.5)))
        rows = max(1, (n + cols - 1) // cols)
        while cols * rows < n:
            cols += 1
        cell_w = avail_w // cols
        cell_h = avail_h // rows
        out = []
        for r in range(rows):
            for c in range(cols):
                cx = TW_INNER_PAD + cell_w * c + cell_w // 2
                cy = TW_HEADER_H  + cell_h * r + cell_h // 2
                out.append((cx, cy))
        return out[:n]

    def _nearest_slot(self, px, py, brick_w=None, brick_h=None):
        """Find the slot a brick at (px, py) overlaps the most. Returns
        (col, row) only if that overlap exceeds SNAP_THRESHOLD of the
        brick's area — close enough is good enough.

        If brick_w/brick_h are not given, falls back to the dragged shape
        (or the task default brick size).
        """
        if brick_w is None or brick_h is None:
            if self.dragging is not None:
                brick_w = self.dragging.w
                brick_h = self.dragging.h
            else:
                brick_w = self.brick_w
                brick_h = self.brick_h

        bx1, by1 = px - brick_w / 2.0, py - brick_h / 2.0
        bx2, by2 = px + brick_w / 2.0, py + brick_h / 2.0
        brick_area = max(1.0, brick_w * brick_h)

        t = self.tasks[self.task_idx % len(self.tasks)]
        best = None
        best_ratio = 0.0
        for col, row, _ in t["layout"]:
            sx1, sy1, sx2, sy2 = self._slot_rect(col, row)
            ix1, iy1 = max(bx1, sx1), max(by1, sy1)
            ix2, iy2 = min(bx2, sx2), min(by2, sy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            overlap = (ix2 - ix1) * (iy2 - iy1)
            ratio = overlap / brick_area
            if ratio > best_ratio:
                best_ratio = ratio
                best = (col, row)

        return best if best_ratio >= self.SNAP_THRESHOLD else None

    def _load_task(self):
        """Reset state, place any pre-built bricks into slots, then spawn
        the pool of remaining bricks in the task workspace."""
        # If a previous task was active and we're reloading without having
        # marked it complete, treat it as abandoned (e.g., manual jump).
        self.metrics.end_task(completed=False)

        self._configure_slots()
        self.shapes = []
        self._history      = []
        self.clipboard = None
        self.highlighted.clear()

        t = self.tasks[self.task_idx % len(self.tasks)]
        layout = t["layout"]
        pre_placed = t.get("pre_placed", [])

        # 1) Pre-placed bricks go directly into their slots (action workspace).
        labels_pre = "abcdefghijklmnop"
        for i, (col, row, kind) in enumerate(pre_placed):
            cx, cy = self._slot_center(col, row)
            self.shapes.append(Shape(
                id    = self._next_id,
                px    = float(cx),
                py    = float(cy),
                w     = self.brick_w, h = self.brick_h,
                label = labels_pre[i % len(labels_pre)],
                kind  = kind,
                slot  = (col, row),
            ))
            self._next_id += 1

        # 2) Pool: only the colors the user still needs to add to win
        #    (slots whose pre-placed kind doesn't match the layout) plus
        #    distractors. Pool size scales with what the layout actually
        #    needs — small tasks get a tight pool, big tasks get a large
        #    one so every required brick fits.
        placed = {(c, r): k for c, r, k in pre_placed}
        needed = [exp for c, r, exp in layout if placed.get((c, r)) != exp]
        pool_size = max(len(needed) + 3, 12)
        positions = self._build_scatter_positions(pool_size)

        pool = list(needed)
        # Find-task tasks specify their own narrow distractor palette
        # (target + visually-similar colors). Other tasks fall back to
        # the default 6-colour set, so the new distractor colours never
        # leak into Gate / Bridge / Cleanup / Sunshine etc.
        allowed = t.get("pool_colors") or DEFAULT_POOL_COLORS
        while len(pool) < pool_size:
            pool.append(random.choice(allowed))
        pool = pool[:pool_size]
        random.shuffle(pool)

        labels = "ABCDEFGHIJKLMNOPQRSTUVWXY"
        for i, (rx, ry) in enumerate(positions):
            jx = rx + random.randint(-4, 4)
            jy = ry + random.randint(-3, 3)
            self.shapes.append(Shape(
                id    = self._next_id,
                px    = float(TW_X + jx),
                py    = float(TW_Y + jy),
                w     = self.brick_w, h = self.brick_h,
                label = labels[i % len(labels)],
                kind  = pool[i],
            ))
            self._next_id += 1

        # Begin metrics record for this task
        self.metrics.start_task(self.task_idx, t["name"], len(layout))

    def _slot_expected(self, slot):
        """Return the expected color for a slot in the current task, or None."""
        t = self.tasks[self.task_idx % len(self.tasks)]
        for col, row, exp in t["layout"]:
            if (col, row) == slot:
                return exp
        return None

    def _is_paste_target(self, shape) -> bool:
        """True if a pinch in C mode would PASTE (replace) onto `shape`
        instead of copying it. Triggered when clipboard has content and
        the hovered brick is sitting in a slot — i.e. the user is aiming
        at the drop zone to swap a piece."""
        return (self.mode == "C"
                and self.clipboard is not None
                and shape is not None
                and shape.slot is not None)

    def _check_answer(self) -> bool:
        t = self.tasks[self.task_idx % len(self.tasks)]
        for col, row, expected in t["layout"]:
            occ = next((s for s in self.shapes
                        if s.alive and s.slot == (col, row)), None)
            if occ is None or occ.kind != expected:
                return False
        return True

    def _next_task(self):
        self.task_idx += 1
        self._correct_t = 0.0
        self._load_task()
        idx = self.task_idx % len(self.tasks)
        self._notify(f"Task {idx + 1}: {self.tasks[idx]['name']}!")

    def jump_to_task(self, idx: int):
        """Jump directly to task `idx` (0-based). No-op if out of range."""
        if not (0 <= idx < len(self.tasks)):
            return
        self.task_idx  = idx
        self._correct_t = 0.0
        self._load_task()
        self._notify(f"Jumped to Task {idx + 1}: {self.tasks[idx]['name']}")

    # ── helpers ──────────────────────────────

    def _push(self, g: str):
        self._last_left = g

        # Commit-latency tracking: record when a new gesture first appears,
        # so we can measure how long until it stabilizes into a mode.
        if g != "unknown":
            if self._pending_gesture != g:
                self._pending_gesture = g
                self._gesture_first_t = time.time()
        else:
            self._pending_gesture = None
            self._gesture_first_t = None

        self._buf.append(g)
        if len(self._buf) > self.STABLE:
            self._buf.pop(0)
        if len(self._buf) == self.STABLE and len(set(self._buf)) == 1:
            self._stable_count = self.STABLE
            new = self._buf[0]
            if new in ("C", "F", "X", "Z", "open"):
                if new != self.mode:
                    latency = ((time.time() - self._gesture_first_t)
                               if self._gesture_first_t else 0.0)
                    old_mode = self.mode
                    # Clear the F-mode "find" highlight on any mode change.
                    self.highlighted.clear()
                    self.mode = new
                    self.metrics.log_mode_change(old_mode, new, latency)
                    self._pending_gesture = None
                    self._gesture_first_t = None
        else:
            # count contiguous matches at the tail
            tail = self._buf[-1]
            n = 0
            for x in reversed(self._buf):
                if x == tail: n += 1
                else: break
            self._stable_count = n

    def _hit(self, px, py, pad=None) -> Optional[Shape]:
        if pad is None:
            pad = self.HIT_PAD
        for s in reversed(self.shapes):
            if s.alive and abs(px - s.px) < s.w / 2 + pad \
                       and abs(py - s.py) < s.h / 2 + pad:
                return s
        return None

    def _notify(self, msg: str):
        self._notif   = msg
        self._notif_t = time.time()

    def _action_hint(self) -> str:
        """One-line preview of what the next pinch will do, given the
        current mode + cursor + clipboard state. Empty string when the
        action would be a no-op or the mode doesn't need a hint."""
        if self.mode == "C":
            paste_over = self._is_paste_target(self.hovered)
            if self.hovered is not None and not paste_over:
                return f"Copy '{self.hovered.label}'"
            if self.clipboard is not None:
                if paste_over:
                    return (f"Paste '{self.clipboard.label}' "
                            f"(replace '{self.hovered.label}')")
                return f"Paste '{self.clipboard.label}'"
            return "(clipboard empty)"
        if self.mode == "F" and self.hovered is not None:
            return f"Find {self.hovered.kind} pieces"
        if self.mode == "X" and self.hovered is not None:
            return f"Delete '{self.hovered.label}'"
        if self.mode == "Z" and self._history:
            return f"Undo (stack: {len(self._history)})"
        return ""

    # ── execute pinch ─────────────────────────

    def _execute(self):
        obj = self.hovered
        latency = ((time.time() - self._pinch_start_t)
                   if self._pinch_start_t else 0.0)
        if self.mode == "C":
            paste_over = self._is_paste_target(obj)
            if obj and not paste_over:
                self.clipboard = copy.copy(obj)
                self._notify(f"Copied: {obj.label}")
                self.metrics.log_action("C", latency,
                                        subaction="copy",
                                        target=obj.label, kind=obj.kind)
            elif self.clipboard and self._pinch_pt:
                self._push_history()
                new       = copy.copy(self.clipboard)
                new.id    = self._next_id
                # bbox center sits slightly above the visual body center
                # (stud on top), so compensate for body alignment.
                stud_h    = max(7, new.h // 8)
                cx, cy    = self._pinch_pt[0], self._pinch_pt[1] - stud_h / 2

                # If the cursor's virtual brick overlaps a slot enough,
                # snap to that slot's center instead of the raw cursor.
                snap = self._nearest_slot(
                    cx, cy, brick_w=new.w, brick_h=new.h
                )
                if snap is not None:
                    # evict any current occupant of the target slot —
                    # also relocate it to the pool so its visual position
                    # matches its (now slot-less) logical state (see the
                    # drag-eviction comment above for the bug this fixes).
                    for other in self.shapes:
                        if other.alive and other.slot == snap:
                            other.slot = None
                            fx, fy = self._find_pool_free_pos()
                            other.px, other.py = float(fx), float(fy)
                    sx, sy   = self._slot_center(*snap)
                    new.px   = float(sx)
                    new.py   = float(sy)
                    new.slot = snap
                else:
                    new.px   = cx
                    new.py   = cy
                    new.slot = None

                new.label = self.clipboard.label + "'"
                new.alive = True
                self.shapes.append(new)
                self._next_id += 1

                # Placement log (paste landed on a slot)
                if snap is not None:
                    self.metrics.log_placement(
                        expected=self._slot_expected(snap),
                        actual=new.kind, source="paste")

                self.metrics.log_action("C", latency,
                                        subaction="paste",
                                        pasted=new.label, kind=new.kind,
                                        snapped=(snap is not None))

                # snapping into the right slot can complete the task
                if snap is not None and self._check_answer():
                    self._correct_t = time.time()
                    self.metrics.end_task(completed=True)

                self._notify(f"Pasted: {new.label}"
                             + (" -> slot" if snap is not None else ""))
            else:
                self._notify("Clipboard empty")
                self.metrics.log_action("C", latency, subaction="empty_clipboard")
        elif self.mode == "F":
            if obj:
                self.highlighted = {s.id for s in self.shapes
                                    if s.alive and s.kind == obj.kind}
                self._notify(f"{obj.kind}: {len(self.highlighted)} highlighted")
                self.metrics.log_action("F", latency,
                                        kind=obj.kind,
                                        count=len(self.highlighted))
        elif self.mode == "X":
            if obj:
                self._push_history()
                obj.alive = False
                obj.slot  = None
                self.highlighted.discard(obj.id)
                self._notify(f"Deleted: {obj.label}")
                self.metrics.log_action("X", latency,
                                        target=obj.label, kind=obj.kind)
        elif self.mode == "Z":
            if self._history:
                self._restore_snapshot(self._history.pop())
                self._notify(f"Undo (steps left: {len(self._history)})")
                self.metrics.log_action("Z", latency,
                                        remaining=len(self._history))

    # ── update ───────────────────────────────

    def update(self, result, raw_frame):
        self._raw_frame = raw_frame

        if self._correct_t > 0:
            if time.time() - self._correct_t > self.CORRECT_S:
                self._next_task()
            return

        left_g = "unknown"
        r_lm   = None
        for lm_list, hd_list in zip(result.hand_landmarks, result.handedness):
            # camera was flipped horizontally -> swap left/right semantics
            if hd_list[0].category_name == "Right":   # user's left
                left_g = classify_left(lm_list)
            else:
                r_lm = lm_list

        self.metrics.log_frame(right_hand_detected=(r_lm is not None))

        self._push(left_g)

        if r_lm:
            # Pointer = weighted blend of index tip (8) and thumb tip (4).
            w = self.INDEX_WEIGHT
            ix = (w * r_lm[8].x + (1 - w) * r_lm[4].x) * W + self.PTR_X_OFFSET
            iy = (w * r_lm[8].y + (1 - w) * r_lm[4].y) * H - self.PTR_Y_OFFSET
            self.r_ptr = (ix, iy)

            # Hysteresis: lower threshold to engage, higher to release.
            pd = pinch_dist(r_lm)
            if self.r_pinch:
                new_pinch = pd < self.PINCH_OFF
            else:
                new_pinch = pd < self.PINCH_ON

            # Pinch-start timestamp for response latency
            if new_pinch and not self._prev_pin:
                self._pinch_start_t = time.time()
                # Reveal the gesture-guide panel only when the pinch lands
                # inside the Instruction Panel rectangle. Pinches elsewhere
                # (workspace, pool, sidebar) leave it hidden so the panel
                # doesn't pop open every time the user grabs a brick.
                if (IP_X <= ix <= IP_X + IP_W
                        and IP_Y <= iy <= IP_Y + IP_H):
                    self._guide_until_t = time.time() + self.GUIDE_REVEAL_S

            # Anchor the cursor to the index fingertip in BOTH states so the
            # aiming point doesn't shift when the thumb closes.
            self._pinch_pt = (ix, iy)

            if self.mode == "open":
                if new_pinch:
                    if not self._prev_pin:
                        hit = self._hit(ix, iy)
                        if hit:
                            # Snapshot BEFORE we mutate the picked-up
                            # brick, so undo can fully rewind the drag.
                            self._push_history()
                            self.dragging     = hit
                            hit.slot          = None
                            self._drag_offset = (hit.px - ix, hit.py - iy)
                    if self.dragging:
                        self.dragging.px = ix + self._drag_offset[0]
                        self.dragging.py = iy + self._drag_offset[1]
                else:
                    if self.dragging:
                        s = self._nearest_slot(self.dragging.px, self.dragging.py)
                        if s is not None:
                            for other in self.shapes:
                                if other.id != self.dragging.id and other.slot == s:
                                    other.slot = None
                                    # Send the displaced brick back to the
                                    # pool so its visual position matches
                                    # its (now slot-less) logical state.
                                    # Otherwise it keeps sitting at the
                                    # slot's coordinates; if the user
                                    # later X-deletes whatever displaced
                                    # it, the original brick LOOKS like
                                    # it's in the slot but actually has
                                    # slot=None, so the correct-state
                                    # green never turns on and the task
                                    # can't complete.
                                    fx, fy = self._find_pool_free_pos()
                                    other.px, other.py = float(fx), float(fy)
                            self.dragging.slot = s
                            cx, cy = self._slot_center(*s)
                            self.dragging.px = float(cx)
                            self.dragging.py = float(cy)
                            # Placement log (drag landed on a slot)
                            self.metrics.log_placement(
                                expected=self._slot_expected(s),
                                actual=self.dragging.kind, source="drag")
                            # Drag-release response latency relative to pinch start
                            self.metrics.log_action(
                                "drag", (time.time() - self._pinch_start_t)
                                        if self._pinch_start_t else 0.0,
                                target=self.dragging.label,
                                kind=self.dragging.kind,
                                snapped=True)
                        else:
                            self.dragging.slot = None
                            self.metrics.log_action(
                                "drag", (time.time() - self._pinch_start_t)
                                        if self._pinch_start_t else 0.0,
                                target=self.dragging.label,
                                kind=self.dragging.kind,
                                snapped=False)
                        if self._check_answer():
                            self._correct_t = time.time()
                            self.metrics.end_task(completed=True)
                    self.dragging = None

            self.r_pinch = new_pinch
            self.hovered = self._hit(ix, iy)
        else:
            self.r_ptr = self.hovered = self._pinch_pt = None
            self.r_pinch  = False
            self.dragging = None

        if self.r_pinch and not self._prev_pin and self.mode != "open":
            self._execute()
        self._prev_pin = self.r_pinch

    # ── low-level draw helpers ───────────────

    def _put(self, frame, text, x, y, scale, color, thick=1):
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def _fill(self, frame, x1, y1, x2, y2, color, alpha):
        if alpha >= 1.0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            return
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

    def _panel(self, frame, x, y, w, h, title=None, bg=PANEL_BG, alpha=0.97):
        self._fill(frame, x, y, x + w, y + h, bg, alpha)
        cv2.rectangle(frame, (x, y), (x + w, y + h), PANEL_BD, 1)
        if title:
            self._put(frame, title, x + 10, y + 22, 0.50, PANEL_HD, 1)
            cv2.line(frame, (x + 8, y + 30), (x + w - 8, y + 30), PANEL_BD, 1)

    def _draw_layout_mini(self, frame, x, y, w, h, layout):
        """Render a layout as mini lego tiles fitted into (x, y, w, h).
        Used as a fallback for tasks that don't have a reference image."""
        if not layout:
            return
        min_c = min(c for c, _, _ in layout)
        max_c = max(c for c, _, _ in layout)
        min_r = min(r for _, r, _ in layout)
        max_r = max(r for _, r, _ in layout)
        used_c = max_c - min_c + 1
        used_r = max_r - min_r + 1
        pad = 3
        cell = max(8, min((w - pad * (used_c + 1)) // used_c,
                          (h - pad * (used_r + 1)) // used_r))
        grid_w = used_c * cell + (used_c - 1) * pad
        grid_h = used_r * cell + (used_r - 1) * pad
        gx = x + (w - grid_w) // 2
        gy = y + (h - grid_h) // 2
        for col, row, kind in layout:
            bx = gx + (col - min_c) * (cell + pad)
            by = gy + (row - min_r) * (cell + pad)
            draw_mini_lego(frame, bx, by, cell, cell, SHAPE_COLORS[kind])

    # ── high-level panel drawers ─────────────

    def _draw_stage_sidebar(self, frame):
        cv2.rectangle(frame, (SS_X, SS_Y),
                      (SS_X + SS_W, SS_Y + SS_H), SIDEBAR, -1)
        cv2.rectangle(frame, (SS_X, SS_Y),
                      (SS_X + SS_W, SS_Y + SS_H), PANEL_BD, 1)

        n = len(self.tasks)
        pad = 6
        cell_h = (SS_H - pad * (n + 1)) // n
        cell_w = SS_W - 2 * pad
        cur = self.task_idx % n

        for i in range(n):
            x1 = SS_X + pad
            y1 = SS_Y + pad + i * (cell_h + pad)
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            is_cur  = (i == cur)
            is_done = (i < cur)
            bg_col  = ACTIVE if is_cur else (235, 235, 230)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bg_col, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (90, 90, 80) if is_cur else (185, 185, 180),
                          2 if is_cur else 1)

            # task image thumbnail
            img = self.tasks[i].get("image")
            label_band_h = 22
            thumb_h = cell_h - label_band_h - 8
            thumb_w = cell_w - 8
            if img is not None and thumb_h > 4 and thumb_w > 4:
                sh, sw = img.shape[:2]
                scale = min(thumb_w / sw, thumb_h / sh)
                rw, rh = max(1, int(sw * scale)), max(1, int(sh * scale))
                resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
                ox = x1 + (cell_w - rw) // 2
                oy = y1 + 4 + (thumb_h - rh) // 2
                frame[oy:oy + rh, ox:ox + rw] = resized

            # done check
            if is_done:
                cv2.line(frame, (x2 - 16, y1 + 8), (x2 - 11, y1 + 13),
                         (60, 160, 60), 2, cv2.LINE_AA)
                cv2.line(frame, (x2 - 11, y1 + 13), (x2 - 5, y1 + 5),
                         (60, 160, 60), 2, cv2.LINE_AA)

            self._put(frame, f"Task {i + 1}",
                      x1 + 8, y2 - 8, 0.38,
                      (60, 60, 55) if is_cur else (80, 80, 75))

    def _draw_instructor(self, frame):
        self._panel(frame, IV_X, IV_Y, IV_W, IV_H, "Instructor")
        cx = IV_X + IV_W // 2
        cy = IV_Y + IV_H // 2 + 14
        # silhouette
        cv2.circle(frame, (cx, cy - 30), 22, (55, 55, 55), -1, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy + 36), (52, 38), 0, 180, 360,
                    (55, 55, 55), -1, cv2.LINE_AA)
        # mode hint
        hint = self.MODE_DESC.get(self.mode, self.mode)
        (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        self._put(frame, hint, cx - tw // 2, IV_Y + IV_H - 12,
                  0.42, (95, 95, 90))

    def _draw_reference(self, frame):
        self._panel(frame, RF_X, RF_Y, RF_W, RF_H, "Reference")

        idx = self.task_idx % len(self.tasks)
        t = self.tasks[idx]
        layout = t["layout"]

        # subtitle (task name)
        title = f"Task {idx + 1}: {t['name']}"
        self._put(frame, title, RF_X + 12, RF_Y + 52, 0.46, (80, 80, 75))

        # task image (the visual goal)
        img_top    = RF_Y + 60
        img_bottom = RF_Y + RF_H - 30
        img = t.get("image")
        if img is not None:
            sh, sw = img.shape[:2]
            avail_w = RF_W - 16
            avail_h = img_bottom - img_top
            scale = min(avail_w / sw, avail_h / sh)
            rw, rh = max(1, int(sw * scale)), max(1, int(sh * scale))
            resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
            ox = RF_X + (RF_W - rw) // 2
            oy = img_top + (avail_h - rh) // 2
            frame[oy:oy + rh, ox:ox + rw] = resized
            cv2.rectangle(frame, (ox - 1, oy - 1),
                          (ox + rw + 1, oy + rh + 1),
                          (180, 180, 175), 1)
        else:
            # Fallback: render the target layout as a small lego grid so
            # the reference panel still has something useful to look at
            # when a task image (e.g., task5.png) hasn't been provided.
            self._draw_layout_mini(
                frame, RF_X + 12, img_top,
                RF_W - 24, img_bottom - img_top, layout,
            )

        # placement progress
        placed_n = sum(
            1 for col, row, kind in layout
            if any(s.alive and s.slot == (col, row) and s.kind == kind
                   for s in self.shapes)
        )
        cap = f"{placed_n} / {len(layout)} placed"
        self._put(frame, cap, RF_X + 12, RF_Y + RF_H - 14,
                  0.42, (95, 95, 90))

    def _draw_self_view(self, frame):
        self._panel(frame, SV_X, SV_Y, SV_W, SV_H, "Self View")
        if self._raw_frame is None:
            return
        in_x, in_y = SV_X + 8, SV_Y + 36
        in_w, in_h = SV_W - 16, SV_H - 44
        sh, sw = self._raw_frame.shape[:2]
        scale = min(in_w / sw, in_h / sh)
        rw, rh = int(sw * scale), int(sh * scale)
        thumb = cv2.resize(self._raw_frame, (rw, rh))
        ox = in_x + (in_w - rw) // 2
        oy = in_y + (in_h - rh) // 2
        frame[oy:oy + rh, ox:ox + rw] = thumb
        cv2.rectangle(frame, (ox, oy), (ox + rw, oy + rh),
                      (160, 160, 155), 1)

    def _draw_feedback_bar(self, frame):
        self._fill(frame, SF_X, SF_Y, SF_X + SF_W, SF_Y + SF_H, FB_BG, 1.0)
        cv2.rectangle(frame, (SF_X, SF_Y),
                      (SF_X + SF_W, SF_Y + SF_H), FB_BD, 1)

        if self.mode == "open":
            primary = "[Left Hand] open palm  |  Drag mode"
            mc      = (190, 200, 210)
        else:
            primary = (f"[Left Hand] {self.mode}  |  "
                       f"{self.MODE_DESC.get(self.mode, '')}  detected")
            mc = self.MODE_BGR.get(self.mode, (190, 200, 210))

        # Tells the user what their next pinch will actually do — crucial
        # in C mode where pinch can mean either copy or paste depending
        # on whether the cursor is over a brick.
        hint = self._action_hint()
        if hint:
            primary += f"   ->  {hint}"

        self._put(frame, primary, SF_X + 16, SF_Y + 28, 0.56, mc, 1)

        # transient notification on the right
        if self._notif and time.time() - self._notif_t < self.NOTIF_S:
            nt = f">  {self._notif}"
            (nw, _), _ = cv2.getTextSize(nt, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.50, 1)
            self._put(frame, nt, SF_X + SF_W - nw - 16, SF_Y + 28,
                      0.50, (220, 225, 200))

    def _draw_instruction_panel(self, frame):
        self._panel(frame, IP_X, IP_Y, IP_W, IP_H, "Gesture Shortcuts")

        # Hide the gesture cards by default. They reveal for
        # GUIDE_REVEAL_S after any pinch, then fade back to hidden so
        # the panel doesn't crowd the screen during normal play.
        if time.time() >= self._guide_until_t:
            hint = "Pinch to show gesture guide"
            (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.46, 1)
            self._put(frame, hint,
                      IP_X + (IP_W - tw) // 2,
                      IP_Y + IP_H // 2 + 4,
                      0.46, (140, 140, 135))
            return

        cards = [
            ("C", "Copy",   self.MODE_BGR["C"]),
            ("F", "Find",   self.MODE_BGR["F"]),
            ("X", "Delete", self.MODE_BGR["X"]),
            ("Z", "Undo",   self.MODE_BGR["Z"]),
        ]
        n = len(cards)
        pad = 8
        ix = IP_X + pad
        iy = IP_Y + 38
        iw = IP_W - 2 * pad
        ih = IP_H - 46
        cw = (iw - (n - 1) * pad) // n

        # vertical split inside each card: photo on top, label below
        LABEL_H = 22
        IMG_PAD = 4

        for i, (letter, label, color) in enumerate(cards):
            x1 = ix + i * (cw + pad)
            y1 = iy
            x2 = x1 + cw
            y2 = y1 + ih
            active = (self.mode == letter)

            # card background
            if active:
                self._fill(frame, x1, y1, x2, y2, color, 0.18)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (235, 235, 230), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (180, 180, 175), 1)

            # photo region
            img_x1 = x1 + IMG_PAD
            img_x2 = x2 - IMG_PAD
            img_y1 = y1 + IMG_PAD
            img_y2 = y2 - LABEL_H - IMG_PAD
            img = self._gesture_imgs.get(letter)

            if img is not None:
                sh, sw = img.shape[:2]
                avail_w = img_x2 - img_x1
                avail_h = img_y2 - img_y1
                scale = min(avail_w / sw, avail_h / sh)
                rw, rh = max(1, int(sw * scale)), max(1, int(sh * scale))
                resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
                ox = img_x1 + (avail_w - rw) // 2
                oy = img_y1 + (avail_h - rh) // 2
                frame[oy:oy + rh, ox:ox + rw] = resized
            else:
                # fallback: big letter
                (lw, lh), _ = cv2.getTextSize(letter,
                                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
                lx = x1 + (cw - lw) // 2
                ly = img_y1 + (img_y2 - img_y1 + lh) // 2
                self._put(frame, letter, lx, ly, 1.5,
                          color if active else (90, 90, 85), 4)

            # divider above label
            cv2.line(frame, (x1 + 6, y2 - LABEL_H),
                     (x2 - 6, y2 - LABEL_H),
                     color if active else (200, 200, 195), 1)

            # label text
            (tw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.50, 1)
            tx = x1 + (cw - tw) // 2
            ty = y2 - 6
            self._put(frame, label, tx, ty, 0.50,
                      color if active else (65, 65, 60),
                      2 if active else 1)

    def _draw_action_workspace(self, frame):
        self._panel(frame, AW_X, AW_Y, AW_W, AW_H, "Action Workspace")
        # drop zone
        dz_border = (75, 205, 75) if self._correct_t > 0 else (155, 155, 145)
        self._fill(frame, self.dz_x, self.dz_y,
                   self.dz_x + self.dz_w, self.dz_y + self.dz_h,
                   (215, 215, 210), 0.45)
        cv2.rectangle(frame, (self.dz_x, self.dz_y),
                      (self.dz_x + self.dz_w, self.dz_y + self.dz_h),
                      dz_border, 2)
        self._put(frame, "DROP ZONE", self.dz_x + 10, self.dz_y + 22,
                  0.48, (100, 100, 95))

        # Slot preview:
        #   - while dragging (pinch + dragging shape) -> cyan: where it'll snap
        #   - in C mode with clipboard ready, just hovering -> yellow: where
        #     a paste pinch would land
        drag_preview = None
        paste_preview = None
        if self.dragging is not None and self.r_pinch:
            drag_preview = self._nearest_slot(
                self.dragging.px, self.dragging.py
            )
        elif (self.mode == "C" and self.clipboard is not None
              and self.r_ptr is not None
              and (self.hovered is None
                   or self._is_paste_target(self.hovered))):
            # Show paste preview when the next pinch would commit a paste:
            # over empty space OR over a brick that's already in a slot
            # (the paste will replace it — needed for Task 5 cleanup).
            cx, cy = self.r_ptr
            stud_h = max(7, self.clipboard.h // 8)
            paste_preview = self._nearest_slot(
                cx, cy - stud_h / 2.0,
                brick_w=self.clipboard.w, brick_h=self.clipboard.h,
            )

        # slots derived from current task layout
        t = self.tasks[self.task_idx % len(self.tasks)]
        for col, row, expected in t["layout"]:
            x1, y1, x2, y2 = self._slot_rect(col, row)
            occ = next((s for s in self.shapes
                        if s.alive and s.slot == (col, row)), None)
            correct = occ is not None and occ.kind == expected
            is_drag_pv  = (drag_preview  == (col, row)) and not correct
            is_paste_pv = (paste_preview == (col, row)) and not correct
            exp_col = SHAPE_COLORS[expected]

            if not correct:
                fill_alpha = 0.28 if (is_drag_pv or is_paste_pv) else 0.10
                self._fill(frame, x1, y1, x2, y2, exp_col, fill_alpha)

            if correct:
                slot_bc, slot_bw = (75, 205, 75), 2
            elif is_paste_pv:
                slot_bc, slot_bw = (60, 220, 245), 3   # yellow (BGR)
            elif is_drag_pv:
                slot_bc, slot_bw = (75, 235, 230), 3   # cyan
            else:
                slot_bc, slot_bw = (155, 155, 150), 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), slot_bc, slot_bw)

            if not correct:
                self._put(frame, expected[0].upper(),
                          x1 + 5, y2 - 6, 0.36, exp_col)

    def _draw_task_workspace(self, frame):
        self._panel(frame, TW_X, TW_Y, TW_W, TW_H, "Task Workspace")
        # subtle grid hint at the bottom info
        n_alive = sum(1 for s in self.shapes if s.alive and s.slot is None)
        self._put(frame, f"{n_alive} pieces",
                  TW_X + TW_W - 80, TW_Y + 22, 0.42, (110, 110, 105))

    def _draw_shapes(self, frame):
        for s in self.shapes:
            if not s.alive:
                continue
            col = SHAPE_COLORS[s.kind]
            is_hov  = bool(self.hovered  and self.hovered.id  == s.id)
            is_drag = bool(self.dragging and self.dragging.id == s.id)
            is_hl   = s.id in self.highlighted
            in_slot = s.slot is not None

            alpha = (0.96 if is_drag else
                     0.90 if (is_hov or in_slot or is_hl) else 0.85)
            # When hovered in a non-open mode, paint the outline in that
            # mode's color so the user can tell at a glance which action
            # the next pinch will fire on this brick.
            mode_action = is_hov and self.mode in ("C", "F", "X")
            # In C mode, hovering an in-slot brick will PASTE (replace)
            # rather than copy — show that with the same yellow used for
            # the slot paste preview.
            paste_target = is_hov and self._is_paste_target(s)
            bw = (3 if is_drag else
                  3 if mode_action else
                  3 if is_hl else
                  2 if (is_hov or in_slot) else 1)
            border = None
            if is_drag:
                border = (255, 255, 255)
            elif paste_target:
                border = (60, 220, 245)            # yellow paste indicator
            elif mode_action:
                border = self.MODE_BGR[self.mode]
            elif is_hov:
                border = _shade(col, 0.45)
            elif is_hl:
                # F-mode "find" highlight: bright hot pink so highlighted
                # bricks pop visibly regardless of the brick's own color.
                border = (180, 105, 255)

            draw_lego(frame, int(s.px), int(s.py), s.w, s.h, col,
                      alpha_body=alpha, border=border, border_w=bw,
                      highlight=is_drag)

            # label
            tc = _shade(col, 0.55)
            (lw, _), _ = cv2.getTextSize(s.label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.42, 1)
            self._put(frame, s.label,
                      int(s.px) - lw // 2, int(s.py) + 6,
                      0.42, (255, 255, 255), 2)
            self._put(frame, s.label,
                      int(s.px) - lw // 2, int(s.py) + 6,
                      0.42, tc, 1)

    def _draw_status(self, frame):
        self._panel(frame, SST_X, SST_Y, SST_W, SST_H, "Status")
        # Stable bar (4 bars, increasing)
        n_bars = 4
        max_h = 50
        bar_w = 12
        bar_gap = 5
        total_w = n_bars * bar_w + (n_bars - 1) * bar_gap
        ix = SST_X + (SST_W - total_w) // 2
        iy = SST_Y + 36

        ratio = min(1.0, self._stable_count / max(1, self.STABLE))
        active = int(round(ratio * n_bars))
        if active >= n_bars:
            bar_color = (75, 205, 75)
            label = "Stable"
        elif active > 0:
            bar_color = (35, 200, 220)
            label = "Tracking"
        else:
            bar_color = (170, 170, 165)
            label = "..."

        for i in range(n_bars):
            bh = int(max_h * (i + 1) / n_bars)
            x1 = ix + i * (bar_w + bar_gap)
            y2 = iy + max_h
            y1 = y2 - bh
            c = bar_color if i < active else (210, 210, 205)
            cv2.rectangle(frame, (x1, y1), (x1 + bar_w, y2), c, -1)
            cv2.rectangle(frame, (x1, y1), (x1 + bar_w, y2),
                          (105, 105, 100), 1)

        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.44, 1)
        self._put(frame, label, SST_X + (SST_W - tw) // 2,
                  SST_Y + SST_H - 10, 0.44, (70, 70, 65))

    def _draw_progress(self, frame):
        self._panel(frame, PI_X, PI_Y, PI_W, PI_H, "Progress")
        n = len(self.tasks)
        cur = self.task_idx % n
        progress = (cur + (1.0 if self._correct_t > 0 else 0.0)) / n

        # vertical bar
        bw = 22
        bx = PI_X + PI_W // 2 - bw // 2
        by = PI_Y + 42
        bh = PI_H - 90

        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh),
                      (215, 215, 210), -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh),
                      (165, 165, 160), 1)

        fill_h = int(bh * progress)
        if fill_h > 0:
            grad_col = (75, 205, 75)
            cv2.rectangle(frame,
                          (bx, by + bh - fill_h),
                          (bx + bw, by + bh),
                          grad_col, -1)

        # tick marks for each stage
        for i in range(1, n):
            ty = by + bh - int(bh * i / n)
            cv2.line(frame, (bx - 4, ty), (bx + bw + 4, ty),
                     (130, 130, 125), 1)

        # percent
        pct = f"{int(progress * 100)}%"
        (tw, _), _ = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        self._put(frame, pct, PI_X + (PI_W - tw) // 2,
                  PI_Y + PI_H - 14, 0.50, (70, 70, 65))

        # current stage label
        st = f"Stage {cur + 1}/{n}"
        (tw2, _), _ = cv2.getTextSize(st, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        self._put(frame, st, PI_X + (PI_W - tw2) // 2,
                  PI_Y + 56, 0.40, (95, 95, 90))

    def _draw_pointer(self, frame):
        if self.r_ptr is None or self._correct_t > 0:
            return
        px, py = int(self.r_ptr[0]), int(self.r_ptr[1])
        mc = self.MODE_BGR.get(self.mode, (145, 145, 145))
        if self.r_pinch:
            # solid grab cursor — clearly committed
            cv2.circle(frame, (px, py),  6, mc, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 13, mc,  2, cv2.LINE_AA)
        else:
            # crosshair-style aim cursor: outer ring + small center dot
            cv2.circle(frame, (px, py), 11, mc, 2, cv2.LINE_AA)
            cv2.circle(frame, (px, py),  2, mc, -1, cv2.LINE_AA)
            # pre-grab affordance: extra white halo when about to grab
            if self.hovered is not None:
                cv2.circle(frame, (px, py), 16, (255, 255, 255), 1,
                           cv2.LINE_AA)

    def _draw_correct_overlay(self, frame):
        if self._correct_t <= 0:
            return
        elapsed = time.time() - self._correct_t
        alpha   = min(0.70, elapsed * 1.4)
        self._fill(frame, AW_X, AW_Y, AW_X + AW_W, AW_Y + AW_H,
                   (55, 200, 75), alpha * 0.30)
        cv2.rectangle(frame, (self.dz_x - 4, self.dz_y - 4),
                      (self.dz_x + self.dz_w + 4, self.dz_y + self.dz_h + 4),
                      (55, 200, 75), 3)
        txt = "CORRECT!"
        scale, thick = 2.0, 4
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,
                                       scale, thick)
        tx = AW_X + (AW_W - tw) // 2
        ty = self.dz_y - 24
        self._put(frame, txt, tx + 3, ty + 3, scale, (18, 75, 18), thick)
        self._put(frame, txt, tx, ty, scale, (75, 235, 75), thick)
        remain = max(0.0, self.CORRECT_S - elapsed)
        nxt = f"Next stage in {remain:.1f}s ..."
        (nw, _), _ = cv2.getTextSize(nxt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        self._put(frame, nxt, AW_X + (AW_W - nw) // 2,
                  ty + 38, 0.55, (170, 230, 170))

    # ── main draw ────────────────────────────

    def draw(self, frame: np.ndarray) -> None:
        frame[:] = BG

        self._draw_stage_sidebar(frame)
        self._draw_instructor(frame)
        self._draw_reference(frame)
        self._draw_self_view(frame)
        self._draw_instruction_panel(frame)
        self._draw_status(frame)
        self._draw_progress(frame)

        # workspaces (panels) before shapes so shapes float above
        self._draw_action_workspace(frame)
        self._draw_task_workspace(frame)

        # shapes
        self._draw_shapes(frame)

        # feedback bar above main content
        self._draw_feedback_bar(frame)

        # pointer + overlays
        self._draw_pointer(frame)
        self._draw_correct_overlay(frame)

    def draw_landmarks(self, frame, result):
        # draw small landmarks inside self-view region only
        sh, sw = self._raw_frame.shape[:2] if self._raw_frame is not None else (1, 1)
        in_x, in_y = SV_X + 8, SV_Y + 36
        in_w, in_h = SV_W - 16, SV_H - 44
        scale = min(in_w / sw, in_h / sh)
        rw, rh = int(sw * scale), int(sh * scale)
        ox = in_x + (in_w - rw) // 2
        oy = in_y + (in_h - rh) // 2

        for lm_list, hd_list in zip(result.hand_landmarks, result.handedness):
            is_left_user = hd_list[0].category_name == "Right"
            col = (100, 180, 255) if is_left_user else (80, 200, 120)
            pts = [(ox + int(lm.x * rw), oy + int(lm.y * rh))
                   for lm in lm_list]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], col, 1, cv2.LINE_AA)
            for i, pt in enumerate(pts):
                cv2.circle(frame, pt, 3 if i == 0 else 2, col, -1, cv2.LINE_AA)

# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    ensure_model()

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.70,
        min_hand_presence_confidence=0.70,
        min_tracking_confidence=0.60,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    WIN    = "Gesture Workspace v5"
    app    = App()
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Gesture Workspace v5  |  'q' quit  |  'f' fullscreen  |  "
          "'1'-'9' jump  |  'u' mark last action unintentional")

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, raw = cap.read()
            if not ok:
                break

            raw    = cv2.flip(raw, 1)
            rgb    = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            ts_ms  = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            app.update(result, raw)
            app.draw(canvas)
            app.draw_landmarks(canvas, result)

            cv2.imshow(WIN, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                cur = cv2.getWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN)
                nxt = cv2.WINDOW_NORMAL if cur == cv2.WINDOW_FULLSCREEN \
                      else cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, nxt)
            elif key == ord("u"):
                if app.metrics.mark_last_unintentional():
                    print("[metrics] last action marked unintentional")
                else:
                    print("[metrics] no recent action to mark")
            elif ord("1") <= key <= ord("9"):
                app.jump_to_task(key - ord("1"))

    cap.release()
    cv2.destroyAllWindows()
    app.metrics.write_to_disk()


if __name__ == "__main__":
    main()
