#!/usr/bin/env python3
"""
Gesture Workspace  (MediaPipe Tasks API)
=========================================
Layout:
  Top-left    : gesture guide  (fist to toggle on/off)
  Bottom-left : answer key
  Top-right   : webcam self-view
  Center      : play area  +  drop zone with slots

Left-hand gestures:
  fist        -> toggle gesture guide
  open palm   -> drag mode  (default)
  C           -> copy / paste
  F           -> find & highlight same type
  thumbs down -> delete mode
  Z (index)   -> undo last delete

Install:  pip install opencv-python mediapipe numpy
Run:      python asl_gesture_shortcuts.py
"""

import copy, os, random, time, urllib.request
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
#  Layout  (1280 × 720)
# ──────────────────────────────────────────────

W, H = 1280, 720

LS_W = 305          # left strip width

# Play area — full height, right of left strip
PA_X, PA_Y = LS_W,  0
PA_W, PA_H = W - LS_W, H

# Drop zone — center of play area, large
DZ_W, DZ_H = 700, 280
DZ_X = PA_X + (PA_W - DZ_W) // 2
DZ_Y = PA_Y + PA_H - DZ_H - 20

# Slot grid inside drop zone
SLOT_COLS  = 4
SLOT_ROWS  = 2
SLOT_PAD   = 12
SLOT_W     = (DZ_W - SLOT_PAD * (SLOT_COLS + 1)) // SLOT_COLS
SLOT_H     = (DZ_H - SLOT_PAD * (SLOT_ROWS + 1) - 32) // SLOT_ROWS  # 32 for label

# Answer key panel — bottom-left
AK_X, AK_Y = 8,  H - 270
AK_W, AK_H = LS_W - 16, 262

# Gesture guide — top-left (toggleable)
GG_X, GG_Y = 8,  8
GG_W, GG_H = LS_W - 16, 200

# Webcam thumbnail — top-right, 16:9
CAM_W = 240
CAM_H = CAM_W * 9 // 16        # = 135
CAM_X = W - CAM_W - 8
CAM_Y = 8

# ──────────────────────────────────────────────
#  Colors (BGR)
# ──────────────────────────────────────────────

BG       = (18, 18, 18)
PANEL_BG = (242, 242, 239)
PANEL_BD = (175, 175, 170)

SHAPE_COLORS = {
    "red":    ( 50, 100, 220),
    "green":  ( 70, 185,  90),
    "blue":   (200, 115,  45),
    "yellow": ( 35, 200, 210),
    "purple": (175,  55, 160),
}

# ──────────────────────────────────────────────
#  Questions — list of slot defs: (col, row, kind)
#  Slots are 0-indexed inside the drop zone grid
# ──────────────────────────────────────────────

QUESTIONS = [
    [(0,0,"green"),  (1,0,"red"),    (2,0,"green")],
    [(0,0,"blue"),   (1,0,"yellow"), (2,0,"green"), (0,1,"red")],
    [(0,0,"red"),    (1,0,"purple"), (2,0,"blue"),  (1,1,"green"), (3,0,"yellow")],
    [(0,0,"yellow"), (1,0,"yellow"), (2,0,"red"),   (3,0,"purple")],
    [(0,0,"green"),  (1,0,"blue"),   (2,0,"red"),   (0,1,"yellow"), (1,1,"purple")],
]

# Scatter positions — relative to PA_X/PA_Y, kept away from top-right (webcam)
SCATTER_TEMPLATES = [
    (160,  70), (320,  55), (480,  75), (620,  60),
    (160, 170), (310, 165), (460, 155), (600, 170),
    (230, 270), (420, 260),
]

SH_W, SH_H = 110, 68

# ──────────────────────────────────────────────
#  Slot helpers
# ──────────────────────────────────────────────

def slot_rect(col, row):
    x = DZ_X + SLOT_PAD + col * (SLOT_W + SLOT_PAD)
    y = DZ_Y + 36 + row * (SLOT_H + SLOT_PAD)
    return x, y, x + SLOT_W, y + SLOT_H

def slot_center(col, row):
    x1, y1, x2, y2 = slot_rect(col, row)
    return (x1 + x2) // 2, (y1 + y2) // 2

def nearest_slot(px, py, question):
    """Return (col,row) if (px,py) is inside a valid slot, else None."""
    for col, row, _ in question:
        x1, y1, x2, y2 = slot_rect(col, row)
        if x1 <= px <= x2 and y1 <= py <= y2:
            return (col, row)
    return None

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
    slot:  object = None   # (col, row) when snapped

# ──────────────────────────────────────────────
#  Gesture Classifier
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
    # ── extension: tip above PIP ──────────────────────────────
    idx_ext  = _ext(lm,  8,  6)
    mid_ext  = _ext(lm, 12, 10)
    ring_ext = _ext(lm, 16, 14)
    pnk_ext  = _ext(lm, 20, 18)

    # ── full curl: tip below knuckle (MCP) ────────────────────
    # y increases downward, so tip.y > mcp.y means tip curled under
    idx_curl  = lm[8].y  > lm[5].y  + 0.01
    mid_curl  = lm[12].y > lm[9].y  + 0.01
    ring_curl = lm[16].y > lm[13].y + 0.01
    pnk_curl  = lm[20].y > lm[17].y + 0.01

    d_ti = float(np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y))

    # open: all four fingers extended
    if idx_ext and mid_ext and ring_ext and pnk_ext:
        return "open"

    # fist: all four fingertips curled below their MCP
    if idx_curl and mid_curl and ring_curl and pnk_curl:
        return "fist"

    # Z: index only, straight up
    if idx_ext and not mid_ext and not ring_ext and not pnk_ext:
        return "Z"

    # F: OK sign — thumb+index circle, mid/ring/pinky extended
    if d_ti < 0.07 and mid_ext and ring_ext and pnk_ext:
        return "F"

    # X: 나머지 3 손가락이 MCP 아래로 완전히 말림
    others_fully_curled = mid_curl and ring_curl and pnk_curl
    if not mid_ext and not ring_ext and not pnk_ext and others_fully_curled:
        return "X"

    # C: 나머지 3 손가락이 살짝만 구부러짐 (완전히 말리지 않음)
    all_bent = (not idx_ext and not mid_ext and not ring_ext and not pnk_ext)
    not_fist = not (idx_curl and mid_curl and ring_curl and pnk_curl)
    if all_bent and not_fist and not others_fully_curled:
        return "C"

    return "unknown"

def pinch_dist(lm) -> float:
    return float(np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y))

# ──────────────────────────────────────────────
#  App
# ──────────────────────────────────────────────

class App:
    STABLE    = 10
    PINCH_T   = 0.055
    NOTIF_S   = 1.4
    CORRECT_S = 2.5
    MODE_BGR  = {
        "C":    (180, 120,  50),
        "F":    (175,  55, 195),
        "X":    ( 40,  40, 210),
        "Z":    ( 95, 190,  75),
        "open": (145, 145, 145),
    }

    def __init__(self):
        self.q_idx          = 0
        self.shapes:        list[Shape] = []
        self.clipboard:     Optional[Shape] = None
        self._delete_stack: list[Shape] = []
        self.mode           = "open"
        self._buf:          list[str] = []
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
        self.show_guide     = False
        self._prev_fist     = False
        self._raw_frame     = None
        self._next_id       = 0
        self._correct_t     = 0.0

        # gesture guide image (asl_ui.png 을 스크립트와 같은 폴더에 저장)
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asl_ui.png")
        raw_img  = cv2.imread(img_path)
        if raw_img is not None:
            ih, iw = raw_img.shape[:2]
            # fit inside GG_W x GG_H while keeping aspect ratio
            scale   = min(GG_W / iw, GG_H / ih)
            rw, rh  = int(iw * scale), int(ih * scale)
            resized = cv2.resize(raw_img, (rw, rh))
            # center on black canvas
            canvas  = np.zeros((GG_H, GG_W, 3), dtype=np.uint8)
            ox = (GG_W - rw) // 2
            oy = (GG_H - rh) // 2
            canvas[oy:oy+rh, ox:ox+rw] = resized
            self._guide_img = canvas
            print(f"Guide image loaded: {img_path}")
        else:
            self._guide_img = None
            print(f"[info] Place asl_ui.png next to the script to show gesture guide image.")

        self._load_question()

    # ── question management ──────────────────

    def _load_question(self):
        self.shapes = []
        self._delete_stack = []
        self.clipboard = None
        self.highlighted.clear()

        q            = QUESTIONS[self.q_idx % len(QUESTIONS)]
        answer_kinds = [kind for _, _, kind in q]

        pool = list(answer_kinds)
        while len(pool) < len(SCATTER_TEMPLATES):
            pool.append(random.choice(list(SHAPE_COLORS.keys())))
        random.shuffle(pool)

        labels = "ABCDEFGHIJKLMNOP"
        for i, (rx, ry) in enumerate(SCATTER_TEMPLATES):
            jx = rx + random.randint(-18, 18)
            jy = ry + random.randint(-12, 12)
            self.shapes.append(Shape(
                id    = self._next_id,
                px    = float(PA_X + jx),
                py    = float(PA_Y + jy),
                w     = SH_W, h = SH_H,
                label = labels[i],
                kind  = pool[i],
            ))
            self._next_id += 1

    def _check_answer(self) -> bool:
        """All slots in the question filled with the correct kind."""
        q = QUESTIONS[self.q_idx % len(QUESTIONS)]
        for col, row, expected in q:
            occ = next((s for s in self.shapes
                        if s.alive and s.slot == (col, row)), None)
            if occ is None or occ.kind != expected:
                return False
        return True

    def _next_question(self):
        self.q_idx     += 1
        self._correct_t = 0.0
        self._load_question()
        self._notify(f"Question {self.q_idx + 1}!")

    # ── helpers ──────────────────────────────

    def _push(self, g: str):
        self._buf.append(g)
        if len(self._buf) > self.STABLE:
            self._buf.pop(0)
        if len(self._buf) == self.STABLE and len(set(self._buf)) == 1:
            new = self._buf[0]
            if new == "fist":
                if not self._prev_fist:
                    self.show_guide = not self.show_guide
                self._prev_fist = True
                return
            self._prev_fist = False
            if new in ("C", "F", "X", "Z", "open"):
                if new == "open" and self.mode != "open":
                    self.highlighted.clear()
                self.mode = new

    def _hit(self, px, py) -> Optional[Shape]:
        for s in reversed(self.shapes):
            if s.alive and abs(px - s.px) < s.w/2 and abs(py - s.py) < s.h/2:
                return s
        return None

    def _in_dz(self, px, py) -> bool:
        return DZ_X < px < DZ_X + DZ_W and DZ_Y < py < DZ_Y + DZ_H

    def _notify(self, msg: str):
        self._notif   = msg
        self._notif_t = time.time()

    # ── execute pinch ─────────────────────────

    def _execute(self):
        obj = self.hovered
        if self.mode == "C":
            if obj:
                self.clipboard = copy.copy(obj)
                self._notify(f"Copied: {obj.label}")
            elif self.clipboard and self._pinch_pt:
                new       = copy.copy(self.clipboard)
                new.id    = self._next_id
                new.px    = self._pinch_pt[0]
                new.py    = self._pinch_pt[1]
                new.label = self.clipboard.label + "'"
                new.slot  = None
                new.alive = True
                self.shapes.append(new)
                self._next_id += 1
                self._notify(f"Pasted: {new.label}")
            else:
                self._notify("Clipboard empty")
        elif self.mode == "F":
            if obj:
                self.highlighted = {s.id for s in self.shapes
                                    if s.alive and s.kind == obj.kind}
                self._notify(f"{obj.kind}: {len(self.highlighted)} highlighted")
        elif self.mode == "X":
            if obj:
                obj.alive = False
                obj.slot  = None
                self._delete_stack.append(obj)
                self.highlighted.discard(obj.id)
                self._notify(f"Deleted: {obj.label}")
        elif self.mode == "Z":
            if self._delete_stack:
                r       = self._delete_stack.pop()
                r.alive = True
                self._notify(f"Restored: {r.label}")

    # ── update ───────────────────────────────

    def update(self, result, raw_frame):
        self._raw_frame = raw_frame

        if self._correct_t > 0:
            if time.time() - self._correct_t > self.CORRECT_S:
                self._next_question()
            return

        left_g = "unknown"
        r_lm   = None
        for lm_list, hd_list in zip(result.hand_landmarks, result.handedness):
            if hd_list[0].category_name == "Right":   # flip → user's left hand
                left_g = classify_left(lm_list)
            else:
                r_lm = lm_list

        self._push(left_g)

        if r_lm:
            ix = r_lm[8].x * W
            iy = r_lm[8].y * H
            self.r_ptr = (ix, iy)

            mx = (r_lm[4].x + r_lm[8].x) / 2 * W
            my = (r_lm[4].y + r_lm[8].y) / 2 * H
            new_pinch = pinch_dist(r_lm) < self.PINCH_T
            self._pinch_pt = (mx, my) if new_pinch else (ix, iy)

            if self.mode == "open":
                if new_pinch:
                    if not self._prev_pin:
                        hit = self._hit(mx, my)
                        if hit:
                            self.dragging      = hit
                            hit.slot           = None
                            self._drag_offset  = (hit.px - mx, hit.py - my)
                    if self.dragging:
                        self.dragging.px = mx + self._drag_offset[0]
                        self.dragging.py = my + self._drag_offset[1]
                else:
                    if self.dragging:
                        q = QUESTIONS[self.q_idx % len(QUESTIONS)]
                        s = nearest_slot(self.dragging.px, self.dragging.py, q)
                        if s is not None:
                            # evict previous occupant
                            for other in self.shapes:
                                if other.id != self.dragging.id and other.slot == s:
                                    other.slot = None
                            self.dragging.slot = s
                            cx, cy = slot_center(*s)
                            self.dragging.px = float(cx)
                            self.dragging.py = float(cy)
                        else:
                            self.dragging.slot = None
                        if self._check_answer():
                            self._correct_t = time.time()
                    self.dragging = None

            self.r_pinch = new_pinch
            self.hovered = self._hit(mx, my)
        else:
            self.r_ptr = self.hovered = self._pinch_pt = None
            self.r_pinch  = False
            self.dragging = None

        if self.r_pinch and not self._prev_pin and self.mode != "open":
            self._execute()
        self._prev_pin = self.r_pinch

    # ── draw helpers ─────────────────────────

    def _put(self, frame, text, x, y, scale, color, thick=1):
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def _fill(self, frame, x1, y1, x2, y2, color, alpha):
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

    # ── draw ─────────────────────────────────

    def draw(self, frame: np.ndarray) -> None:
        frame[:] = BG

        # ── play area background ──────────────
        self._fill(frame, PA_X, PA_Y, PA_X+PA_W, PA_Y+PA_H, PANEL_BG, 0.90)
        cv2.rectangle(frame, (PA_X,PA_Y),(PA_X+PA_W,PA_Y+PA_H), PANEL_BD, 1)

        # ── drop zone ────────────────────────
        dz_border = (75, 205, 75) if self._correct_t > 0 else (145, 145, 135)
        self._fill(frame, DZ_X, DZ_Y, DZ_X+DZ_W, DZ_Y+DZ_H, (195,195,190), 0.60)
        cv2.rectangle(frame,(DZ_X,DZ_Y),(DZ_X+DZ_W,DZ_Y+DZ_H), dz_border, 2)
        self._put(frame,"DROP ZONE", DZ_X+8, DZ_Y+24, 0.46,(115,115,110))

        # ── slots ────────────────────────────
        q = QUESTIONS[self.q_idx % len(QUESTIONS)]
        for col, row, expected in q:
            x1,y1,x2,y2 = slot_rect(col, row)
            occ = next((s for s in self.shapes
                        if s.alive and s.slot == (col, row)), None)
            correct = occ is not None and occ.kind == expected
            exp_col = SHAPE_COLORS[expected]

            # faint color hint in empty/wrong slots
            if not correct:
                self._fill(frame, x1, y1, x2, y2, exp_col, 0.12)
            slot_bc = (75,205,75) if correct else (150,150,145)
            cv2.rectangle(frame,(x1,y1),(x2,y2), slot_bc, 2 if correct else 1)
            # kind hint letter
            if not correct:
                self._put(frame, expected[0].upper(),
                          x1+5, y2-6, 0.36, exp_col)

        # ── shapes ───────────────────────────
        for s in self.shapes:
            if not s.alive: continue
            x1 = int(s.px - s.w/2);  y1 = int(s.py - s.h/2)
            x2 = int(s.px + s.w/2);  y2 = int(s.py + s.h/2)
            col = SHAPE_COLORS[s.kind]

            is_hov  = bool(self.hovered  and self.hovered.id  == s.id)
            is_drag = bool(self.dragging and self.dragging.id == s.id)
            is_hl   = s.id in self.highlighted
            in_slot = s.slot is not None

            alpha = (0.75 if is_drag else
                     0.60 if is_hov  else
                     0.50 if in_slot else
                     0.40 if is_hl   else 0.28)
            self._fill(frame, x1, y1, x2, y2, col, alpha)

            bw = 3 if is_drag else (2 if (is_hov or is_hl or in_slot) else 1)
            bc = col if (is_drag or is_hov or is_hl or in_slot) \
                     else tuple(int(c*0.55) for c in col)
            cv2.rectangle(frame,(x1,y1),(x2,y2), bc, bw)

            if is_drag:
                m = 9
                for ex, ey in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                    sx = 1 if ex==x1 else -1
                    sy = 1 if ey==y1 else -1
                    cv2.line(frame,(ex,ey),(ex+sx*m,ey),col,2)
                    cv2.line(frame,(ex,ey),(ex,ey+sy*m),col,2)

            tc = col if (is_drag or is_hov or is_hl or in_slot) else (80,80,80)
            self._put(frame, s.label,
                      int(s.px)-len(s.label)*4, int(s.py)+5, 0.42, tc)

        # ── answer key — bottom-left ──────────
        self._fill(frame, AK_X, AK_Y, AK_X+AK_W, AK_Y+AK_H, (245,245,242), 0.96)
        cv2.rectangle(frame,(AK_X,AK_Y),(AK_X+AK_W,AK_Y+AK_H), PANEL_BD, 1)
        self._put(frame, f"Q{self.q_idx+1}  Answer Key",
                  AK_X+10, AK_Y+22, 0.48,(70,70,65))
        cv2.line(frame,(AK_X+8,AK_Y+30),(AK_X+AK_W-8,AK_Y+30), PANEL_BD, 1)

        # mini slot grid matching drop zone layout
        MINI_W, MINI_H, MINI_PAD = 56, 38, 7
        for col, row, kind in q:
            mx1 = AK_X + 10 + col*(MINI_W+MINI_PAD)
            my1 = AK_Y + 40 + row*(MINI_H+MINI_PAD)
            mx2, my2 = mx1+MINI_W, my1+MINI_H
            col_bgr = SHAPE_COLORS[kind]
            occ     = next((s for s in self.shapes
                            if s.alive and s.slot == (col, row)), None)
            correct = occ is not None and occ.kind == kind
            self._fill(frame, mx1, my1, mx2, my2, col_bgr,
                       0.65 if correct else 0.32)
            cv2.rectangle(frame,(mx1,my1),(mx2,my2),
                          (75,205,75) if correct else col_bgr,
                          2 if correct else 1)
            if correct:
                self._put(frame,"v", mx1+MINI_W//2-5, my2-5, 0.42,(40,160,40),2)
            else:
                self._put(frame, kind[:3], mx1+4, my2-6, 0.30, col_bgr)

        # ── webcam — top-right ────────────────
        if self._raw_frame is not None:
            src_h, src_w = self._raw_frame.shape[:2]
            th = int(CAM_W * src_h / src_w)
            thumb = cv2.resize(self._raw_frame, (CAM_W, th))
            if th >= CAM_H:
                thumb = thumb[(th-CAM_H)//2:(th-CAM_H)//2+CAM_H, :]
            else:
                pad = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
                pad[(CAM_H-th)//2:(CAM_H-th)//2+th, :] = thumb
                thumb = pad
            frame[CAM_Y:CAM_Y+CAM_H, CAM_X:CAM_X+CAM_W] = thumb
            cv2.rectangle(frame,(CAM_X,CAM_Y),(CAM_X+CAM_W,CAM_Y+CAM_H),(145,145,140),1)
            self._put(frame,"see myself",
                      CAM_X + CAM_W//2 - 45, CAM_Y - 6, 0.38,(125,125,120))

        # ── gesture guide — top-left ──────────
        if self.show_guide:
            if self._guide_img is not None:
                # asl_ui.png 이미지를 그대로 표시
                frame[GG_Y:GG_Y+GG_H, GG_X:GG_X+GG_W] = self._guide_img
                cv2.rectangle(frame,(GG_X,GG_Y),(GG_X+GG_W,GG_Y+GG_H), PANEL_BD, 1)
                self._put(frame,"fist to hide",
                          GG_X+GG_W//2-45, GG_Y+GG_H+16, 0.35,(90,90,85))
            else:
                # fallback: 텍스트 가이드
                self._fill(frame, GG_X, GG_Y, GG_X+GG_W, GG_Y+GG_H,
                           (245,245,242), 0.96)
                cv2.rectangle(frame,(GG_X,GG_Y),(GG_X+GG_W,GG_Y+GG_H), PANEL_BD, 1)
                self._put(frame,"Gesture Guide  (fist to hide)",
                          GG_X+8, GG_Y+20, 0.37,(75,75,70))
                cv2.line(frame,(GG_X+8,GG_Y+28),(GG_X+GG_W-8,GG_Y+28), PANEL_BD, 1)
                guide_items = [
                    ("open palm",  "drag & drop to slot"),
                    ("C",          "copy  /  paste"),
                    ("F",          "highlight same type"),
                    ("X  (hook)",  "delete shape"),
                    ("Z",          "undo delete"),
                    ("fist",       "toggle this guide"),
                ]
                gy = GG_Y + 46
                for gname, gdesc in guide_items:
                    gc = (180,120,50) if gname=="C" else \
                         (175,55,195) if gname=="F" else \
                         (40,40,210)  if "thumb" in gname else \
                         (95,190,75)  if gname=="Z" else \
                         (145,145,145)
                    self._put(frame, gname,           GG_X+10, gy, 0.37, gc)
                    self._put(frame, f"-> {gdesc}",   GG_X+98, gy, 0.33, (90,90,85))
                    gy += 24
        else:
            self._put(frame,"fist  ->  guide",
                      GG_X+8, GG_Y+20, 0.37,(90,90,85))

        # ── mode badge ───────────────────────
        badge = {
            "C":    "C  |  Copy / Paste",
            "F":    "F  |  Highlight",
            "X":    "X  |  Delete",
            "Z":    f"Z  |  Undo  ({len(self._delete_stack)})",
            "open": "Drag mode",
        }.get(self.mode, self.mode)
        mc = self.MODE_BGR.get(self.mode, (145,145,145))
        cv2.rectangle(frame,(PA_X+10,10),(PA_X+235,40),(14,17,24),-1)
        cv2.rectangle(frame,(PA_X+10,10),(PA_X+235,40), mc, 1)
        self._put(frame, badge, PA_X+18, 31, 0.48, mc)

        # ── right-hand pointer ───────────────
        if self.r_ptr and self._correct_t == 0:
            px, py = int(self.r_ptr[0]), int(self.r_ptr[1])
            mc = self.MODE_BGR.get(self.mode,(145,145,145))
            cv2.circle(frame,(px,py), 5 if self.r_pinch else 11, mc, 2, cv2.LINE_AA)
            if self.r_pinch:
                cv2.circle(frame,(px,py), 3, mc, -1, cv2.LINE_AA)

        # ── notification ─────────────────────
        if self._notif and time.time()-self._notif_t < self.NOTIF_S:
            nw  = len(self._notif)*10 + 40
            nx1 = W//2 - nw//2
            cv2.rectangle(frame,(nx1,H//2-22),(nx1+nw,H//2+18),(20,22,30),-1)
            cv2.rectangle(frame,(nx1,H//2-22),(nx1+nw,H//2+18),(70,75,95),1)
            self._put(frame, self._notif, nx1+14, H//2+6, 0.52,(220,225,235))

        # ── CORRECT overlay ──────────────────
        if self._correct_t > 0:
            elapsed = time.time() - self._correct_t
            alpha   = min(0.70, elapsed * 1.4)
            self._fill(frame, DZ_X-4, DZ_Y-4, DZ_X+DZ_W+4, DZ_Y+DZ_H+4,
                       (55,200,75), alpha * 0.42)
            cv2.rectangle(frame,(DZ_X-4,DZ_Y-4),(DZ_X+DZ_W+4,DZ_Y+DZ_H+4),
                          (55,200,75), 3)
            txt   = "CORRECT!"
            scale, thick = 2.2, 4
            (tw,th2),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            tx = W//2 - tw//2
            ty = DZ_Y - 30
            self._put(frame, txt, tx+3, ty+3, scale, (18,75,18), thick)
            self._put(frame, txt, tx,   ty,   scale, (75,235,75), thick)
            remain = max(0.0, self.CORRECT_S - elapsed)
            self._put(frame, f"Next in {remain:.1f}s ...",
                      W//2-90, ty+46, 0.55,(170,230,170))

    def draw_landmarks(self, frame, result):
        fh, fw = frame.shape[:2]
        for lm_list, hd_list in zip(result.hand_landmarks, result.handedness):
            is_left = hd_list[0].category_name == "Right"
            col = (100,180,255) if is_left else (80,200,120)
            pts = [(int(lm.x*fw), int(lm.y*fh)) for lm in lm_list]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], col, 1, cv2.LINE_AA)
            for i, pt in enumerate(pts):
                cv2.circle(frame, pt, 4 if i==0 else 2, col, -1, cv2.LINE_AA)

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

    WIN    = "Gesture Workspace"
    app    = App()
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Gesture Workspace  |  'q' quit  |  'f' fullscreen toggle")

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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()