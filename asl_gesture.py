#!/usr/bin/env python3
"""
ASL Gesture Shortcuts — Webcam Demo  (MediaPipe Tasks API)
==========================================================
  C  ->  pinch object = copy / pinch empty = paste
  F  ->  pinch object = highlight same type
  D  ->  pinch object = delete
  Z  ->  pinch = undo last delete (silent if stack empty)
  open palm -> reset mode

Install:  pip install opencv-python mediapipe numpy
Run:      python asl_gesture_shortcuts.py
"""

import copy, os, time, urllib.request
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")

# ─────────────────────────────────────────────
#  Scene
# ─────────────────────────────────────────────

@dataclass
class SceneObj:
    id:    int
    cx:    float
    cy:    float
    w:     float
    h:     float
    label: str
    kind:  str
    color: tuple   # BGR
    alive: bool = True

INITIAL_SCENE = [
    SceneObj(0, 0.18, 0.30, 0.14, 0.14, "Cube A",  "cube",   (200, 130,  60)),
    SceneObj(1, 0.48, 0.24, 0.16, 0.13, "Sphere",  "sphere", ( 70, 190,  80)),
    SceneObj(2, 0.76, 0.30, 0.14, 0.14, "Cube B",  "cube",   (200, 130,  60)),
    SceneObj(3, 0.28, 0.68, 0.14, 0.12, "Cone",    "cone",   ( 60,  80, 210)),
    SceneObj(4, 0.62, 0.66, 0.14, 0.14, "Cube C",  "cube",   (200, 130,  60)),
]

# ─────────────────────────────────────────────
#  Gesture Classifier
# ─────────────────────────────────────────────

def _ext(lm, tip: int, pip: int, thresh=0.025) -> bool:
    return lm[tip].y < lm[pip].y - thresh

def classify_left(lm) -> str:
    idx  = _ext(lm,  8,  6)
    mid  = _ext(lm, 12, 10)
    ring = _ext(lm, 16, 14)
    pnk  = _ext(lm, 20, 18)
    d_ti = float(np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y))

    # open palm
    if idx and mid and ring and pnk:
        return "open"

    # Z: index only up
    if idx and not mid and not ring and not pnk:
        return "Z"

    # F: OK sign — thumb+index circle, others extended
    if d_ti < 0.07 and mid and ring and pnk:
        return "F"

    # C: all fingers curled, tips above palm center
    if not idx and not mid and not ring and not pnk:
        avg_y  = (lm[8].y + lm[12].y + lm[16].y + lm[20].y) / 4
        palm_y = (lm[0].y + lm[9].y) / 2
        if avg_y < palm_y - 0.01:
            return "C"

    # D: thumbs down — all fingers folded, thumb tip pointing downward
    # thumb tip (lm[4]) is below thumb MCP (lm[2]) by a clear margin
    if not idx and not mid and not ring and not pnk:
        thumb_down = lm[4].y > lm[2].y + 0.04
        if thumb_down:
            return "D"

    return "unknown"

def pinch_dist(lm) -> float:
    return float(np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y))

# ─────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

class App:
    STABLE  = 10
    PINCH_T = 0.055
    NOTIF_S = 1.5
    MODE_BGR = {
        "C":    ( 60, 130, 200),
        "F":    (200,  80, 180),
        "D":    ( 50,  50, 210),
        "Z":    ( 80, 180, 130),
        "open": (140, 140, 140),
    }

    def __init__(self):
        self.objects        = [copy.copy(o) for o in INITIAL_SCENE]
        self._next_id       = len(self.objects)
        self.clipboard:     Optional[SceneObj] = None
        self._delete_stack: list[SceneObj] = []
        self.mode           = "open"
        self._buf:          list[str] = []
        self.highlighted:   set[int] = set()
        self.hovered:       Optional[SceneObj] = None
        self.r_ptr:         Optional[tuple] = None
        self._r_ptr_stable: Optional[tuple] = None  # pinch 직전 위치 고정용
        self.r_pinch        = False
        self._prev_pin      = False
        self._notif         = ""
        self._notif_t       = 0.0

    def _push(self, g: str):
        self._buf.append(g)
        if len(self._buf) > self.STABLE:
            self._buf.pop(0)
        if len(self._buf) == self.STABLE and len(set(self._buf)) == 1:
            new = self._buf[0]
            if new in ("C", "F", "D", "Z", "open"):
                if new == "open" and self.mode != "open":
                    self.highlighted.clear()
                self.mode = new

    def _hit(self, nx, ny) -> Optional[SceneObj]:
        for o in self.objects:
            if o.alive and abs(nx - o.cx) < o.w/2 and abs(ny - o.cy) < o.h/2:
                return o
        return None

    def _notify(self, msg: str):
        self._notif   = msg
        self._notif_t = time.time()

    def _execute(self):
        obj = self.hovered
        if self.mode == "C":
            if obj:
                self.clipboard = copy.copy(obj)
                self._notify(f"Copied: {obj.label}")
            elif self.clipboard and self.r_ptr:
                new       = copy.copy(self.clipboard)
                new.id    = self._next_id
                new.cx    = self.r_ptr[0]
                new.cy    = self.r_ptr[1]
                new.label = self.clipboard.label + "'"
                new.alive = True
                self.objects.append(new)
                self._next_id += 1
                self._notify(f"Pasted: {new.label}")
            else:
                self._notify("Clipboard empty")
        elif self.mode == "F":
            if obj:
                self.highlighted = {o.id for o in self.objects if o.alive and o.kind == obj.kind}
                self._notify(f"{obj.kind}: {len(self.highlighted)} highlighted")
        elif self.mode == "D":
            if obj:
                obj.alive = False
                self._delete_stack.append(obj)
                self.highlighted.discard(obj.id)
                self._notify(f"Deleted: {obj.label}")
        elif self.mode == "Z":
            if self._delete_stack:
                restored       = self._delete_stack.pop()
                restored.alive = True
                self._notify(f"Restored: {restored.label}")

    def update(self, result):
        left_g = "unknown"
        r_lm   = None
        for lm_list, hd_list in zip(result.hand_landmarks, result.handedness):
            if hd_list[0].category_name == "Right":   # flipped -> user's left hand
                left_g = classify_left(lm_list)
            else:
                r_lm = lm_list

        self._push(left_g)

        if r_lm:
            new_pinch = pinch_dist(r_lm) < self.PINCH_T

            # 포인터 표시: 검지 끝(lm[8])
            self.r_ptr = (r_lm[8].x, r_lm[8].y)

            if new_pinch:
                # 선택 판정: 엄지+검지 끝의 중간 지점 (실제로 손가락이 닿는 곳)
                mx = (r_lm[4].x + r_lm[8].x) / 2
                my = (r_lm[4].y + r_lm[8].y) / 2
                self._r_ptr_stable = (mx, my)
            else:
                self._r_ptr_stable = self.r_ptr

            self.r_pinch = new_pinch
            self.hovered = self._hit(*self._r_ptr_stable)
        else:
            self.r_ptr = self.hovered = self._r_ptr_stable = None
            self.r_pinch = False

        if self.r_pinch and not self._prev_pin:
            self._execute()
        self._prev_pin = self.r_pinch

    def draw_landmarks(self, frame: np.ndarray, result) -> None:
        H, W = frame.shape[:2]
        for lm_list, hd_list in zip(result.hand_landmarks, result.handedness):
            is_left_user = hd_list[0].category_name == "Right"
            col = (100, 180, 255) if is_left_user else (80, 200, 120)
            pts = [(int(lm.x * W), int(lm.y * H)) for lm in lm_list]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], col, 1, cv2.LINE_AA)
            for i, pt in enumerate(pts):
                cv2.circle(frame, pt, 4 if i == 0 else 2, col, -1, cv2.LINE_AA)

    def draw(self, frame: np.ndarray) -> None:
        H, W = frame.shape[:2]
        put = lambda text, x, y, scale, color, thickness=1: cv2.putText(
            frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
        )

        # objects
        for o in self.objects:
            if not o.alive: continue
            cx, cy = int(o.cx * W), int(o.cy * H)
            hw, hh = int(o.w * W / 2), int(o.h * H / 2)
            x1, y1, x2, y2 = cx-hw, cy-hh, cx+hw, cy+hh

            is_hov = bool(self.hovered and self.hovered.id == o.id)
            is_hl  = o.id in self.highlighted

            a  = 0.45 if is_hov else (0.25 if is_hl else 0.12)
            ov = frame.copy()
            cv2.rectangle(ov, (x1,y1), (x2,y2), o.color, -1)
            cv2.addWeighted(ov, a, frame, 1-a, 0, frame)

            bc = o.color if (is_hov or is_hl) else (100, 105, 115)
            cv2.rectangle(frame, (x1,y1), (x2,y2), bc, 2 if (is_hov or is_hl) else 1)
            tc = o.color if (is_hov or is_hl) else (190, 195, 205)
            put(o.label, cx - len(o.label)*4, cy + 5, 0.45, tc)

        # pointer
        if self.r_ptr:
            px, py = int(self.r_ptr[0]*W), int(self.r_ptr[1]*H)
            mc = self.MODE_BGR.get(self.mode, (160,160,160))
            cv2.circle(frame, (px,py), 6 if self.r_pinch else 13, mc, 2, cv2.LINE_AA)
            if self.r_pinch:
                cv2.circle(frame, (px,py), 4, mc, -1, cv2.LINE_AA)

        # mode badge
        badge = {
            "C":    "C  |  Copy / Paste",
            "F":    "F  |  Find & Highlight",
            "D":    "D  |  Delete",
            "Z":    f"Z  |  Undo  (stack: {len(self._delete_stack)})",
            "open": "   Default",
        }.get(self.mode, "Waiting...")
        mc = self.MODE_BGR.get(self.mode, (140,140,140))
        cv2.rectangle(frame, (10,10), (310,42), (15,18,25), -1)
        cv2.rectangle(frame, (10,10), (310,42), mc, 1)
        put(badge, 18, 33, 0.58, mc)

        if self.clipboard:
            put(f"Clipboard: {self.clipboard.label}", 10, 58, 0.42, (140,145,175))

        # notification
        if self._notif and time.time() - self._notif_t < self.NOTIF_S:
            nw  = len(self._notif) * 11 + 40
            nx1 = W//2 - nw//2
            cv2.rectangle(frame, (nx1, H//2-22), (nx1+nw, H//2+18), (15,18,25), -1)
            cv2.rectangle(frame, (nx1, H//2-22), (nx1+nw, H//2+18), (70,75,95), 1)
            put(self._notif, nx1+14, H//2+6, 0.58, (220,225,235))

        # legend
        legend = [
            "C + object  -> copy",
            "C + empty   -> paste",
            "F + object  -> highlight",
            "thumbs down + object -> delete",
            "Z (index up) -> undo delete",
            "open palm   -> reset mode",
        ]
        for i, line in enumerate(legend):
            put(line, W-230, 24 + i*22, 0.38, (120,125,145))

# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    app = App()
    print("ASL Gesture Shortcuts running  |  press 'q' to quit")

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts_ms  = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            app.update(result)
            app.draw_landmarks(frame, result)
            app.draw(frame)

            cv2.imshow("ASL Gesture Shortcuts", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()