import cv2
import time
from collections import deque
import mediapipe as mp
from typing import Any, Deque, Final, Literal, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

mp_hands: Any = mp.solutions.hands
mp_drawing: Any = mp.solutions.drawing_utils
mp_styles: Any = mp.solutions.drawing_styles

# --- Config ---
FRAME_WIDTH: Final[int] = 1280
FRAME_HEIGHT: Final[int] = 720

# Tuning parameters
MAX_TRAIL: Final[int] = 5
LEFT_SMOOTHING_ALPHA: Final[float] = 0.35   # EMA smoothing for left hand
RIGHT_SMOOTHING_ALPHA: Final[float] = 0.55  # slightly snappier for right hand chop

LEFT_SWIPE_MIN_DIST: Final[int] = 120     # pixels horizontal displacement threshold
LEFT_SWIPE_MAX_Y_DEV: Final[int] = 60     # pixels vertical tolerance while swiping
LEFT_SWIPE_MIN_SPEED: Final[int] = 400    # pixels/second minimal speed for swipe

# Right-hand chop thresholds
CHOP_MIN_SPEED: Final[int] = 450          # pixels/second downward speed for chop
CHOP_MIN_DIST: Final[int] = 80            # minimal downward displacement in pixels
CHOP_MAX_X_DEV: Final[int] = 140          # horizontal tolerance while chopping

# For overlay persistence
OVERLAY_DURATION_SEC: Final[float] = 0.8

# Drawing helpers
ARROW_COLOR: Final[Tuple[int, int, int]] = (0, 255, 255)  # yellow
CIRCLE_COLOR: Final[Tuple[int, int, int]] = (0, 0, 255)   # red


# --- Classes ---

# Helper for exponential moving average smoothing
class EmaPoint:
    def __init__(self, alpha: float) -> None:
        self.alpha: float = alpha
        self.has_value: bool = False
        self.x: float = 0.0
        self.y: float = 0.0

    def update(self, x: float, y: float) -> Tuple[float, float]:
        if not self.has_value:
            self.x, self.y = x, y
            self.has_value = True
        else:
            self.x = self.alpha * x + (1 - self.alpha) * self.x
            self.y = self.alpha * y + (1 - self.alpha) * self.y
        return self.x, self.y

# Track recent positions and timestamps for velocity computation
class MotionTracker:
    def __init__(self, maxlen: int = 6) -> None:
        self.points: Deque[Tuple[float, float, float]] = deque(maxlen=maxlen)

    def add(self, x: float, y: float, t: float) -> None:
        self.points.append((x, y, t))

    def displacement(self) -> Tuple[float, float, float]:
        if len(self.points) < 2:
            return 0.0, 0.0, 0.0
        x0, y0, t0 = self.points[0]
        x1, y1, t1 = self.points[-1]
        dt = max(t1 - t0, 1e-6)
        return x1 - x0, y1 - y0, dt

    def velocity(self) -> Tuple[float, float]:
        dx, dy, dt = self.displacement()
        return dx / dt, dy / dt

# Gesture state to show overlays transiently
class OverlayState:
    def __init__(self) -> None:
        self.kind: Optional[Literal['left', 'right', 'circle']] = None
        self.until: float = 0.0

    def trigger(self, kind: Literal['left', 'right', 'circle'], duration: float = OVERLAY_DURATION_SEC) -> None:
        self.kind = kind
        self.until = time.time() + duration

    def active(self) -> bool:
        return time.time() < self.until

class HandAnchor:
    def __init__(self, alpha: float) -> None:
        self.smoother: EmaPoint = EmaPoint(alpha)
        self.motion: MotionTracker = MotionTracker(MAX_TRAIL)
        self.last_x: int = 0
        self.last_y: int = 0

    def update(self, lm: Any, w: int, h: int, t: float) -> Tuple[int, int]:
        x = int(lm.x * w)
        y = int(lm.y * h)
        sx, sy = self.smoother.update(float(x), float(y))
        self.motion.add(float(sx), float(sy), t)
        self.last_x, self.last_y = int(sx), int(sy)
        return self.last_x, self.last_y

    def recent_displacement(self, n: int = 3) -> Tuple[float, float, float]:
        if len(self.motion.points) < 2:
            return 0.0, 0.0, 0.0
        pts = list(self.motion.points)[-n:]
        x0, y0, t0 = pts[0]
        x1, y1, t1 = pts[-1]
        dt = max(t1 - t0, 1e-6)
        return x1 - x0, y1 - y0, dt

# --- Game ---
class GameState:
    def __init__(self, frame_w: int, frame_h: int) -> None:
        self.w: int = frame_w
        self.h: int = frame_h
        self.cuke_x: int = frame_w // 2
        self.cuke_y: int = int(frame_h * 0.75)
        self.cuke_w: int = int(frame_w * 0.18)
        self.cuke_h: int = int(frame_h * 0.08)
        self.knife_active: bool = False
        self.knife_until: float = 0.0
        self.slice_until: float = 0.0

    def move_cucumber_to(self, x: int) -> None:
        half = self.cuke_w // 2
        self.cuke_x = max(half, min(self.w - half, x))

    def trigger_knife(self, duration: float = 0.18) -> None:
        self.knife_active = True
        self.knife_until = time.time() + duration

    def update(self) -> None:
        t = time.time()
        if self.knife_active and t > self.knife_until:
            self.knife_active = False
        if self.slice_until and t > self.slice_until:
            self.slice_until = 0.0

    def hit_test(self, knife_rect: Tuple[int, int, int, int]) -> bool:
        kx, ky, kw, kh = knife_rect
        cx, cy = self.cuke_x, self.cuke_y
        cw, ch = self.cuke_w, self.cuke_h
        x1, y1, w1, h1 = (cx - cw // 2, cy - ch // 2, cw, ch)
        x2, y2, w2, h2 = knife_rect
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)


# --- Helpers ---

def draw_arrow(img: NDArray[np.uint8], direction: Literal['left', 'right']) -> None:
    h, w = img.shape[:2]
    center_y = h // 2
    length = int(0.25 * w)
    thickness = 10
    if direction == 'left':
        start = (w // 2 + length // 2, center_y)
        end = (w // 2 - length // 2, center_y)
    else:
        start = (w // 2 - length // 2, center_y)
        end = (w // 2 + length // 2, center_y)
    cv2.arrowedLine(img, start, end, ARROW_COLOR, thickness, tipLength=0.35)

def draw_circle(img: NDArray[np.uint8]) -> None:
    h, w = img.shape[:2]
    cv2.circle(img, (w // 2, h // 2), int(min(w, h) * 0.12), CIRCLE_COLOR, thickness=12)

# Handedness helper
def is_left(handedness: Any) -> bool:
    # MediaPipe Hands uses label 'Left' for the person's left hand
    return handedness.classification[0].label == 'Left'

# Gesture detectors
def detect_left_hand_swipe(anchor: HandAnchor, frame_w: int, frame_h: int) -> Optional[Literal['left', 'right']]:
    dx, dy, dt = anchor.motion.displacement()
    if dt <= 0.0:
        return None
    speed_x = abs(dx) / dt
    if abs(dy) > LEFT_SWIPE_MAX_Y_DEV:
        return None
    if speed_x < LEFT_SWIPE_MIN_SPEED:
        return None
    if abs(dx) < LEFT_SWIPE_MIN_DIST:
        return None
    return 'right' if dx > 0 else 'left'

def detect_right_hand_chop(anchor: HandAnchor) -> bool:
    # Favor recent movement to detect quick chops
    rdx, rdy, rdt = anchor.recent_displacement(n=3)
    if rdt <= 0.0:
        return False
    r_speed_y = rdy / rdt

    # Fall back to whole-trail if needed
    dx, dy, dt = anchor.motion.displacement()
    speed_y = dy / dt if dt > 0 else 0.0

    if max(r_speed_y, speed_y) < CHOP_MIN_SPEED:
        return False
    if max(rdy, dy) < CHOP_MIN_DIST:
        return False
    if abs(max(rdx, dx, key=abs)) > CHOP_MAX_X_DEV:
        return False
    return True

def draw_cucumber(img: Any, gs: GameState) -> None:
    x, y = gs.cuke_x, gs.cuke_y
    w, h = gs.cuke_w, gs.cuke_h
    rect_color = (60, 170, 60)
    edge_color = (40, 120, 40)
    cv2.ellipse(img, (x, y), (w // 2, h // 2), 0, 0, 360, rect_color, -1)
    cv2.ellipse(img, (x, y), (w // 2, h // 2), 0, 0, 360, edge_color, 4)
    for off in (-w // 6, 0, w // 6):
        cv2.line(img, (x + off, y - h // 2 + 6), (x + off, y + h // 2 - 6), (80, 200, 80), 2)

def draw_knife(img: Any, gs: GameState, right_anchor: HandAnchor) -> None:
    h, w = img.shape[:2]
    blade_w = int(0.12 * w)
    blade_h = int(0.18 * h)
    handle_h = int(0.06 * h)
    x = min(max(0, right_anchor.last_x - blade_w // 2), w - blade_w)
    y_start = int(0.02 * h)
    y_drop = int(0.12 * h)
    if gs.knife_active:
        y_start += y_drop
    y_end = y_start + blade_h
    if gs.knife_active:
        cv2.rectangle(img, (x, y_start), (x + blade_w, y_end), (230, 230, 230), -1)
        cv2.rectangle(img, (x, y_start), (x + blade_w, y_end), (160, 160, 160), 3)
        cv2.rectangle(img, (x + blade_w // 3, y_end), (x + 2 * blade_w // 3, y_end + handle_h), (50, 50, 50), -1)
        if gs.hit_test((x, y_start, blade_w, y_end - y_start)):
            gs.slice_until = max(gs.slice_until, time.time() + 0.25)
    if gs.slice_until and time.time() < gs.slice_until:
        cv2.line(img, (gs.cuke_x - gs.cuke_w // 2, gs.cuke_y), (gs.cuke_x + gs.cuke_w // 2, gs.cuke_y), (0, 0, 255), 6)

def draw_hud(img: Any) -> None:
    cv2.putText(img, 'Left hand moves cucumber. Right hand chop to slice. Press x to exit.', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)

# --- Run ---
cv2.destroyAllWindows()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

left_anchor = HandAnchor(LEFT_SMOOTHING_ALPHA)
right_anchor = HandAnchor(RIGHT_SMOOTHING_ALPHA)

gs = GameState(FRAME_WIDTH, FRAME_HEIGHT)

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t_now = time.time()
        results = hands.process(rgb)
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                anchor_lm = hand_lms.landmark[0]
                if is_left(handedness):
                    left_anchor.update(anchor_lm, FRAME_WIDTH, FRAME_HEIGHT, t_now)
                else:
                    right_anchor.update(anchor_lm, FRAME_WIDTH, FRAME_HEIGHT, t_now)
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
        if len(left_anchor.motion.points) > 0:
            gs.move_cucumber_to(left_anchor.last_x)
        if detect_right_hand_chop(right_anchor):
            gs.trigger_knife()
        gs.update()
        draw_cucumber(frame, gs)
        draw_knife(frame, gs, right_anchor)
        draw_hud(frame)
        cv2.imshow('Cucumber Slicer (1-cell)', frame)
        if (cv2.waitKey(1) & 0xFF) == ord('x'):
            break
cap.release()
cv2.destroyAllWindows()
print('Game stopped.')
