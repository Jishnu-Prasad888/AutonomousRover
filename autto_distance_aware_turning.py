"""
Autonomous Rover — FIXED: Straight when clear, turns only when needed
"""

import cv2
import asyncio
import websockets
import time
import statistics
from collections import deque
from gpiozero import Motor, PWMOutputDevice
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# MOTOR SETUP (ALL SPEEDS = 0.1 as requested)
# ─────────────────────────────────────────────────────────────
left_motor  = Motor(forward=17, backward=27)
right_motor = Motor(forward=22, backward=23)
left_pwm    = PWMOutputDevice(18)
right_pwm   = PWMOutputDevice(19)

BASE_SPEED  = 0.1
TURN_SPEED  = 0.1  
INNER_SPEED = 0.1

def _set_pwm(l, r):
    left_pwm.value  = max(0.0, min(1.0, l))
    right_pwm.value = max(0.0, min(1.0, r))

def drive_forward(speed=None):
    s = speed or BASE_SPEED
    _set_pwm(s, s)
    left_motor.forward()
    right_motor.forward()
    print(f"  ▶  FORWARD  (speed={s:.2f})")

def steer_left(sharp=False):
    inner = 0.0 if sharp else INNER_SPEED
    _set_pwm(inner, TURN_SPEED)
    left_motor.forward()
    right_motor.forward()
    print(f"  ↰  {'SHARP LEFT' if sharp else 'CURVE LEFT'}")

def steer_right(sharp=False):
    inner = 0.0 if sharp else INNER_SPEED
    _set_pwm(TURN_SPEED, inner)
    left_motor.forward()
    right_motor.forward()
    print(f"  ↱  {'SHARP RIGHT' if sharp else 'CURVE RIGHT'}")

def stop_motors():
    left_motor.stop()
    right_motor.stop()
    _set_pwm(0, 0)
    print("  ■  STOP")

# ─────────────────────────────────────────────────────────────
# CAMERA + YOLO (unchanged)
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

model        = YOLO('yolov8n.pt')
TARGET_CLASS = 39
FRAME_W  = 320
FRAME_H  = 240
FRAME_CX = FRAME_W / 2

SHARP_THRESH     = 0.40
CURVE_THRESH     = 0.15
ARRIVED_AREA_FRAC = 0.35

# ─────────────────────────────────────────────────────────────
# ULTRASONIC (unchanged - filtering works well)
# ─────────────────────────────────────────────────────────────
SENSOR_WINDOW    = 7
OBSTACLE_CONSEC  = 4
DIST_OBSTACLE_M  = 0.28
SENSOR_TIMEOUT_S = 3.0

_dist_buffer     = deque(maxlen=SENSOR_WINDOW)
_obstacle_count  = 0
_last_sensor_time = 0.0
latest_distance  = 9.9

def update_distance(raw_cm: float):
    if raw_cm <= 0 or raw_cm > 400:
        return
    _dist_buffer.append(raw_cm / 100.0)

def filtered_distance() -> float:
    if not _dist_buffer:
        return 9.9
    return statistics.median(_dist_buffer)

def obstacle_blocking() -> bool:
    global _obstacle_count
    if time.time() - _last_sensor_time > SENSOR_TIMEOUT_S:
        _obstacle_count = 0
        return False
    fd = filtered_distance()
    if fd < DIST_OBSTACLE_M:
        _obstacle_count += 1
    else:
        _obstacle_count = max(0, _obstacle_count - 1)
    return _obstacle_count >= OBSTACLE_CONSEC

# ─────────────────────────────────────────────────────────────
# WEBSOCKET (unchanged)
# ─────────────────────────────────────────────────────────────
async def distance_server(websocket):
    global _last_sensor_time, latest_distance
    print(f"[WS] ESP32 connected: {websocket.remote_address}")
    try:
        async for msg in websocket:
            try:
                _last_sensor_time = time.time()
                raw = float(msg)
                latest_distance = raw / 100.0
                update_distance(raw)
            except ValueError:
                print("[WS] bad packet:", msg)
    except websockets.exceptions.ConnectionClosed:
        print("[WS] ESP32 disconnected")

async def start_ws_server():
    async with websockets.serve(distance_server, "0.0.0.0", 8081):
        print("[WS] Server ready on :8081")
        await asyncio.Future()

# ─────────────────────────────────────────────────────────────
# STATE MACHINE
# ─────────────────────────────────────────────────────────────
SEARCH   = "SEARCH"
APPROACH = "APPROACH"
BLOCKED  = "BLOCKED"
ARRIVED  = "ARRIVED"

state        = SEARCH
search_ticks = 0
straight_ticks = 0  # NEW: Track straight movement

# ─────────────────────────────────────────────────────────────
# MAIN LOOP — STRAIGHT WHEN CLEAR
# ─────────────────────────────────────────────────────────────
async def rover_loop():
    global state, search_ticks, straight_ticks

    print("[ROVER] Straight-line search mode")

    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.05)
            continue
        frame = cv2.flip(frame, 0)

        # YOLO detection
        results = model.predict(frame, conf=0.45, verbose=False)
        best_score  = -1
        target_cx   = None
        target_area = 0.0

        for r in results:
            for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                if int(cls_id) != TARGET_CLASS:
                    continue
                x1, y1, x2, y2 = map(float, box)
                area  = (x2 - x1) * (y2 - y1)
                score = area * float(conf)
                if score > best_score:
                    best_score  = score
                    target_cx   = (x1 + x2) / 2.0
                    target_area = area

        target_detected = target_cx is not None
        target_close    = target_detected and (target_area / (FRAME_W * FRAME_H) >= ARRIVED_AREA_FRAC)

        blocked = obstacle_blocking()

        # State transitions
        if blocked:
            if state != BLOCKED:
                print(f"\n[STATE] → BLOCKED  filtered={filtered_distance():.2f}m")
            state = BLOCKED
            straight_ticks = 0

        elif state == BLOCKED:
            print("[STATE] BLOCKED → SEARCH (clear)")
            state = SEARCH

        else:
            if target_close:
                new_state = ARRIVED
            elif target_detected:
                new_state = APPROACH
            else:
                new_state = SEARCH

            if new_state != state:
                print(f"[STATE] {state} → {new_state}")
                state = new_state

        # ── EXECUTE ──
        if state == BLOCKED:
            stop_motors()

        elif state == ARRIVED:
            stop_motors()
            print("[ROVER] ★ ARRIVED")
            await asyncio.sleep(1.5)
            state = SEARCH

        elif state == APPROACH:
            error = (target_cx - FRAME_CX) / (FRAME_W / 2)
            abs_e = abs(error)
            area_frac = target_area / (FRAME_W * FRAME_H)
            proximity = 1.0 - max(0.0, (area_frac - 0.10) * 2)
            spd = max(0.08, BASE_SPEED * proximity)  # Slightly slower

            if abs_e > SHARP_THRESH:
                steer_left(sharp=True)  if error < 0 else steer_right(sharp=True)
            elif abs_e > CURVE_THRESH:
                steer_left(sharp=False) if error < 0 else steer_right(sharp=False)
            else:
                drive_forward(spd)

        elif state == SEARCH:
            search_ticks += 1
            straight_ticks += 1

            # STRAIGHT WHEN CLEAR (>1.5m) OR after successful straight movement
            clear_ahead = filtered_distance() > 1.5
            
            if clear_ahead and straight_ticks < 50:  # 2.5s straight
                drive_forward()
                print("  ▶  STRAIGHT (clear path)")
            
            # WALL-FOLLOW AFTER 2.5s OR close obstacle
            elif search_ticks > 30 or not clear_ahead:
                # Alternate sides every 25 ticks (~1.25s)
                turn_dir = "left" if (search_ticks // 25) % 2 == 0 else "right"
                
                if filtered_distance() < 0.8:  # Close wall → sharp turn
                    if turn_dir == "left":
                        steer_left(sharp=True)
                    else:
                        steer_right(sharp=True)
                else:  # Normal curve
                    if turn_dir == "left":
                        steer_left(sharp=False)
                    else:
                        steer_right(sharp=False)
                
                straight_ticks = 0  # Reset straight counter
            else:
                # Switch direction after long straight
                search_ticks = 0

        print(f"    [sensor] raw={latest_distance:.2f}m filtered={filtered_distance():.2f}m "
              f"streak={_obstacle_count}/4 state={state} straight={straight_ticks}")

        await asyncio.sleep(0.05)

async def run():
    await asyncio.gather(start_ws_server(), rover_loop())

try:
    asyncio.run(run())
except KeyboardInterrupt:
    print("\n[ROVER] Shutdown")
finally:
    stop_motors()
    cap.release()
    cv2.destroyAllWindows()
