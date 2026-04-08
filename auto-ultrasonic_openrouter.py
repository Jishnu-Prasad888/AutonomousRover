import cv2
import asyncio
import websockets
import time
import statistics
import json
import requests
from collections import deque
from gpiozero import Motor, PWMOutputDevice
from ultralytics import YOLO
from datetime import datetime

# ─────────────────────────────────────────────
# MOTOR SETUP
# ─────────────────────────────────────────────
left_motor  = Motor(forward=17, backward=27)
right_motor = Motor(forward=22, backward=23)
left_pwm    = PWMOutputDevice(18)
right_pwm   = PWMOutputDevice(19)

BASE_SPEED  = 0.1
TURN_SPEED  = 0.1  

def _set_pwm(l, r):
    left_pwm.value  = max(0.0, min(1.0, l))
    right_pwm.value = max(0.0, min(1.0, r))

def drive_forward():
    _set_pwm(BASE_SPEED, BASE_SPEED)
    left_motor.forward()
    right_motor.forward()
    print(f"  ▶  FORWARD")

def drive_back():
    _set_pwm(BASE_SPEED, BASE_SPEED)
    left_motor.backward()
    right_motor.backward()
    print(f"  ◀  BACKWARD")

def turn_left():
    _set_pwm(0.0, TURN_SPEED)
    left_motor.forward()
    right_motor.forward()
    print(f"  ↰  LEFT")

def turn_right():
    _set_pwm(TURN_SPEED, 0.0)
    left_motor.forward()
    right_motor.forward()
    print(f"  ↱  RIGHT")

def stop_motors():
    left_motor.stop()
    right_motor.stop()
    _set_pwm(0, 0)
    print("  ■  STOP")

# ─────────────────────────────────────────────
# CAMERA + YOLO
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
model = YOLO('yolov8n.pt')

# ─────────────────────────────────────────────
# ULTRASONIC SENSOR
# ─────────────────────────────────────────────
SENSOR_WINDOW = 7
OBSTACLE_CONSEC = 4
DIST_OBSTACLE_M = 0.28
SENSOR_TIMEOUT_S = 3.0

_dist_buffer = deque(maxlen=SENSOR_WINDOW)
_obstacle_count = 0
_last_sensor_time = 0.0
latest_distance = 9.9

def update_distance(raw_cm: float):
    if raw_cm <= 0 or raw_cm > 400: return
    _dist_buffer.append(raw_cm / 100.0)

def filtered_distance() -> float:
    return statistics.median(_dist_buffer) if _dist_buffer else 9.9

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

# ─────────────────────────────────────────────
# WEBSOCKET SERVER
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# OPENROUTER INTEGRATION
# ─────────────────────────────────────────────
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY = ""
MODEL_NAME = "google/gemma-4-26b-a4b-it:free"

# Simple obstacle avoidance when AI fails
def simple_obstacle_avoidance(distance, blocked):
    """Fallback logic when AI is unavailable"""
    if blocked or distance < 0.3:
        print(f"[FALLBACK] Obstacle at {distance:.2f}m - turning right")
        return "right"
    elif distance < 0.5:
        print(f"[FALLBACK] Close obstacle at {distance:.2f}m - turning slightly")
        return "left"
    else:
        return "front"

def extract_yolo_data(results, frame_shape):
    """Extract YOLO detection data as JSON"""
    data = []
    h, w = frame_shape[:2]
    for r in results:
        if r.boxes is not None:
            for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                x1, y1, x2, y2 = box.tolist()
                cls_name = r.names[int(cls_id)]
                center_x, center_y = (x1+x2)/2, (y1+y2)/2
                width, height = x2-x1, y2-y1
                data.append({
                    'class': cls_name,
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [float(center_x/w), float(center_y/h)],
                    'size': [float(width/w), float(height/h)]
                })
    return data

def get_ai_direction(yolo_data, distance, timestamp):
    """Send YOLO + distance to OpenRouter and get direction"""
    
    system_prompt = """You are controlling an autonomous rover. Analyze the YOLO detection data and distance sensor.
If distance < 0.3 meters, respond with "back" or "turn".
If objects are detected, navigate around them.
RESPOND WITH EXACTLY ONE WORD: "front", "back", "right", "left", "stop"
ONE WORD ONLY."""

    context = json.dumps({
        "timestamp": timestamp,
        "distance_meters": distance,
        "yolo_detections": yolo_data
    }, indent=2)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ],
        "temperature": 0.1,
        "max_tokens": 10  # Limit response length
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=10.0)
        
        if resp.status_code == 429:
            print(f"[AI] Rate limited - using fallback")
            return None  # Signal to use fallback
            
        resp.raise_for_status()
        result = resp.json()
        direction = result["choices"][0]["message"]["content"].strip().lower()
        
        # Validate response
        if direction in ["front", "back", "right", "left", "stop"]:
            print(f"[AI] Direction: {direction}")
            return direction
        else:
            print(f"[AI] Invalid response: {direction}")
            return None
            
    except Exception as e:
        print(f"[AI] OpenRouter error: {e}")
        return None

# ─────────────────────────────────────────────
# AI ROVER LOOP
# ─────────────────────────────────────────────
async def ai_rover_loop():
    print("[AI-ROVER] Starting navigation with 10s AI requests + fallback")
    last_capture = 0
    last_request_time = 0
    REQUEST_DELAY = 10.0
    last_direction = "front"  # Start moving forward by default
    consecutive_failures = 0
    
    while True:
        current_time = time.time()
        if current_time - last_capture >= 0.5:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.05)
                continue

            frame = cv2.flip(frame, 0)
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            # Run YOLO detection
            results = model.predict(frame, conf=0.25, verbose=False)
            yolo_data = extract_yolo_data(results, frame.shape)

            # Get distance
            distance = filtered_distance()
            blocked = obstacle_blocking()

            # Decide direction
            direction = None
            
            # Only call OpenRouter if delay has passed
            if current_time - last_request_time >= REQUEST_DELAY:
                direction = get_ai_direction(yolo_data, distance, timestamp)
                last_request_time = current_time
                
                if direction is None:
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0
                    last_direction = direction
            
            # Use fallback if AI failed or we have too many failures
            if direction is None:
                if consecutive_failures >= 2:
                    # Use simple obstacle avoidance
                    direction = simple_obstacle_avoidance(distance, blocked)
                else:
                    # Keep last working direction
                    direction = last_direction
                    print(f"[FALLBACK] Using last direction: {direction}")
            
            # Execute movement
            if direction == "front": 
                drive_forward()
            elif direction == "back": 
                drive_back()
            elif direction == "right": 
                turn_right()
            elif direction == "left": 
                turn_left()
            elif direction == "stop": 
                stop_motors()
            else: 
                stop_motors()

            last_capture = current_time
            print(f"[CYCLE] {timestamp} | dist={distance:.2f}m | objs={len(yolo_data)} | dir={direction} | blocked={blocked} | fails={consecutive_failures}")

        await asyncio.sleep(0.05)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
async def run():
    await asyncio.gather(start_ws_server(), ai_rover_loop())

if __name__ == "__main__":
    try:
        print("[SETUP] OpenRouter with 10s rate limiting + obstacle avoidance fallback")
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[ROVER] Shutdown")
    finally:
        stop_motors()
        cap.release()
        cv2.destroyAllWindows()
