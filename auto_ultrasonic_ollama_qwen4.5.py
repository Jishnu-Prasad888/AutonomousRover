"""
Autonomous Rover — AI-Driven Navigation with YOLO + Ollama Qwen:4b
Capture every 0.5s → YOLO → Text data → Ollama → GPIO Control
Strict directions: front, back, right, left, stop ONLY
"""

import cv2
import asyncio
import websockets
import time
import statistics
import json
import requests  # For Ollama API
from collections import deque
from gpiozero import Motor, PWMOutputDevice
from ultralytics import YOLO
import base64
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# MOTOR SETUP
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

# ─────────────────────────────────────────────────────────────
# CAMERA + YOLO
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
model = YOLO('yolov8n.pt')
FRAME_W, FRAME_H = 320, 240

# ─────────────────────────────────────────────────────────────
# ULTRASONIC (unchanged)
# ─────────────────────────────────────────────────────────────
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
# OLLAMA INTEGRATION
# ─────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen:4b"

def image_to_base64(image):
    """Convert cv2 image to base64 for Ollama"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def extract_yolo_data(results, frame_shape):
    """Extract complete YOLO detection data as text"""
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
                    'center': [float(center_x/w), float(center_y/h)],  # normalized
                    'size': [float(width/w), float(height/h)]          # normalized
                })
    
    return data

def get_ai_direction(yolo_data, distance, timestamp):
    """Send YOLO data to Ollama and get strict direction"""
    
    # Create strict system prompt
    system_prompt = """You are controlling an autonomous rover. Analyze ONLY the YOLO detection data provided.
RESPOND WITH EXACTLY ONE WORD from these options ONLY: "front", "back", "right", "left", "stop"
- front: path clear ahead, target centered
- back: too close to obstacle behind
- right: turn right to target/obstacle avoidance
- left: turn left to target/obstacle avoidance  
- stop: immediate danger or target reached

NEVER explain. NEVER use other words. One word only."""

    # Format detection data
    detections_text = json.dumps(yolo_data, indent=2)
    distance_text = f"Distance to nearest obstacle: {distance:.2f}m"
    context = f"""Timestamp: {timestamp}
{distance_text}

YOLO DETECTIONS:
{detections_text}"""

    try:
        # Capture current frame for context
        ret, frame = cap.read()
        if ret:
            img_b64 = image_to_base64(frame)
        else:
            img_b64 = None

        # Ollama request
        payload = {
            "model": MODEL_NAME,
            "prompt": f"{system_prompt}\n\n{context}",
            "stream": False,
            "options": {"temperature": 0.1}  # Deterministic
        }
        
        if img_b64:
            payload["images"] = [img_b64]

        response = requests.post(OLLAMA_URL, json=payload, timeout=15.0)
        if response.status_code == 200:
            result = response.json()
            direction = result['response'].strip().lower()
            print(f"[AI] Direction: {direction}")
            return direction
        else:
            print(f"[AI] Ollama error: {response.status_code}")
            return "stop"
            
    except Exception as e:
        print(f"[AI] Error: {e}")
        return "stop"

# ─────────────────────────────────────────────────────────────
# MAIN AI ROVER LOOP (0.5s cycle)
# ─────────────────────────────────────────────────────────────
async def ai_rover_loop():
    print("[AI-ROVER] Starting YOLO+Ollama navigation (0.5s cycles)")
    last_capture = 0
    
    while True:
        current_time = time.time()
        
        # Capture every 0.5 seconds
        if current_time - last_capture >= 0.5:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.05)
                continue
                
            frame = cv2.flip(frame, 0)
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            # 1. Run YOLO instantly on captured frame
            results = model.predict(frame, conf=0.25, verbose=False)
            yolo_data = extract_yolo_data(results, frame.shape)
            
            # 2. Get sensor data
            distance = filtered_distance()
            blocked = obstacle_blocking()
            
            # 3. Send to Ollama
            direction = get_ai_direction(yolo_data, distance, timestamp)
            
            # 4. Execute STRICT direction
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
                stop_motors()  # Fallback
            
            last_capture = current_time
            
            print(f"[CYCLE] {timestamp} | dist={distance:.2f}m | "
                  f"objs={len(yolo_data)} | dir={direction} | "
                  f"blocked={blocked}")
        
        await asyncio.sleep(0.05)  # Small sleep for responsiveness

# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────
async def run():
    await asyncio.gather(start_ws_server(), ai_rover_loop())

if __name__ == "__main__":
    try:
        # Prerequisites check
        print("[SETUP] Ensure Ollama is running: ollama serve")
        print("[SETUP] Ensure Qwen:4b is pulled: ollama pull qwen:4b")
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[ROVER] Shutdown")
    finally:
        stop_motors()
        cap.release()
        cv2.destroyAllWindows()
