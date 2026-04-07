#include <Arduino.h>
#include <WiFi.h>
#include <ArduinoWebsockets.h>

using namespace websockets;

#define TRIG_PIN 5
#define ECHO_PIN 18
#define NUM_READINGS 5

const char* ssid = "bail";
const char* password = "bail2345678";

// ✅ Point to Raspberry Pi IP and port 81
const char* ws_server = "ws://10.205.183.99:8081";

long readings[NUM_READINGS];
int readIndex = 0;
long total = 0;

unsigned long startTime;
WebsocketsClient client;

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  for (int i = 0; i < NUM_READINGS; i++) readings[i] = 0;
  startTime = millis();

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected to WiFi");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());

  Serial.println("Connecting to WebSocket (Raspberry Pi)...");
  while (!client.connect(ws_server)) {
    Serial.println("Retrying WebSocket connection...");
    delay(1000);
  }
  Serial.println("WebSocket connected to Pi!");
}

long measureDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  long distance = (duration == 0) ? 0 : duration * 0.034 / 2;
  return distance;
}

long getFilteredDistance() {
  if (millis() - startTime < 3000) return 400; // safe fallback during warmup
  total -= readings[readIndex];
  readings[readIndex] = measureDistance();
  total += readings[readIndex];
  readIndex = (readIndex + 1) % NUM_READINGS;
  return total / NUM_READINGS;
}

void loop() {
  client.poll(); // ✅ Required to keep connection alive

  if (millis() - startTime >= 3000) {
    long dist = getFilteredDistance();

    if (client.available()) {
      String payload = String(dist);
      client.send(payload);
      Serial.print("Sent: ");
      Serial.print(payload);
      Serial.println(" cm");
    } else {
      Serial.println("WebSocket disconnected, reconnecting...");
      client.connect(ws_server);
    }
  }

  delay(100);
}
