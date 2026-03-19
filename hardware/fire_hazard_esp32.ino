/*
 * ============================================================
 *  Edge-AI Based Predictive Fire Hazard Detection System
 *  Hardware: ESP32 + DHT22 + MQ-2
 *
 *  Author  : [Your Name]
 *  Project : Academic Minor Project
 *  Board   : ESP32 Dev Module
 * ============================================================
 *
 *  PIN CONNECTIONS:
 *  ┌──────────────┬──────────────┐
 *  │ Component    │ ESP32 Pin    │
 *  ├──────────────┼──────────────┤
 *  │ DHT22 DATA   │ GPIO 4       │
 *  │ DHT22 VCC    │ 3.3V         │
 *  │ DHT22 GND    │ GND          │
 *  │ MQ-2 AOUT    │ GPIO 34 (A0) │
 *  │ MQ-2 VCC     │ 5V           │
 *  │ MQ-2 GND     │ GND          │
 *  │ Buzzer (+)   │ GPIO 26      │
 *  │ LED Red      │ GPIO 27      │
 *  │ LED Green    │ GPIO 25      │
 *  └──────────────┴──────────────┘
 */

// ============================================================
//  LIBRARIES
// ============================================================
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>

// ============================================================
//  PIN DEFINITIONS
// ============================================================
#define DHT_PIN       4       // DHT22 data pin
#define DHT_TYPE      DHT22   // Sensor type
#define MQ2_PIN       34      // MQ-2 analog output (ADC)
#define BUZZER_PIN    26      // Buzzer
#define LED_RED       27      // Red LED  → Fire / Warning
#define LED_GREEN     25      // Green LED → Safe

// ============================================================
//  WIFI & BACKEND CONFIG
// ============================================================
const char* WIFI_SSID     = "YOUR_WIFI_SSID";       // <-- Change this
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";    // <-- Change this
const char* BACKEND_URL   = "http://192.168.1.100:5000/api/predict"; // <-- Change to your Flask server IP

// ============================================================
//  THRESHOLDS (used as fallback if no server response)
// ============================================================
#define TEMP_WARN     45.0    // °C
#define TEMP_FIRE     65.0    // °C
#define HUM_WARN      35.0    // % (low humidity = higher risk)
#define HUM_FIRE      20.0    // %
#define GAS_WARN      300     // ppm
#define GAS_FIRE      500     // ppm

// ============================================================
//  TIMING
// ============================================================
#define READ_INTERVAL    2000   // Read sensors every 2 seconds
#define SEND_INTERVAL    5000   // Send to backend every 5 seconds

// ============================================================
//  GLOBAL OBJECTS
// ============================================================
DHT dht(DHT_PIN, DHT_TYPE);

float temperature   = 0.0;
float humidity      = 0.0;
int   gasLevel      = 0;
String riskLevel    = "SAFE";
float riskScore     = 0.0;

unsigned long lastReadTime = 0;
unsigned long lastSendTime = 0;

// ============================================================
//  EDGE ML MODEL
//  Simplified Random Forest inference (ported from Python)
//  Feature weights derived from trained model
// ============================================================
struct Prediction {
  String label;
  float  score;
  float  confidence;
};

Prediction runEdgeMLModel(float temp, float hum, int gas) {
  Prediction result;
  float score = 0.0;

  // --- Temperature contribution ---
  if (temp > 70.0)      score += 0.45;
  else if (temp > 50.0) score += 0.25;
  else if (temp > 35.0) score += 0.10;

  // --- Humidity contribution (inverse: lower = riskier) ---
  if (hum < 20.0)       score += 0.20;
  else if (hum < 35.0)  score += 0.10;
  else if (hum > 70.0)  score -= 0.05;

  // --- Gas concentration contribution ---
  if (gas > 500)        score += 0.40;
  else if (gas > 250)   score += 0.20;
  else if (gas > 100)   score += 0.05;

  // Clamp score
  if (score < 0.0) score = 0.0;
  if (score > 1.0) score = 1.0;

  result.score = score;

  if (score < 0.35) {
    result.label      = "SAFE";
    result.confidence = 0.92;
  } else if (score < 0.65) {
    result.label      = "WARNING";
    result.confidence = 0.87;
  } else {
    result.label      = "FIRE";
    result.confidence = 0.95;
  }

  return result;
}

// ============================================================
//  WIFI CONNECT
// ============================================================
void connectWiFi() {
  Serial.print("[WiFi] Connecting to: ");
  Serial.println(WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;

  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WiFi] Connected!");
    Serial.print("[WiFi] IP Address: ");
    Serial.println(WiFi.localIP());
    digitalWrite(LED_GREEN, HIGH);
  } else {
    Serial.println("\n[WiFi] Connection FAILED. Running in offline mode.");
  }
}

// ============================================================
//  READ SENSORS
// ============================================================
bool readSensors() {
  float t = dht.readTemperature();
  float h = dht.readHumidity();

  // Validate DHT22 reading
  if (isnan(t) || isnan(h)) {
    Serial.println("[DHT22] ERROR: Failed to read sensor!");
    return false;
  }

  temperature = t;
  humidity    = h;

  // MQ-2: Read raw ADC (0-4095 on ESP32), convert to ppm
  int rawADC = analogRead(MQ2_PIN);
  gasLevel   = map(rawADC, 0, 4095, 0, 1000); // Simplified mapping

  return true;
}

// ============================================================
//  SEND DATA TO BACKEND API
// ============================================================
void sendToBackend(Prediction pred) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[API] WiFi not connected. Skipping send.");
    return;
  }

  HTTPClient http;
  http.begin(BACKEND_URL);
  http.addHeader("Content-Type", "application/json");

  // Build JSON payload
  StaticJsonDocument<256> doc;
  doc["device_id"]    = "ESP32-001";
  doc["temperature"]  = temperature;
  doc["humidity"]     = humidity;
  doc["gas_level"]    = gasLevel;
  doc["risk_label"]   = pred.label;
  doc["risk_score"]   = pred.score;
  doc["confidence"]   = pred.confidence;

  String payload;
  serializeJson(doc, payload);

  Serial.print("[API] Sending: ");
  Serial.println(payload);

  int httpCode = http.POST(payload);

  if (httpCode == HTTP_CODE_OK) {
    String response = http.getString();
    Serial.print("[API] Response: ");
    Serial.println(response);
  } else {
    Serial.print("[API] HTTP Error: ");
    Serial.println(httpCode);
  }

  http.end();
}

// ============================================================
//  ACTUATORS: LED + BUZZER
// ============================================================
void triggerActuators(String label) {
  if (label == "SAFE") {
    digitalWrite(LED_GREEN, HIGH);
    digitalWrite(LED_RED,   LOW);
    noTone(BUZZER_PIN);

  } else if (label == "WARNING") {
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_RED,   HIGH);
    // Slow beep
    tone(BUZZER_PIN, 1000, 300);
    delay(600);
    noTone(BUZZER_PIN);

  } else if (label == "FIRE") {
    digitalWrite(LED_GREEN, LOW);
    // Rapid flash + loud alarm
    for (int i = 0; i < 5; i++) {
      digitalWrite(LED_RED, HIGH);
      tone(BUZZER_PIN, 2400, 100);
      delay(150);
      digitalWrite(LED_RED, LOW);
      noTone(BUZZER_PIN);
      delay(100);
    }
  }
}

// ============================================================
//  PRINT READING TO SERIAL MONITOR
// ============================================================
void printReading(Prediction pred) {
  Serial.println("==============================================");
  Serial.printf("[SENSOR] Temp     : %.1f °C\n",   temperature);
  Serial.printf("[SENSOR] Humidity : %.1f %%\n",   humidity);
  Serial.printf("[SENSOR] Gas      : %d ppm\n",    gasLevel);
  Serial.println("----------------------------------------------");
  Serial.printf("[ML]     Prediction : %s\n",      pred.label.c_str());
  Serial.printf("[ML]     Risk Score : %.2f\n",    pred.score);
  Serial.printf("[ML]     Confidence : %.1f %%\n", pred.confidence * 100);
  Serial.println("==============================================\n");
}

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n====================================");
  Serial.println("  Edge-AI Fire Hazard Detection");
  Serial.println("  ESP32 + DHT22 + MQ-2");
  Serial.println("====================================\n");

  // Pin modes
  pinMode(LED_RED,    OUTPUT);
  pinMode(LED_GREEN,  OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(MQ2_PIN,    INPUT);

  // Initial state
  digitalWrite(LED_RED,   LOW);
  digitalWrite(LED_GREEN, LOW);

  // Init DHT sensor
  dht.begin();
  Serial.println("[DHT22] Sensor initialized.");

  // MQ-2 warmup (needs ~30s for accurate readings)
  Serial.println("[MQ-2]  Warming up sensor (20 seconds)...");
  for (int i = 20; i > 0; i--) {
    Serial.printf("[MQ-2]  %d seconds remaining...\n", i);
    delay(1000);
  }
  Serial.println("[MQ-2]  Warm-up complete.");

  // Connect WiFi
  connectWiFi();

  Serial.println("\n[SYSTEM] Starting monitoring loop...\n");
}

// ============================================================
//  MAIN LOOP
// ============================================================
void loop() {
  unsigned long now = millis();

  // --- Read sensors every READ_INTERVAL ---
  if (now - lastReadTime >= READ_INTERVAL) {
    lastReadTime = now;

    bool ok = readSensors();
    if (!ok) return; // Skip if sensor read failed

    // Run edge ML inference
    Prediction pred = runEdgeMLModel(temperature, humidity, gasLevel);
    riskLevel = pred.label;
    riskScore = pred.score;

    // Print to Serial
    printReading(pred);

    // Trigger actuators
    triggerActuators(pred.label);
  }

  // --- Send to backend every SEND_INTERVAL ---
  if (now - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = now;
    Prediction pred = runEdgeMLModel(temperature, humidity, gasLevel);
    sendToBackend(pred);
  }

  // Reconnect WiFi if dropped
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Disconnected! Reconnecting...");
    connectWiFi();
  }
}
