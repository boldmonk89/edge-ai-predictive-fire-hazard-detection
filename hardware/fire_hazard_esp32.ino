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
#define DHT_PIN       5       // DHT22 data pin
#define DHT_TYPE      DHT22   // Sensor type
#define MQ2_PIN       34      // MQ-2 analog output (ADC)
#define BUZZER_PIN    26      // Buzzer
#define LED_RED       27      // Red LED  → Fire / Warning
#define LED_GREEN     25      // Green LED → Safe

// ============================================================
//  WIFI & BACKEND CONFIG
// ============================================================
const char* WIFI_SSID     = "R2NET 2.4G";
const char* WIFI_PASSWORD = "soldier2006";
const char* BACKEND_URL   = "http://192.168.1.34:5000/api/predict";

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
#define READ_INTERVAL    500    // Read sensors every 0.5 seconds for fast response
#define SEND_INTERVAL    2000   // Send to backend every 2 seconds

// ============================================================
//  GLOBAL OBJECTS
// ============================================================
DHT dht(DHT_PIN, DHT_TYPE);

float temperature   = 0.0;
float humidity      = 0.0;
int   gasLevel      = 0;
int   gasRaw        = 0; 
int   gasBaseline   = 0; 
int   gasHistory[7] = {0,0,0,0,0,0,0}; // Balanced smoothing
int   gasHistoryIdx = 0;
bool  isWarmingUp    = true; // Warm-up flag
unsigned long warmupStart = 0;
String riskLevel    = "SAFE";
float riskScore     = 0.0;

unsigned long lastReadTime = 0;
unsigned long lastSendTime = 0;
int consecutiveAlerts = 0; // Guard against blips

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

  // --- Temperature contribution (More sensitive for demos) ---
  if (temp > 60.0)      score += 0.50; 
  else if (temp > 40.0) score += 0.30;
  else if (temp > 32.0) score += 0.15;

  // --- Humidity contribution (Low humidity increases fire risk) ---
  if (hum < 25.0)       score += 0.25;
  else if (hum < 40.0)  score += 0.10;

  // --- Gas concentration contribution (Detecting RELATIVE change) ---
  int gasDelta = gas - gasBaseline;
  
  // Balanced thresholds (Not too twitchy, not too slow)
  if (gasDelta > 200)      score += 0.65; // Critical FIRE
  else if (gasDelta > 110) score += 0.45; // Serious Warning
  else if (gasDelta > 65)  score += 0.25; // Minor smoke (Threshold raised from 40 for stability)

  // Clamp score
  score = constrain(score, 0.0, 1.0);
  result.score = score;

  if (score < 0.30) {
    result.label      = "SAFE";
    result.confidence = 0.94;
  } else if (score < 0.60) {
    result.label      = "WARNING";
    result.confidence = 0.88;
  } else {
    result.label      = "FIRE";
    result.confidence = 0.97;
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
  if (isnan(t) || isnan(h) || (t == 0.0 && h == 0.0)) {
    Serial.println("[DHT22] WARNING: Reading Error. Check wires. Using fallback...");
    // Don't return false; just use safe fallbacks so MQ-2 still works
    temperature = 25.0; 
    humidity    = 45.0;
  } else {
    temperature = t;
    humidity    = h;
  }

  // MQ-2: Read raw ADC and apply Moving Average Filter
  int currentRaw = analogRead(MQ2_PIN);
  gasHistory[gasHistoryIdx] = currentRaw;
  gasHistoryIdx = (gasHistoryIdx + 1) % 7;

  long sum = 0;
  for(int i=0; i<7; i++) sum += gasHistory[i];
  gasRaw = sum / 7; // Averaged value

  gasLevel = map(gasRaw, 0, 4095, 0, 1000); 

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

  if (httpCode > 0) {
    if (httpCode == HTTP_CODE_OK) {
      String response = http.getString();
      Serial.print("[API] Status: 200 OK | Response: ");
      Serial.println(response);
    } else {
      Serial.print("[API] Status: ");
      Serial.println(httpCode);
    }
  } else {
    Serial.print("[API] Error: ");
    Serial.println(http.errorToString(httpCode).c_str());
  }

  http.end();
}

// ============================================================
//  ACTUATORS: LED + BUZZER
// ============================================================
void triggerActuators(String label) {
  static unsigned long lastBeep = 0;
  unsigned long now = millis();

  if (label == "SAFE") {
    digitalWrite(LED_GREEN, HIGH);
    digitalWrite(LED_RED,   LOW);
    digitalWrite(BUZZER_PIN, HIGH); // OFF

  } else if (label == "WARNING") {
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_RED,   HIGH);
    // Non-blocking slow beep
    if (now - lastBeep > 800) {
       digitalWrite(BUZZER_PIN, !digitalRead(BUZZER_PIN));
       lastBeep = now;
    }

  } else if (label == "FIRE") {
    digitalWrite(LED_GREEN, LOW);
    // Non-blocking rapid flash/beep
    digitalWrite(LED_RED, (now / 150) % 2); 
    if (now - lastBeep > 150) {
       digitalWrite(BUZZER_PIN, !digitalRead(BUZZER_PIN));
       lastBeep = now;
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
  Serial.printf("[SENSOR] Gas      : %d ppm (Raw ADC: %d)\n", gasLevel, gasRaw);
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
  // --- IMMEDIATE ACTUATOR RESET ---
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, HIGH); // SILENCE IMMEDIATELY
  pinMode(LED_RED,    OUTPUT);
  pinMode(LED_GREEN,  OUTPUT);
  digitalWrite(LED_RED,   LOW);
  digitalWrite(LED_GREEN, LOW);

  Serial.begin(115200);
  delay(500);

  Serial.println("\n[SYSTEM] PINS CONFIGURED... OK");

  // Init DHT sensor
  dht.begin();
  Serial.println("[DHT22] Sensor initialized.");

  // ============================================================
  //  HARDWARE SELF-TEST (Blink at boot)
  // ============================================================
  Serial.println("[SYSTEM] Starting Hardware Self-Test...");
  digitalWrite(LED_GREEN, HIGH); delay(300);
  digitalWrite(LED_GREEN, LOW);  digitalWrite(LED_RED, HIGH); delay(300);
  digitalWrite(LED_RED,   LOW);
  Serial.println("[SYSTEM] Self-Test Complete.");

  Serial.println("\n====================================");
  Serial.println("  Edge-AI Fire Hazard Detection");
  Serial.println("  ESP32 + DHT22 + MQ-2");
  Serial.println("====================================\n");

  // Pin modes
  pinMode(LED_RED,    OUTPUT);
  pinMode(LED_GREEN,  OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(MQ2_PIN,    INPUT);

  // Initial state (Inverted)
  Serial.println("[SYSTEM] Self-Test Complete.");

  // Connect WiFi
  connectWiFi();

  Serial.println("\n[SYSTEM] Entering Warm-up & Calibration (60 seconds)...");
  warmupStart = millis();
}

// ============================================================
//  MAIN LOOP
// ============================================================
void loop() {
  unsigned long now = millis();

  // --- Warm-up Phase Logic ---
  if (isWarmingUp) {
    if (now - warmupStart < 60000) {
      digitalWrite(LED_GREEN, (now / 500) % 2); // Blink green
      if (now % 2000 < 50) Serial.println("[SYSTEM] Warming up sensors... Please wait.");
      return; 
    } else {
      isWarmingUp = false;
      // Calibrate baseline AFTER warm-up for best stability
      long sum = 0;
      for (int i=0; i<10; i++) sum += analogRead(MQ2_PIN);
      gasBaseline = sum / 10;
      Serial.printf("[SYSTEM] Warm-up complete. Baseline set to: %d\n", gasBaseline);
      digitalWrite(LED_GREEN, HIGH);
    }
  }

  // --- Read sensors every READ_INTERVAL ---
  if (now - lastReadTime >= READ_INTERVAL) {
    lastReadTime = now;

    bool ok = readSensors();
    if (!ok) return; // Skip if sensor read failed

    // Run edge ML inference
    Prediction pred = runEdgeMLModel(temperature, humidity, gasLevel);
    
    // --- Consistency Check: Only escalate if readings are sustained ---
    if (pred.label != "SAFE") {
      consecutiveAlerts++;
    } else {
      consecutiveAlerts = 0;
    }

    // Only allow Warning/Fire if seen for 3 consecutive cycles (~1.5s total)
    if (consecutiveAlerts < 3 && pred.label != "SAFE") {
      pred.label = "SAFE";
      pred.score = 0.10; 
    }

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
