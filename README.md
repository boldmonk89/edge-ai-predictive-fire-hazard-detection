# Edge-AI Based Predictive Fire Hazard Detection System

## 📌 Project Overview
This project implements an **Edge-AI based predictive fire hazard detection system** that identifies potential fire risks *before ignition* using IoT sensor data and machine learning.

Traditional fire alarm systems are reactive and trigger alerts only after smoke or fire appears. In contrast, this system predicts hazardous conditions early by analyzing **combined environmental patterns** such as temperature, humidity, and gas concentration directly at the edge, resulting in low latency and reduced false alarms.

---

## 🎯 Objectives
- Predict fire hazards before ignition
- Perform machine learning inference at the **edge device**
- Reduce false alarms compared to threshold-based systems
- Provide real-time visualization using a dashboard
- Minimize dependency on cloud services

---

## 🧠 Problem Statement
Conventional fire detection systems rely on fixed thresholds and react only after smoke or fire is detected.  
Such delayed response can cause significant damage in residential buildings, hostels, factories, and warehouses.

There is a need for a **predictive fire hazard detection system** that can analyze environmental trends and warn users *before* a fire occurs.

---

## 💡 Proposed Solution
The proposed system uses multiple sensors connected to an ESP32 microcontroller to continuously monitor environmental parameters.  
A supervised machine learning model analyzes the combined sensor data to classify fire risk levels as:

- **Safe**
- **Warning**
- **Fire Risk**

The prediction is performed locally at the edge, and the results are sent to a backend API and visualized through a dashboard.

---

## 🏗️ System Architecture
**Data Flow:**

Sensors → ESP32 → Edge ML Model → Backend API → Dashboard

**Key Point:**  
Machine learning inference is executed locally on the edge device to ensure faster response, reliability, and reduced cloud dependency.

---

## 🔧 Hardware Components
- ESP32 (Wi-Fi enabled microcontroller)
- DHT11 / DHT22 (Temperature & Humidity Sensor)
- MQ-2 / MQ-135 (Gas / Smoke Sensor)
- Buzzer and LED
- Breadboard and jumper wires

---

## 💻 Software & Tools

### Programming & Machine Learning
- Python
- Scikit-learn
- Logistic Regression / Random Forest

### Embedded Development
- Arduino IDE
- ESP32 libraries

### Dashboard & Visualization
- Lovable (used strictly for UI and visualization)

---

## 📊 Dataset Description
The dataset consists of environmental sensor readings with the following attributes:
- Temperature
- Humidity
- Gas concentration
- Fire risk label (Safe / Warning / Fire)

The dataset can be collected using real sensors or simulated for training and testing purposes in an academic environment.

---

## 📁 Project Structure

- **hardware/**  
  Contains ESP32 source code, circuit diagram, and list of hardware components.

- **data/**  
  Includes the dataset used for training the machine learning model along with its description.

- **model/**  
  Consists of the machine learning training script, saved model, and explanation of the algorithm used.

- **backend/**  
  Handles API logic for receiving sensor data and serving fire risk predictions.

- **dashboard/**  
  Documentation and screenshots of the Lovable-based dashboard used for visualization.

- **docs/**  
  System architecture diagram, flowchart, and presentation (PPT) outline.

- **report/**  
  Final academic project report submitted for evaluation.

---

## 🖥️ Dashboard
The system includes a web-based dashboard to visualize fire risk levels and sensor readings in real time.

### Features:
- Real-time fire risk status (Low / Medium / High)
- Visualization of temperature, humidity, and gas levels
- Visual alert indication for hazardous conditions

**Note:**  
The dashboard is developed using Lovable and is used strictly for visualization purposes.  
No sensor data processing or machine learning inference is performed on the dashboard.

---

## 🚀 Key Features
- Predictive fire hazard detection (not reactive)
- Edge-based machine learning inference
- Multi-sensor data analysis
- Reduced latency and false alarms
- Modular and scalable system design
  
---

## 📄 Project Status
- Academic Minor Project
- Team-based implementation
- Hardware and software integrated

---

## 📌 Disclaimer
This project is developed for academic purposes only.  
It demonstrates predictive fire hazard detection concepts and is not a certified industrial or commercial fire safety system.
