# 🛡️ AI-Driven Continuous Behavioral Authentication System

## 📌 Project Overview
The **AI-Driven Continuous Behavioral Authentication System** is a sophisticated, zero-trust security solution designed to protect computer systems from unauthorized access in real-time. Unlike traditional passwords or biometrics (FaceID/Fingerprint) which only authenticate you **once** at login, this system **continuously** monitors your unique behavioral "fingerprint"—the way you move your mouse and type on your keyboard.

If a different person (e.g., a friend or an intruder) touches the laptop, the AI detects a behavioral mismatch within **2-5 seconds**, immediately **locks the system**, and sends a **Critical Security Alert** to the owner's email.

---

## 🛠️ Complete Tech Stack
### **1. Programming Language & Environment**
*   **Python 3.8+**: The core language used for the entire logic, data processing, and AI training.
*   **Git & GitHub**: Version control and code management.

### **2. Machine Learning & Data Science**
*   **Scikit-Learn**: Used for the **Isolation Forest** (Anomaly Detection) algorithm.
*   **Pandas & NumPy**: For high-speed data manipulation and feature vector engineering.
*   **StandardScaler**: To normalize behavioral data (speed, acceleration) so the AI can learn accurately.

### **3. System & Input Interaction**
*   **Pynput**: Captures real-time keyboard events (press/release timing).
*   **Quartz (macOS) / Win32API (Windows) / Xlib (Linux)**: Low-level system libraries for high-precision cursor tracking (100Hz sampling).
*   **Subprocess & Platform**: To execute system-level "Lock" commands across Windows, macOS, and Linux.

### **4. Communication & Integration**
*   **SMTPLIB (SMTP)**: Authenticated integration with Gmail for instant security alerts.
*   **YAML (PyYAML)**: For professional-grade configuration management.

---

## 🏗️ Technical Architecture (The 8-Step Pipeline)

### **Step 1: Data Capture (The Sensor)**
Captures high-precision raw data:
- **Cursor**: X/Y coordinates, velocity ($\Delta$ Distance / $\Delta$ Time), and acceleration.
- **Keyboard**: Key hold duration (Dwell time) and Inter-key delay (Flight time).

### **Step 2: Feature Fusion (The Brains)**
Combines raw signals into advanced behavioral features like:
- `keyboard_typing_speed` (characters per second).
- `cursor_movement_smoothness`.
- `keyboard_pause_frequency`.

### **Step 3: Sliding Window Logic**
Uses a **2-5 second rolling window**. The AI doesn't just look at one click; it looks at the *sequence* of movements over the last few seconds to ensure zero false-alarms while maintaining high security.

### **Step 4: AI Model - Isolation Forest**
Why **Isolation Forest**?
- **One-Class Learning**: It only needs to learn *your* behavior. It doesn't need to see "attacker" data to know someone isn't you.
- **Efficiency**: It is lightweight enough to run in the background without slowing down your laptop.

### **Step 5: Decision & Threshold Logic**
- **Confidence ≥ 0.8**: `ALLOW` (Owner verified).
- **Confidence 0.7 - 0.8**: `MONITOR` (Suspicious activity).
- **Confidence < 0.7**: `LOCK` (Immediate security action).

### **Step 6: Security Action (The Enforcer)**
Triggers multi-platform system locks (`rundll32.exe` on Windows, `gnome-screensaver` on Linux, etc.).

### **Step 7: Real-Time Alerting**
A background `AlertManager` handles authenticated email sending using a **Retrying Queue** system (if internet drops, it tries again later).

### **Step 8: Adaptive Learning**
The system includes an `AdaptiveModelUpdater`. As your typing style changes slightly over months (e.g., getting faster at a keyboard), the AI **gradually adapts** to your new patterns without compromising security.

---

## 🚀 Interview "Key Questions" Prep
*   **Why is this better than a password?** Passwords can be stolen, but your muscle memory (typing rhythm and mouse curves) is nearly impossible to replicate.
*   **How does it handle privacy?** All data collection and AI training happen **locally** on the laptop. No behavioral data is ever sent to a cloud server.
*   **What happens if the laptop is idle?** The system includes an `IdleDetector` that pauses authentication to save CPU when you aren't using the computer.

---
**Project Status:** Production-Ready | Verified Email Alerts | Multi-Threaded Core | Zero-Trust Enabled.
