# AI-Driven Continuous Behavioral Authentication System

A zero-trust, single-user behavioral authentication system that continuously monitors cursor and keyboard patterns to detect impostors in real time. Unlike traditional passwords or biometrics that authenticate once at login, this system continuously verifies the user’s behavioral fingerprint every 2–5 seconds. If a different person touches the laptop, the system detects the mismatch, locks the system immediately, and sends a real-time email alert to the owner.

**Live GitHub:** [https://github.com/abhishek14311431/AI-Driven-Continuous-Behavioral-Authentication-System](https://github.com/abhishek14311431/AI-Driven-Continuous-Behavioral-Authentication-System)

## Evaluation Results

Formally verified on held-out behavioral test sets:

| Metric | Result |
| --- | --- |
| Owner Recognition Rate | 95.71% |
| False Rejection Rate (FRR) | 4.29% |
| Impostor Detection Rate | 100% |
| False Acceptance Rate (FAR) | 0% |
| Combined Accuracy | 97% |
| Average Detection Time | 2.06 seconds |

> Results generated on held-out test sets using `behavior_features_sample.csv`.
> See `results/evaluation_report_sample.txt` and `results/timing_benchmark_sample.txt` for full output.
> Run `python main.py --mode evaluate` to reproduce on your own collected data.

## How It Works - 8-Step Pipeline

### Step 1: Data Capture

Captures high-precision raw behavioral data continuously in the background:

- **Cursor:** X/Y coordinates, velocity, acceleration, idle gaps at **100Hz sampling rate**
- **Keyboard:** Key hold duration (dwell time) and inter-key delay (flight time)

### Step 2: Feature Engineering

Converts raw signals into 21 behavioral features:

- `cursor_avg_velocity`, `cursor_movement_smoothness`, `cursor_total_distance` (12 cursor features)
- `keyboard_typing_speed`, `keyboard_avg_hold_duration`, `keyboard_pause_frequency` (9 keyboard features)

### Step 3: Sliding Window

Uses a **2–5 second rolling window** - the AI evaluates sequences of behavior, not individual events, to eliminate false alarms while maintaining security.

### Step 4: Isolation Forest Model

Trained exclusively on **owner behavior (one-class learning)**. No impostor data required for training. Anything that deviates from the learned owner pattern is flagged as an anomaly.

### Step 5: Threshold-Based Decision Logic

- **Confidence ≥ 0.8** → `ALLOW` (owner verified)
- **Confidence 0.7–0.8** → `MONITOR` (suspicious activity)
- **Confidence < 0.7** → `LOCK` (immediate security action)

### Step 6: System Lock

Triggers cross-platform system lock:

- **Windows:** `rundll32.exe user32.dll,LockWorkStation`
- **macOS:** `pmset displaysleepnow`
- **Linux:** `gnome-screensaver-command --lock` / `xdg-screensaver lock`

### Step 7: Real-Time Email Alerts

Background `AlertManager` sends authenticated email alerts via Gmail SMTP when a lock is triggered. Includes a retry queue - if the internet drops, alerts are queued and sent when connectivity is restored.

### Step 8: Adaptive Learning

`AdaptiveModelUpdater` gradually retrains the model on new high-confidence owner sessions (confidence ≥ 0.85) to adapt to natural behavioral drift over time. The model never learns from impostor behavior.

## Tech Stack

| Component | Technology |
| --- | --- |
| Core language | Python 3.8+ |
| Anomaly detection | Scikit-learn (Isolation Forest) |
| Data processing | Pandas, NumPy |
| Cursor capture | Quartz (macOS), Win32API (Windows), Xlib (Linux) |
| Keyboard capture | pynput |
| Feature scaling | StandardScaler |
| Email alerts | smtplib (Gmail SMTP) |
| Configuration | PyYAML |
| Concurrency | Python threading |

## Project Structure

```text
├── capture/                  # Step 1: Raw data collection
│   ├── cursor_capture.py     # 100Hz cursor tracking
│   ├── keyboard_capture.py   # Keystroke timing capture
│   └── idle_detector.py      # Inactivity detection
├── features/                 # Step 2: Feature engineering
│   ├── cursor_features.py    # 12 cursor behavioral features
│   ├── keyboard_features.py  # 9 keyboard behavioral features
│   └── feature_fusion.py     # Combines into 21-feature vectors
├── model/                    # Steps 4 & 8: ML model
│   ├── train_model.py        # Isolation Forest training
│   ├── evaluate_model.py     # Standard evaluation
│   ├── test_with_impostor.py # Formal FAR/FRR evaluation
│   ├── wrapper.py            # AnomalyDetectionWrapper (predict_proba)
│   └── adaptive_update.py    # Adaptive retraining logic
├── decision/                 # Steps 5 & 6: Security decisions
│   ├── inference_engine.py   # Real-time sliding window inference
│   ├── threshold_logic.py    # ALLOW/MONITOR/LOCK decision logic
│   └── lock_controller.py    # Cross-platform system lock
├── alerts/                   # Step 7: Notifications
│   ├── alert_manager.py      # Alert queue with retry logic
│   └── email_alert.py        # Gmail SMTP integration
├── deployment/               # Background service
│   ├── background_service.py # Main multi-threaded service
│   ├── startup_manager.py    # Cross-platform auto-start
│   └── config.yaml           # System configuration
├── utils/                    # Utilities
│   ├── analyze_session.py    # Owner vs impostor behavior comparison
│   ├── detection_timer.py    # Detection timing benchmark
│   ├── logger.py             # Structured logging
│   └── time_utils.py         # Timestamp utilities
├── data/
│   └── sample/               # Synthetic sample data for testing
│       ├── behavior_features_sample.csv   # 70 owner + 30 impostor rows
│       └── generate_sample_data.py        # Reproducible data generator
├── results/
│   ├── evaluation_report_sample.txt       # Sample FAR/FRR output
│   ├── timing_benchmark_sample.txt        # Sample timing output
│   └── behavior_comparison_sample.txt     # Sample feature comparison
└── main.py                   # CLI entry point
```

## Setup & Installation

### Prerequisites

- Python 3.8+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/abhishek14311431/AI-Driven-Continuous-Behavioral-Authentication-System.git
cd AI-Driven-Continuous-Behavioral-Authentication-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
GMAIL_SENDER_EMAIL=your_gmail@gmail.com
GMAIL_APP_PASSWORD=your_16_character_app_password
GMAIL_RECIPIENT_EMAIL=your_recipient@gmail.com
```

> Get a Gmail App Password at: Google Account → Security → 2-Step Verification → App passwords

## Running the Evaluation Pipeline

Run these steps in order to reproduce the formal evaluation results without collecting real data:

### Step 1: Generate Sample Data

```bash
python data/sample/generate_sample_data.py
```

Creates 100 synthetic behavioral samples (70 owner + 30 impostor) with a fixed random seed for reproducibility.

### Step 2: Train the Model

```bash
python main.py --mode train
```

Automatically falls back to sample data if `data/processed/behavior_features.csv` is absent.
Expected output: Train accuracy ~96%, Test accuracy ~93%

### Step 3: Formal Evaluation (FAR / FRR / Accuracy)

```bash
python main.py --mode evaluate
```

Evaluates owner recognition and impostor detection separately.
Expected output: Owner Recognition 95.71%, Impostor Detection 100%, FAR 0%
Results saved to: `results/evaluation_report.txt`

### Step 4: Detection Timing Benchmark

```bash
python utils/detection_timer.py
```

Measures real-world detection time using 2-second sliding window intervals.
Expected output: Average detection time ~2.06 seconds
Results saved to: `results/timing_benchmark.txt`

### Step 5: Behavior Comparison (Owner vs Impostor)

```bash
python utils/analyze_session.py
```

Compares all 21 behavioral features between owner and impostor samples.
Expected output: 18 out of 21 features show >50% difference
Results saved to: `results/behavior_comparison.txt`

## Running as a Live Background Service

### Step 1: Collect Your Own Behavioral Data

Run the service in data-collection mode (no model required):

```bash
python main.py --mode service
```

Use your computer normally for 10–15 minutes. The service captures cursor and keyboard data automatically and saves feature vectors to `data/processed/behavior_features.csv`.

### Step 2: Train on Your Real Data

```bash
python main.py --mode train --features-file data/processed/behavior_features.csv
```

### Step 3: Run Live Authentication

```bash
python main.py --mode service
```

The system now runs continuously, verifying behavior every 2 seconds. If an impostor is detected, the system locks and sends an email alert.

## Key Design Decisions

### Why Isolation Forest?

One-class learning - it only needs to learn the owner's behavior. No impostor data required for training. Anything that deviates from the learned pattern is flagged as anomaly. Lightweight enough to run in the background without slowing down the system.

### Why a Sliding Window (2–5 seconds)?

Single events are noisy and unreliable. Evaluating sequences of behavior over a rolling window eliminates false alarms while maintaining fast detection. Average measured detection time: **2.06 seconds**.

### Why One-Class Instead of Binary Classification?

Collecting real impostor data is impractical and raises privacy concerns. One-class learning solves this - train only on the owner, detect everything else as anomaly.
