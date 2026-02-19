# 🔐 AI Behavior Authentication System

A state-of-the-art continuous authentication system that uses machine learning to identify users based on their unique behavioral "fingerprint"—the way they move their mouse and type on their keyboard.

## 🚀 Key Features

- **Continuous Authentication**: Monitors behavior every 2-5 seconds for uninterrupted security.
- **Privacy First**: All data collection and model training happen locally on your machine.
- **AI-Powered Detection**: Uses Isolation Forest (Anomaly Detection) to learn *your* patterns and flag anyone else.
- **Automatic Protection**: Seamlessly locks the system when a behavior mismatch is detected.
- **Real-Time Alerts**: Supports email and WhatsApp notifications for security events.
- **Background Service**: Operates silently with minimal system resource usage.

## 📂 Project Structure

```text
ai_behavior_auth/
├── alerts/               # Notification managers (Email, WhatsApp)
├── capture/              # Input capture modules (Mouse, Keyboard, Idle)
├── data/                 # Local data storage for behavior patterns
├── decision/             # Real-time inference and security logic
├── deployment/           # Service management and configuration
├── features/             # Advanced feature engineering pipeline
├── model/                # ML model architecture and training
├── utils/                # Logging and system utilities
└── main.py              # Central entry point
```

## 🛠️ Quick Start

### 1. Installation
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Collection
The system needs to learn your behavior first. Use your computer normally for a few minutes while the service is running in collection mode:
```bash
python main.py --mode service
```

### 3. Training
Once you've used the system enough (100+ samples), train your personal model:
```bash
python main.py --mode train
```

### 4. Continuous Protection
Launch the service to enable full behavioral protection:
```bash
python main.py --mode service
```

## 🧪 Testing the System

### Verify Owner Detection
1. Start the service.
2. Use the computer normally.
3. Check `logs/behavior_auth.log`. You should see `ALLOW` decisions with high confidence.

### Verify Attacker Detection (Caution: Will Lock System)
1. Have someone else use your computer.
2. The system should detect the behavioral anomaly.
3. Your screen will automatically lock within seconds.

## ⚙️ Configuration
Customize the system behavior in `deployment/config.yaml`:
- **Security Thresholds**: Adjust how sensitive the detection is (Allow/Monitor/Lock).
- **Lock Settings**: Enable or disable the automatic locking mechanism.
- **Alerts**: Configure your notification preferences.

## 📜 License & Disclaimer
This project is for educational and security research purposes. Always comply with local privacy regulations.

---
**Secure your workspace with the power of AI.** 🛡️
