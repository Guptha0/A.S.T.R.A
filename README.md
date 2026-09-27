# A.S.T.R.A. (Astronaut Sequence Tracking & Recognition Assistant)

**Mission Statement:** Fully offline, GPU-accelerated Human Activity Recognition (HAR) for protocol enforcement in microgravity orbital environments.

A.S.T.R.A. is an edge-based AI system engineered for orbital habitats. It provides astronauts with ambient, real-time computer vision and auditory feedback to ensure strict adherence to mission-critical scientific procedures without reliance on cloud telemetry.

---

## 🚀 Core Features
- **Deterministic Validation:** Tracks complex multi-step protocols via an autonomous finite state machine.
- **Zero-Cloud Dependency:** Fully offline STT (Speech-To-Text) and vision models for maximum security and network resilience.
- **Gravity-Agnostic Spatial Tracking:** Utilizes robust fingertip keypoint tracking instead of standard posture tracking to mitigate microgravity orientation bias.
- **Hands-Free Control:** Seamless dual-trigger voice ("start", "stop", "reset", "quit") and keyboard operation.

---

## 🛠 Tech Stack
- **Core:** Python 3
- **Vision:** Ultralytics YOLOv8 (Object Localization), MediaPipe Hands (21-Keypoint Tracking)
- **Audio:** Vosk (`en-in-0.5` offline STT Kaldi recognizer), PyGame (Non-blocking audio dispatcher)
- **Acceleration:** PyTorch (CUDA GPU Offloading)

---

## ⚙️ Setup & Installation

To comply with GitHub repository data limits, large AI models and audio assets are not included in this repository. You must manually provision them before running the system.

### 1. Model & Asset Provisioning
1. **YOLOv8 Weights:** Download the standard [yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) weights and place the file directly in the project root folder.
2. **Vosk STT Model:** Download the English (Indian) Vosk model `vosk-model-en-in-0.5` (or your preferred language) from the official [Vosk Models page](https://alphacephei.com/vosk/models). Extract the folder and place it directly in the project root.
3. **Audio Alerts:** Generate the required voice alerts (e.g., `astra_init.mp3`, `recording_started.mp3`) as specified in the codebase, or generate them using a TTS service. Place these `.mp3` files inside the `assets/audio/` (or `audio/`) directory.

### 2. Environment Setup
Clone the repository and set up your Python environment:

```bash
# Create and activate a virtual environment
python -m venv venv

# Windows Activation:
venv\Scripts\activate
# Linux/Mac Activation:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*(Note: Ensure you have installed the GPU-enabled version of PyTorch corresponding to your CUDA toolkit version for optimal performance).*

### 3. Execution
Launch the primary sequence engine:
```bash
python main.py
```

## 🎮 Controls
* **Voice:** Say `"record"`, `"stop"`, `"reset"`, or `"quit"`.
* **Keyboard:** Press `r` to toggle data collection mode, `q` to gracefully quit the program.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
