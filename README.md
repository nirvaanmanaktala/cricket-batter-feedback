# Cricket Batsman Shot Feedback Generator

This project uses computer vision and Google Gemini AI to analyze a cricket batsman's shot in real-time. It captures the player's body posture, tracks the ball, detects swing motion, and generates **actionable feedback like a virtual cricket coach**.

---

# Features

- Real-time pose tracking using **MediaPipe**
- Ball detection using **YOLOv8**
- Shot feedback using **Gemini AI**
- Swing detection logic tailored for cricket batting
- Visual feedback overlay on webcam feed

---

## 📦 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/cricket-batsman-feedback.git
cd cricket-batsman-feedback
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Add your Gemini API key
Create a .env file in the root directory with:

ini
Copy
Edit
GEMINI_API_KEY=your-gemini-api-key-here
You can get the key from: https://aistudio.google.com/app/apikey

Run the Application
bash
Copy
Edit
python main.py
Your webcam will open, and the analyzer will start tracking your pose and swing.

Make a batting motion (simulate a stroke).

The system detects it and sends the pose+ball data to Gemini.

Gemini returns 2-line feedback which is shown on screen.

How It Works
Pose Detection: MediaPipe identifies joints like shoulders, wrists, hips, knees, etc.

Ball Detection: YOLOv8 detects the cricket ball in the frame.

Motion Detection: Wrist movement across X-coordinates is used to detect bat swing.

AI Feedback: Pose + ball data is sent to Gemini, which gives feedback as a cricket coach.

 File Structure
bash
Copy
Edit
cricket-batsman-feedback/
├── main.py                # Main application file
├── yolov8n.pt             # YOLOv8 weights (you must download or train)
├── .env                   # Stores your Gemini API key
├── requirements.txt       # All dependencies
├── .gitignore             # Ignore virtual env, .env, cache, etc.
└── README.md              # This file


Safety & Privacy
Your .env file is ignored by Git (thanks to .gitignore)

No data is stored or uploaded; it's purely local and API call-based

License
MIT License – you are free to use, modify, and share this project. Please give credit.

Acknowledgements
MediaPipe

Ultralytics YOLOv8

Google Gemini AI

@rohtumm inspiration
Built by Nirvaan Manaktala

