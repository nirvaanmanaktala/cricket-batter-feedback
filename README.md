# Cricket Batting Analyzer using Gemini, YOLO and MediaPipe

This project analyzes a batsman's cricket shot using MediaPipe for pose tracking, YOLOv8 for ball detection, and Google Gemini AI API for generating coaching feedback. It runs in real time using your webcam and shows suggestions based on balance, bat swing, footwork and timing.
Currently only functional for right handers
## Features

- The project offers real-time pose tracking using MediaPipe
- Cricket ball detection using YOLOv8
- AI-generated coaching tips using Gemini 2.0 Flash
- Feedback includes:
  - Head and body alignment
  - Foot movement
  - Bat lift and follow-through
  - Release and impact timing
- Detects swing and automatically displays improvement tips.
## Requirements
- Python 3.10 recommended (MediaPipe may not work with python 3.12)
- A working webcam
- macOS (not tested on windows and linux)
## Installation
Instructions are given in reference to macOS
### 1.Clone this:
```bash
git clone https://github.com/yourusername/cricket-batting-analyzer.git
cd cricket-batting-analyzer
```
### 2.create a virtual environment/venv if 1. doesnt work
```bash
python3.10 -m venv venv
source venv/bin/activate    
```
### 3.Install dependencies
```bash
pip install opencv-python mediapipe google-generativeai python-dotenv ultralytics
```
## Setup .env file for Gemini API
1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Copy your Gemini API key
3. Create a file called `.env` in the project root folder and add-
```
GEMINI_API_KEY=api_key(do not share)
```
4. ## Download YOLOv8 Model

This project uses the YOLOv8 Nano model. Download it from:

[https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)
Place `yolov8n.pt` in the same folder as `main.py`.
## Running the App
```bash
python main.py
```
- The webcam feed will open.
- When a swing motion is detected, the system sends the pose + ball data to Gemini.
- Gemini gives a 2-sentence summary with one improvement suggestion.
- Feedback stays visible for a few seconds on the video.
Press `q` to exit the application.
## Project Structure

```
cricket-batting-analyzer/
│
├── main.py                   # Main application script
├── .env                      # Your Gemini API key (not pushed to GitHub)
├── yolov8n.pt                # YOLOv8 nano model for object detection
├── requirements.txt          # (Optional) pip freeze output
└── README.md
```

## Notes
- Make sure you place the `yolov8n.pt` file in the same directory as `main.py`.
- You can use a custom YOLO model trained for cricket balls if needed.
- Assumes a right-handed batsman for now


## License
This project is licensed under the MIT License.


