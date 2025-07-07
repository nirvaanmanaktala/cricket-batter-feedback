# Cricket Batsman Shot Feedback Generator

This project uses computer vision and Google Gemini AI to analyze a cricket batsman's shot in real-time. It captures the player's body posture, tracks the ball, detects swing motion, and generates **actionable feedback like a virtual cricket coach**.

---

# Features

- Real-time pose tracking using **MediaPipe**
- Ball detection using **YOLOv8**
- Shot feedback using **Gemini AI**
- Swing detection logic tailored for cricket batting
- Visual feedback overlay on webcam feed


Make a batting motion (simulate a cricket shot).
The system detects the swing and sends pose + ball data to Gemini. Gemini returns 2-line feedback which appears on the screen.
