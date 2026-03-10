import cv2
import numpy as np
from ultralytics import YOLO
import time
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

class JosephApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        
        # Display area
        self.img1 = Image()
        self.layout.add_widget(self.img1)
        
        # Information label
        self.info_label = Label(text="JosephApp: Ready", size_hint_y=0.1)
        self.layout.add_widget(self.info_label)
        
        # Load YOLO model
        self.model = YOLO('yolov8n.pt')
        self.cap = cv2.VideoCapture(0)
        
        # Variables
        self.ball_points = []
        self.frame_times = []
        self.PITCH_LENGTH = 20.12
        self.CREASE_Y = 450
        
        # Schedule frame updates
        Clock.schedule_interval(self.update, 1.0/30.0)
        return self.layout

    def update(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            return

        h, w, _ = frame.shape
        
        # 1. Ball Detection
        results = self.model.predict(frame, conf=0.3, classes=[32], verbose=False)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                center = (int((x1+x2)/2), int((y1+y2)/2))
                self.ball_points.append(center)
                self.frame_times.append(time.time())
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)

        # 2. Trajectory & Speed
        for i in range(1, len(self.ball_points)):
            cv2.line(frame, self.ball_points[i-1], self.ball_points[i], (0, 255, 255), 2)

        speed_text = "Speed: 0 km/h"
        if len(self.ball_points) > 5:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                speed = (self.PITCH_LENGTH / time_diff) * 3.6
                speed_text = f"Speed: {int(speed)} km/h"
        
        # 3. No-Ball & LBW Logic
        cv2.line(frame, (0, self.CREASE_Y), (w, self.CREASE_Y), (0, 0, 255), 2)
        
        # Update UI
        self.info_label.text = speed_text
        
        # Convert OpenCV frame to Kivy texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tobytes()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img1.texture = texture1

if __name__ == '__main__':
    JosephApp().run()
