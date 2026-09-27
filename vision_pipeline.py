import cv2
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class VisionPipeline:
    """
    Vision Pipeline for offline Human Activity Recognition (HAR).
    Captures live webcam feed, processes hand landmarks using MediaPipe,
    and draws static placeholder zones for future YOLO integration.
    """
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        # Initialize Camera
        self.cap = cv2.VideoCapture(self.camera_id)
        
        # Download task model if it doesn't exist to ensure offline capability
        self.model_path = 'hand_landmarker.task'
        if not os.path.exists(self.model_path):
            print("Downloading MediaPipe hand landmarker model (one-time setup)...")
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete!")

        # Initialize MediaPipe Hand Landmarker (Tasks API)
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Define 3 completely separate, non-overlapping boxes placed side-by-side
        self.boxes = {
            "Main Box": {"rect": (30, 150, 150, 150), "color": (255, 255, 255)},    # Left
            "Red Box": {"rect": (245, 150, 150, 150), "color": (0, 0, 255)},        # Center
            "Yellow Box": {"rect": (460, 150, 150, 150), "color": (0, 255, 255)}    # Right
        }

        # Hand skeleton connections
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

    def draw_placeholders(self, frame):
        """Draws the static placeholder bounding boxes on the frame."""
        for label, data in self.boxes.items():
            x, y, w, h = data["rect"]
            color = data["color"]
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # Add text label
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw_landmarks(self, frame, landmarks):
        """Draws hand skeleton connections and joints manually without mp.solutions."""
        h, w, _ = frame.shape
        # Draw connections
        for connection in self.hand_connections:
            idx1, idx2 = connection
            lm1, lm2 = landmarks[idx1], landmarks[idx2]
            
            x1, y1 = int(lm1.x * w), int(lm1.y * h)
            x2, y2 = int(lm2.x * w), int(lm2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        # Draw joints
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), cv2.FILLED)

    def process_frame(self, frame):
        """
        Processes a single frame: detects hands, draws landmarks.
        Returns the annotated frame and the list of detected hand landmarks.
        This method is designed to be called by an external state machine.
        """
        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands using Tasks API
        detection_result = self.detector.detect(mp_image)
        
        # Draw landmarks if any hands are detected
        hand_landmarks_list = detection_result.hand_landmarks
        for hand_landmarks in hand_landmarks_list:
            self.draw_landmarks(frame, hand_landmarks)
                
        return frame, hand_landmarks_list

    def run(self):
        """
        Starts the standalone camera loop. Displays the UI natively via OpenCV.
        """
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Starting Vision Pipeline. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame from camera.")
                    break
                
                # Flip frame horizontally for a natural mirror effect
                frame = cv2.flip(frame, 1)

                # Process hand landmarks
                frame, landmarks = self.process_frame(frame)

                # Draw the YOLO placeholder zones
                self.draw_placeholders(frame)

                # Display the live feed in a native OpenCV window
                cv2.imshow('Space Mission Assistant - Vision Pipeline', frame)

                # Clean exit condition on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting Vision Pipeline...")
                    break
        finally:
            self.release()

    def release(self):
        """Releases hardware resources and destroys windows."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'detector'):
            self.detector.close()

if __name__ == "__main__":
    # Entry point for standalone testing
    pipeline = VisionPipeline()
    pipeline.run()
