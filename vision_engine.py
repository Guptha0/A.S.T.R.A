import cv2
import torch
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from video_stream import VideoStream

class VisionEngine:
    def __init__(self, obj_model_path="yolov8n.pt", pose_model_path="yolov8n-pose.pt", camera_index=0):
        """
        Initializes the YOLO object model, YOLO pose model (for full body), 
        and MediaPipe hands (for precise fingers).
        """
        print("Loading models...")
        self.obj_model = YOLO(obj_model_path)
        self.pose_model = YOLO(pose_model_path)
        
        if torch.cuda.is_available():
            print("CUDA is available, setting YOLO models to GPU.")
            self.obj_model.to('cuda')
            self.pose_model.to('cuda')
        else:
            print("CUDA not available, running YOLO models on CPU.")
        
        # Initialize MediaPipe Hands using Tasks API
        self.mp_model_path = 'hand_landmarker.task'
        base_options = python.BaseOptions(model_asset_path=self.mp_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands_detector = vision.HandLandmarker.create_from_options(options)
        
        # Hand skeleton connections for manual drawing
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        
        self.camera_index = camera_index
        self.cap = None
        self.proxy_classes = ['bottle', 'cup', 'book', 'cell phone']
        
    def draw_corner_brackets(self, frame, x1, y1, x2, y2, color=(0, 255, 0), length=20, thickness=2):
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)

    def draw_landmarks(self, frame, landmarks):
        """Draws hand skeleton connections and joints manually."""
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
        Runs YOLO pose (full body), YOLO object detection, and MediaPipe hand tracking.
        """
        # 1. Run Pose Detection (Full Body Detection)
        pose_results = self.pose_model(frame, verbose=False)
        
        # Use Ultralytics built-in plotter to draw the full body skeleton
        annotated_frame = pose_results[0].plot()
        
        # 2. Run Object Detection
        obj_results = self.obj_model(frame, verbose=False)
        names = self.obj_model.names
        
        obj_boxes = []
        obj_classes = []
        
        for result in obj_results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = names[cls_id]
                
                if class_name in self.proxy_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    obj_boxes.append([x1, y1, x2, y2])
                    obj_classes.append(class_name)
                    
                    # Draw bounding box (Aerospace Corner Brackets)
                    self.draw_corner_brackets(annotated_frame, x1, y1, x2, y2, color=(0, 255, 0), length=20, thickness=2)
                    
                    label = f"{class_name.upper()} [{conf:.2f}]"
                    
                    # Draw a sleek label box just above the brackets
                    (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w_txt + 4, y1 - 2), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, label, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # 3. Run MediaPipe Hands using Tasks API
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.hands_detector.detect(mp_image)
        
        hand_landmarks_list = detection_result.hand_landmarks
        
        if hand_landmarks_list:
            for hand_landmarks in hand_landmarks_list:
                self.draw_landmarks(annotated_frame, hand_landmarks)
                
        return annotated_frame, obj_boxes, obj_classes, hand_landmarks_list

    def start(self):
        """
        Standalone test loop.
        """
        self.cap = VideoStream(self.camera_index).start()
        
        if not self.cap.ret:
            print(f"Error: Could not open camera {self.camera_index}")
            return

        print("Starting Vision Engine. Press 'q' in the video window to quit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue
                
            annotated_frame, obj_boxes, obj_classes, hand_landmarks_list = self.process_frame(frame)

            cv2.imshow("Vision Engine (Full Body + Hands + Objects)", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cleanup()
        
    def cleanup(self):
        if self.cap is not None:
            self.cap.stop()
        cv2.destroyAllWindows()
        if hasattr(self, 'hands_detector'):
            self.hands_detector.close()
        print("Vision Engine stopped and resources released cleanly.")

if __name__ == "__main__":
    engine = VisionEngine()
    engine.start()
