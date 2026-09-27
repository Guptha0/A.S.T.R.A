import time

class InteractionDetector:
    def __init__(self, cooldown_seconds=2.0):
        """
        Initializes the InteractionDetector module.
        Tracks the cooldown per object class to avoid spamming interaction events.
        """
        self.cooldown_seconds = cooldown_seconds
        self.last_interaction_time = {}
        
        # MediaPipe Landmark Indices for precise fingertip tracking
        self.INDEX_FINGER_TIP = 8
        self.THUMB_TIP = 4
        
    def extract_fingertip_keypoints(self, hand_landmarks_list, frame_width, frame_height):
        """
        Extracts index and thumb tips from mediapipe hand landmarks.
        Returns a list of (x, y) pixel coordinates.
        """
        tips = []
        if hand_landmarks_list:
            for hand_landmarks in hand_landmarks_list:
                # With the Tasks API, hand_landmarks is a list of NormalizedLandmark objects directly
                # Get the index tip
                index_lm = hand_landmarks[self.INDEX_FINGER_TIP]
                tips.append((int(index_lm.x * frame_width), int(index_lm.y * frame_height)))
                
                # Get the thumb tip
                thumb_lm = hand_landmarks[self.THUMB_TIP]
                tips.append((int(thumb_lm.x * frame_width), int(thumb_lm.y * frame_height)))
                
        return tips

    def check_intersection(self, point, box):
        """
        Evaluates if the (x, y) coordinates of a point fall within the [x1, y1, x2, y2] box.
        """
        px, py = point
        x1, y1, x2, y2 = box
        
        return x1 <= px <= x2 and y1 <= py <= y2

    def detect_interaction(self, hand_landmarks_list, obj_boxes, obj_classes, frame_width, frame_height):
        """
        Detects interactions between human fingertips and objects.
        
        hand_landmarks_list: List of mediapipe hand landmark objects
        obj_boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        obj_classes: List of corresponding class names (e.g., ['bottle', 'cup'])
        frame_width: Int, width of the frame for scaling normalized coordinates
        frame_height: Int, height of the frame for scaling normalized coordinates
        
        Returns a list of detected event strings (e.g., ["grabbed_bottle"]).
        """
        events = []
        current_time = time.time()
        
        fingertips = self.extract_fingertip_keypoints(hand_landmarks_list, frame_width, frame_height)
        
        for box, obj_class in zip(obj_boxes, obj_classes):
            for tip in fingertips:
                if self.check_intersection(tip, box):
                    # Enforce debouncing/cooldown tracker
                    last_time = self.last_interaction_time.get(obj_class, 0.0)
                    
                    if current_time - last_time >= self.cooldown_seconds:
                        self.last_interaction_time[obj_class] = current_time
                        events.append(f"grabbed_{obj_class}")
                        
                    # Once a valid interaction logic fires for this box, stop checking other tips
                    break 
                    
        return events
