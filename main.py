import cv2
import queue
import time
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from vision_engine import VisionEngine
from voice_assistant import VoiceAssistant
from interaction_logic import InteractionDetector
from sequence_engine import ExperimentSequence
from video_stream import VideoStream

SYSTEM_NAME = "A.S.T.R.A."
WINDOW_TITLE = f"{SYSTEM_NAME} | Astronaut Sequence Tracking & Recognition Assistant"

def main():
    print("====================================================")
    print(" A.S.T.R.A. - On-Board Experiment Validation Engine ")
    print(" Astronaut Sequence Tracking & Recognition Assistant")
    print(" Status: OFFLINE MODE ACTIVE")
    print("====================================================")
    
    # Create dataset_frames directory if it doesn't exist for Data Collection Mode
    dataset_dir = "dataset_frames"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Created directory: {dataset_dir}")
        
    # 1. Instantiate all four modules
    vision = VisionEngine()
    voice = VoiceAssistant(model_path="vosk-model-en-in-0.5")
    detector = InteractionDetector(cooldown_seconds=2.0)
    sequence = ExperimentSequence()
    
    # Global context for the voice assistant to "see" the footage
    current_visible_objects = []
    
    # Thread-safe voice command queue
    command_queue = queue.Queue()
    
    # Voice command handler
    def handle_voice_command(text):
        print(f"[STT] Command received: {text}")
        command_queue.put(text.lower())
            
    # 2. Start Voice Assistant background listening thread
    voice.start_listening(handle_voice_command)
    voice.play_alert("astra_init")
    
    # 3. Initialize VisionEngine webcam loop locally using VideoStream
    cap = VideoStream(vision.camera_index).start()
    if not cap.ret:
        print("Failed to open camera.")
        voice.shutdown()
        return

    print("--- Starting Main Loop ---")
    print("Press 'q' to exit.")
    print("Press 'r' to toggle Data Collection (Recording) Mode.")
    
    # UI and Data Collection state variables
    recording_mode = False
    warning_end_time = 0.0
    
    last_voice_cmd = ""
    voice_cmd_end_time = 0.0
    
    running = True

    while running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
            
        # Process Voice Commands Thread-Safely
        while not command_queue.empty():
            text = command_queue.get()
            if "record" in text or "start" in text:
                if not recording_mode:
                    recording_mode = True
                    print("--- DATA COLLECTION MODE: ON ---")
                    voice.play_alert("recording_started")
                last_voice_cmd = "RECORD/START [EXECUTED]"
                voice_cmd_end_time = time.time() + 2.0
            elif "stop" in text:
                if recording_mode:
                    recording_mode = False
                    print("--- DATA COLLECTION MODE: OFF ---")
                    voice.play_alert("recording_stopped")
                last_voice_cmd = "STOP [EXECUTED]"
                voice_cmd_end_time = time.time() + 2.0
            elif "reset" in text:
                sequence.current_state_idx = 0
                voice.play_alert("sequence_reset")
                last_voice_cmd = "RESET [EXECUTED]"
                voice_cmd_end_time = time.time() + 2.0
            elif "quit" in text or "exit" in text:
                last_voice_cmd = "QUIT [EXECUTED]"
                voice_cmd_end_time = time.time() + 2.0
                voice.play_alert("system_offline")
                time.sleep(1.0)
                running = False
            elif "status" in text:
                current = sequence.get_state()
                print(f"-> Trigger: Status Check ({current})")
                last_voice_cmd = f"STATUS: {current}"
                voice_cmd_end_time = time.time() + 2.0
            
        # CRITICAL DATA INTEGRITY STEP: 
        # Clone the pristine, untouched frame for the dataset before any YOLO or OpenCV drawing occurs.
        clean_frame = frame.copy()
            
        # Run VisionEngine's combined object and hand tracking
        annotated_frame, obj_boxes, obj_classes, hand_landmarks_list = vision.process_frame(frame)
        
        h, w, _ = frame.shape
                    
        # Update global context for the voice assistant
        current_visible_objects = obj_classes
                    
        # 4. Pass data to InteractionDetector
        events = detector.detect_interaction(hand_landmarks_list, obj_boxes, obj_classes, w, h)
        
        # 5. Feed resulting actions into ExperimentSequence
        for event in events:
            status, msg = sequence.update_state(event)
            print(f"[{event}] -> {msg}")
            
            # Voice Feedback and Visual Warning Logic
            if status == "success":
                target = event.replace("grabbed_", "")
                print(f"{SYSTEM_NAME} protocol confirmed. You have successfully grabbed the {target}.")
                if sequence.get_state() == "PROTOCOL_COMPLETE":
                    voice.play_alert("protocol_complete")
                else:
                    voice.play_alert("step_verified")
            elif status == "violation":
                print(f"{SYSTEM_NAME} warning: invalid step detected. You triggered {event.replace('_', ' ')}. Please follow the protocol.")
                voice.play_alert("warning_violation")
                # Set a 2-second timestamp for the HUD visual warning
                warning_end_time = time.time() + 2.0

        # --- 6. HUD OVERLAYS (PIL Sci-Fi Style) ---
        # Convert annotated_frame to PIL Image for TrueType font rendering and alpha blending
        pil_img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        
        # Create an alpha layer for the semi-transparent banner
        overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Draw semi-transparent black rectangle (60% opacity = ~153 alpha) at the top
        frame_h, frame_w = annotated_frame.shape[:2]
        banner_height = 80
        draw.rectangle(((0, 0), (frame_w, banner_height)), fill=(0, 0, 0, 153))
        
        # Draw state text using a TrueType font
        try:
            # Consolas is standard on Windows
            font_title = ImageFont.truetype("consola.ttf", 22)
            font_state = ImageFont.truetype("consola.ttf", 28)
            font_sub = ImageFont.truetype("consola.ttf", 16)
        except IOError:
            font_title = font_state = font_sub = ImageFont.load_default()
            
        draw.text((20, 10), f"{SYSTEM_NAME} v1.0 // ON-BOARD HAR ASSISTANT", font=font_title, fill=(0, 255, 255, 255))
        draw.text((20, 40), f"EXPERIMENT STATE: {sequence.get_state().upper()}", font=font_state, fill=(255, 255, 255, 255))
        draw.text((frame_w - 350, 15), "SYS: ACTIVE | PROTOCOL ENFORCEMENT ON", font=font_sub, fill=(0, 255, 0, 255))
        
        if time.time() < voice_cmd_end_time and last_voice_cmd:
            draw.text((20, 65), f"VOICE CMD: {last_voice_cmd}", font=font_sub, fill=(255, 255, 0, 255))
            
        if recording_mode:
            try:
                font_small = ImageFont.truetype("consola.ttf", 20)
            except IOError:
                font_small = ImageFont.load_default()
            draw.text((frame_w - 200, 45), "[REC MODE ACTIVE]", font=font_small, fill=(255, 0, 0, 255))
            
            # Save the clean (untouched) frame
            timestamp = str(time.time()).replace('.', '_')
            filename = os.path.join(dataset_dir, f"frame_{timestamp}.jpg")
            cv2.imwrite(filename, clean_frame)
            
        if time.time() < warning_end_time:
            # Flash warning on screen (Center Screen)
            try:
                font_warn = ImageFont.truetype("consola.ttf", 36)
            except IOError:
                font_warn = ImageFont.load_default()
            warning_txt = "WARNING: PROTOCOL ERROR"
            
            # Note: getbbox returns (left, top, right, bottom)
            text_bbox = draw.textbbox((0, 0), warning_txt, font=font_warn)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            text_x = (frame_w - text_w) // 2
            text_y = (frame_h - text_h) // 2
            
            draw.text((text_x, text_y), warning_txt, font=font_warn, fill=(255, 0, 0, 255))
            
        # Composite the overlay onto the original image
        pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay)
        
        # Convert back to OpenCV format
        annotated_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)

        # Show the live feed
        cv2.imshow(WINDOW_TITLE, annotated_frame)
        
        # --- 7. Input Handling ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Exiting loop...")
            voice.play_alert("system_offline")
            time.sleep(1.0)
            running = False
        elif key == ord('r'):
            recording_mode = not recording_mode
            if recording_mode:
                print("--- DATA COLLECTION MODE: ON ---")
                voice.play_alert("recording_started")
            else:
                print("--- DATA COLLECTION MODE: OFF ---")
                voice.play_alert("recording_stopped")
            
    # 8. Graceful Exit
    cap.stop()
    cv2.destroyAllWindows()
    voice.shutdown()
    
    # Brief pause to ensure audio thread dies cleanly
    time.sleep(0.5)
    print("--- System shutdown complete ---")

if __name__ == "__main__":
    main()
