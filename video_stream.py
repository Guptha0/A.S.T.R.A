import cv2
import threading

class VideoStream:
    def __init__(self, src=0):
        self.src = src
        self.stream = None
        self.ret = False
        self.frame = None
        
        self.stopped = False
        self.lock = threading.Lock()
        self.started_event = threading.Event()
        
    def start(self):
        # Start the background thread
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        
        # Block the main thread until the camera is fully initialized inside the background thread
        self.started_event.wait()
        return self
        
    def update(self):
        # IMPORTANT: Initialize cv2.VideoCapture inside the thread that reads from it!
        # On Windows, cv2 uses COM objects (DirectShow/Media Foundation) which can hang 
        # if created in the main thread and polled in a background thread.
        self.stream = cv2.VideoCapture(self.src)
        
        if not self.stream.isOpened():
            with self.lock:
                self.ret = False
                self.frame = None
            self.started_event.set()
            return
            
        # Grab the initial frame
        ret, frame = self.stream.read()
        with self.lock:
            self.ret = ret
            self.frame = frame
            
        # Signal the main thread that we are ready
        self.started_event.set()
        
        # Keep looping infinitely until the thread is stopped
        while not self.stopped:
            ret, frame = self.stream.read()
            
            with self.lock:
                self.ret = ret
                self.frame = frame
                
    def read(self):
        # Return the frame most recently read
        with self.lock:
            frame_copy = self.frame.copy() if self.frame is not None else None
            return self.ret, frame_copy
            
    def stop(self):
        # Indicate that the thread should be stopped
        self.stopped = True
        
        # Wait until stream resources are released
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        if self.stream is not None:
            self.stream.release()
