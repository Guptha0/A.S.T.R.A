import os
import json
import threading
import pyaudio
import pygame
from vosk import Model, KaldiRecognizer

class VoiceAssistant:
    def __init__(self, model_path="vosk-model-en-in-0.5"):
        self.model_path = model_path
        self.is_listening = False
        self.listen_thread = None
        pygame.mixer.init()

    def _play_audio_thread(self, filepath):
        if os.path.exists(filepath):
            try:
                sound = pygame.mixer.Sound(filepath)
                sound.play()
                pygame.time.wait(int(sound.get_length() * 1000) + 100)
            except Exception as e:
                print(f"Error playing {filepath}: {e}")
        else:
            print(f"Warning: Audio cue {filepath} not found.")

    def play_alert(self, alert_name):
        """
        Spawns a short-lived threading.Thread to play the audio file.
        Returns instantly to the main loop.
        """
        filepaths = [
            os.path.join("F:\\NEWMODEL\\audio", f"{alert_name}.mp3"),
            os.path.join("F:\\NEWMODEL\\audio", f"{alert_name}.wav"),
            os.path.join(os.path.dirname(__file__), "audio", f"{alert_name}.mp3"),
            os.path.join(os.path.dirname(__file__), "audio", f"{alert_name}.wav"),
            os.path.join(os.path.dirname(__file__), "assets", "audio", f"{alert_name}.mp3"),
            os.path.join(os.path.dirname(__file__), "assets", "audio", f"{alert_name}.wav"),
        ]
        
        target_path = filepaths[-1]
        for fp in filepaths:
            if os.path.exists(fp):
                target_path = fp
                break
                
        t = threading.Thread(target=self._play_audio_thread, args=(target_path,), daemon=True)
        t.start()

    def start_listening(self, callback):
        if not os.path.exists(self.model_path):
            print(f"Error: Vosk model not found at '{self.model_path}'.")
            return

        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop, args=(callback,), daemon=True)
        self.listen_thread.start()
        print("VoiceAssistant is now listening in the background...")

    def stop_listening(self):
        self.is_listening = False

    def _listen_loop(self, callback):
        model = Model(self.model_path)
        grammar = ["record", "start", "stop", "reset", "quit", "exit", "astra", "status", "[unk]"]
        recognizer = KaldiRecognizer(model, 16000, json.dumps(grammar))
        
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        stream.start_stream()

        while self.is_listening:
            data = stream.read(4000, exception_on_overflow=False)
            if len(data) == 0:
                continue
                
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    callback(text)

        stream.stop_stream()
        stream.close()
        audio.terminate()
        
    def shutdown(self):
        self.stop_listening()
        if self.listen_thread and self.listen_thread.is_alive():
            if threading.current_thread() != self.listen_thread:
                self.listen_thread.join(timeout=2.0)
        pygame.mixer.quit()
