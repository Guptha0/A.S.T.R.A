import os
import torch
import numpy as np
import wave

def save_wav(filename, audio_numpy, sample_rate):
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2) # 16-bit
        f.setframerate(sample_rate)
        audio_int16 = (audio_numpy * 32767.0).astype(np.int16)
        f.writeframes(audio_int16.tobytes())

def generate_cues():
    audio_dir = os.path.join(os.path.dirname(__file__), "assets", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    prompts = {
        "astra_init.wav": "A.S.T.R.A. initialized. Sequence tracking online.",
        "step_verified.wav": "Step verified.",
        "warning_violation.wav": "Warning: Protocol violation detected.",
        "protocol_complete.wav": "Experiment sequence completed.",
        "recording_started.wav": "Data recording started.",
        "recording_stopped.wav": "Data recording stopped.",
        "sequence_reset.wav": "Sequence reset to IDLE.",
        "system_offline.wav": "System offline. Goodbye."
    }

    print("Loading Silero TTS model...")
    device = torch.device('cpu')
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='en',
        speaker='v3_en'
    )
    model.to(device)
    sample_rate = 24000
    
    for filename, text in prompts.items():
        filepath = os.path.join(audio_dir, filename)
        if not os.path.exists(filepath):
            print(f"Generating {filename}...")
            audio = model.apply_tts(
                text=text,
                speaker='en_0',
                sample_rate=sample_rate
            )
            audio_np = audio.numpy()
            save_wav(filepath, audio_np, sample_rate)
            print(f"Saved {filepath}")
        else:
            print(f"Skipping {filename}, already exists.")

if __name__ == "__main__":
    generate_cues()
