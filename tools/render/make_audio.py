"""Voice the sim's speech log: speech.json (tick, name, text) -> one mixed WAV
aligned to the rendered frame timeline, one piper voice per speaker.

Usage: python make_audio.py <frames_dir> <out_wav> <fps>
"""
import json
import os
import subprocess
import sys
import wave

import numpy as np

FRAMES, OUT, FPS = sys.argv[1], sys.argv[2], float(sys.argv[3])
SP = os.path.dirname(os.path.abspath(__file__))
VOICES = {}          # speaker -> onnx, assigned round-robin from what we have
AVAILABLE = [os.path.join(SP, "voices", v) for v in
             ("en_US-ryan-medium.onnx", "en_US-lessac-medium.onnx")]

meta = json.load(open(os.path.join(FRAMES, "speech.json")))
saved = sorted(int(f[1:5]) for f in os.listdir(FRAMES) if f.endswith(".npz"))
RATE = 22050
clips = []
for tick, name, text in meta["speech"]:
    if name not in VOICES:
        VOICES[name] = AVAILABLE[len(VOICES) % len(AVAILABLE)]
    frame_i = next((i for i, s in enumerate(saved) if s >= tick), len(saved) - 1)
    t0 = frame_i / FPS
    wav_path = os.path.join(FRAMES, f"say_{tick}_{name}.wav")
    subprocess.run([sys.executable, "-m", "piper", "-m", VOICES[name],
                    "-f", wav_path, "--", text], check=True,
                   capture_output=True)
    with wave.open(wav_path) as wf:
        assert wf.getframerate() == RATE, wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), np.int16)
    clips.append((t0, data))
    print(f"{name} @ {t0:.2f}s: {len(data)/RATE:.2f}s of speech")

total = max(t0 + len(d) / RATE for t0, d in clips) + 0.5
master = np.zeros(int(total * RATE), np.float32)
for t0, d in clips:
    i = int(t0 * RATE)
    master[i:i + len(d)] += d.astype(np.float32)
master = np.clip(master, -32767, 32767).astype(np.int16)
with wave.open(OUT, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(RATE)
    wf.writeframes(master.tobytes())
print("mixed ->", OUT, f"({total:.1f}s)")
