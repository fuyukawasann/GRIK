import os
import vosk
import sys
import wave


### Download the video
os.system('gdown https://drive.google.com/uc?id=1AyxtBXOQTw2Ya2-EOCQ6ZvM9UdeBnqC2')

### Convert mp4 to wav
os.system('ffmpeg -i "DEMO(ENG).mp4" -acodec pcm_s16le -ac 1 -ar 16000 "DEMO.wav"')


model = vosk.Model("vosk-model-small-en-us-0.15")


wf = wave.open("DEMO.wav", "rb")

if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
    print("Audio File Format must be a 16khz mono WAV")
    sys.exit(1)

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        print(rec.Result())
    else:
        print(rec.PartialResult())

print(rec.FinalResult())

