import vosk
import pyaudio
import wave

model = vosk.Model("vosk-model-small-en-us-0.15")



wf = wave.open("demo.wav", "rb")


p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
		channels = wf.getnchannels(),
		rate=wf.getframerate(),
		input=True,
		frames_per_buffer=1024)

recognizer = vosk.KaldiRecognizer(model, wf.getframmerate())

data = wf.readframes(1024)

while data:
    if recognizer.AcceptWaveform(data):
        print(recognizer.Result())
    data = wf.readframes(1024)

stream.stop_stream()
stream.close()
p.terminate()

