import wave
import json
from vosk import Model, KaldiRecognizer

# Vosk 영어 모델 로드
model = Model("vosk-model-small-en-us-0.15")

# WAV 파일 로드
wf = wave.open("your_audio_file.wav", "rb")

# 인식기 생성 (WAV 파일의 샘플링 속도 설정)
rec = KaldiRecognizer(model, wf.getframerate())

# 음성 인식 실행
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        result = rec.Result()
        print(json.loads(result))

# 마지막 결과 출력
print(json.loads(rec.FinalResult()))