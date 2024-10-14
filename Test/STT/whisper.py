from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa

# 모델 및 프로세서 로드
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# WAV 파일 로드 및 전처리
audio, rate = librosa.load("demo.wav", sr=16000)
inputs = processor(audio, return_tensors="pt", sampling_rate=rate)

# 추론 실행
with torch.no_grad():
    predicted_ids = model.generate(inputs.input_features)

# 텍스트 디코딩
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)