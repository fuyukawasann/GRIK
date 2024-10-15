import sys
import scipy.io.wavfile as wav
import deepspeech

model_file_path = '~/Desktop/graduation project/deepspeech/deepspeech-0.5.1-models.pbmm'
alphabet_file_path = '~/Desktop/graduation project/deepspeech/alphabet.txt'
lm_file_path = '~/Desktop/graduation project/deepspeech/lm.binary'
trie_file_path = '~/Desktop/graduation project/deepspeech/trie'

model = deepspeech.Model(model_file_path, 500)

model.enableDecoderWithLM(lm_file_path, trie_file_path, 0.75, 1.85)


fs, audio = wav.read('demo.wav')

text = model.stt(audio)

print("Recognized TEXT:", text)
