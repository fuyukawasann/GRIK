########## Project Description ##########
# This is a python script to extract script from Audio.
# Input: Path of the audio file
# Output: Path of the audio file
# Required Library: AudioExtract.py, sys, jetson_voice
# BUILD: Oct 15, 2024 (KST)
##########################################

########## Change Log ##########
# Oct 15, 2024 (KST)
# Initial Build
###############################


import sys
from AudioExtract import audioExtractor as ae
from QuartzNet import QuartzNet
from CTCBeamSearchDecorder import CTCBeamSearchDecorder
import torchaudio

class ASR:
    def __init__(self, video_file_path, audio_file_path):
        self.video_file_path = video_file_path
        self.audio_file_path = audio_file_path

    def extract_and_recognize(self):
        # Make AudioExtractor Object
        print("AudioExtract ...")
        ae_obj = ae(self.video_file_path, self.audio_file_path)
        ae.extract_audio()
        print("AudioExtract ... DONE")

	
        print("ASR...")
        audio, sr = torchaudio.load(self.audio_file_path)

        model = QuartzNet()
        decoder = CTCBeamSearchDecorder(num_classes=29)

        logits = model(audio)
        results = decoder(logits)
        print("ASR... DONE")

        return results
