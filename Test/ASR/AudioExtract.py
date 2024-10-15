########## Project Description ##########
# This is a python script to extract audiofile from Video.
# Input: Path of the video file
# Output: Path of the audio file
# Required Library: os(pre-installed)
# BUILD: Oct 15, 2024 (KST)
##########################################

########## Change Log ##########
# Oct 15, 2024 (KST)
# Using ffmpeg to extract wav format audio file
###############################

import os


class audioExtractor:
    def __init__(self, video_file_path, audio_file_path):
        self.video_file_path = video_file_path
        self.audio_file_path = audio_file_path

    def extract_audio(self):
        command = f'ffmpeg -i "{self.video_file_path}" -acodec pcm_s16le -ac 1 -ar 16000 "{self.audio_file_path}"'
        os.system(command)