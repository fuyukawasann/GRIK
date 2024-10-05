# you must install whisper and ffmpeg, pytubefix
#######
# Initial Build #
#  BUILD: 2024.10.05


from pytubefix import YouTube
from pytubefix.cli import on_progress

url = "https://youtu.be/bZffNkYobYg?si=GW7CFHbM_1E1cBi7"

yt = YouTube(url, on_progress_callback=on_progress)

ys = yt.streams.get_highest_resolution()
ys.download()
