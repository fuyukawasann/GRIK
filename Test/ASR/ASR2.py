import os
import subprocess
from AudioExtract import audioExtractor as ae

def run_jetson_voice():





    # Step 1: Clone the jetson-voice repository
    print("Clone GITHUB Repository...")
    subprocess.run(['git', 'clone', 'https://github.com/dusty-nv/jetson-voice.git'])
    print("Clone GITHUB Repository... DONE")

    # Step 2: Navigate to the cloned directory
    print("Change Directory...")
    os.chdir('jetson-voice')
    print("Change Directory... Done")

    # Step 3: Build the Docker Image
    print("Build docker Image...")
    subprocess.run(['sudo', 'docker', 'build', '-t', 'jetson_voice', '-f', 'Dockerfile.aarch64', '.'])
    print("Build docker Image... Done")



    os.system('gdown https://drive.google.com/uc?id=1rHIBqX3mA-gztt8wu0Wur7cHlzcS8sSE')
    print("MP4 -> WAV...")
    ae_obj = ae("DEMO(ENG).mp4", "demo.wav")
    ae_obj.extract_audio()
    print("MP4 -> WAV... Done")
    audio_file = "demo.wav"



    # Step 4: Run the Docker container and execute ASR
    print("Run Docker...")
    subprocess.run(['sudo', 'docker', 'run', '--runtime', 'nvidia', '-it', '--rm', '--network', 'host', '-e', 'DISPLAY=$DISPLAY', '-v', '/tmp/.X11-unix/:/tmp/.X11-unix', '--device', '/dev/bus/usb', '--device', '/dev/snd', 'jetson_voice', 'python', 'examples/asr.py', '--wav', audio_file])
    print("Run Docker... Done")

    # Step 5: Exit the container (optional)
    print("Exit...")
    subprocess.run(['exit'])
    print("Exit... Done")

    # Step 6: Change Directory
    print("Change Dir...")
    os.chdir('../')
    print("Change Dir... Done")


if __name__ == "__main__":

    

    print("Run Docker...")
    run_jetson_voice()
    print("Run Docker... Done")