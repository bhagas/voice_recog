import os
import subprocess

# Loop through the filesystem
for root, dirs, files in os.walk("./dataset/Actor_01", topdown=False):
    # Loop through files
    for name in files:
        # Consider only mp4
        if name.endswith('.mp4'):
            
            # Using ffmpeg to convert the mp4 in wav
            # Example command: "ffmpeg -i C:/test.mp4 -ab 160k -ac 2 -ar 44100 -vn audio.wav"
            command = "ffmpeg -i ." + root[1:] + "/" + name + " " + "-ab 160k -ac 2 -ar 44100 -vn ./dataset/Actor_01/" +  name[:-3] + "wav"
            
            #print(command)
            
            # Execute conversion
            try:
                subprocess.call(command, shell=True)
                
            # Skip the file in case of error
            except ValueError:
                continue