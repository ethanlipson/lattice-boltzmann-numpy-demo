import subprocess
import sys
import os

timeStr = sys.argv[1]
framerate = 60

os.makedirs(f"videos/{timeStr.split('/')[0]}", exist_ok=True)
subprocess.run(
    f"ffmpeg -y -framerate {framerate} -pattern_type glob -i 'frames/{timeStr}/*.png' -c:v libx264 -pix_fmt yuv420p videos/{timeStr}.mp4",
    shell=True,
    capture_output=True,
)
