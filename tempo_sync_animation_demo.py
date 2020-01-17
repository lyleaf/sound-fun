import cv2
import librosa
import librosa.display
import wave
import pyaudio
from cv2 import VideoWriter, VideoWriter_fourcc

VIDEO_PATH = 'Demo_Full_1.mp4'
video = cv2.VideoCapture(VIDEO_PATH)


# Read a video to get FPS
fps = video.get(cv2.CAP_PROP_FPS)
print('Image frames per second is %d' % fps)
video.release()

# Create audio mp3 from mp4 
# ffmpeg -i holdon.mp4 holdon.mp3 not the same length
# ffmpeg -i holdon.mp4 -async 1 holdon.wav
AUDIO_PATH = 'Demo_Full_1.wav' # difference between wav and mp3
seconds = 5
y, sr = librosa.load(AUDIO_PATH)
print(int(len(y)))
print('Sample rate %d' % sr)
time = int(len(y)/sr)
print('Song time %d' % time)

interval = []
for i in range(time):
    sample_frames = sr * seconds
    sample = y[i*sr:i*sr+sample_frames]
    tempo, beat_frames = librosa.beat.beat_track(y=sample, sr=sr) #y should be audio time series np.ndarray [shape=(n,)] or None
    #print('Estimated tempo: {:.2f} beats per minute'.format(tempo))  
    interval.append(60/tempo)
print(interval)

downtempo = [] #in seconds
downtempo.append(5)
while(downtempo[-1] <= time):
    downtempo.append(interval[int(downtempo[-1])-5]+downtempo[-1])  
print(downtempo)

uptempo = []
for i in range(len(downtempo)-1):
    uptempo.append((downtempo[i]+downtempo[i+1])/2) 
tempo = []
for i in range(len(uptempo)):
    tempo.append(downtempo[i])
    tempo.append(uptempo[i])
tempo_frames = np.multiply(tempo,fps).astype(int)
print(tempo_frames)

# Create output video with animation
width = 1280
height = 720
FPS = fps
radius = 150
paint_h = int(height/2)
paint_x = int(width/2)

fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter('./Demo_Full_1_noises.avi', fourcc, float(FPS), (width, height))

index = 0
black = False
for timestamp in range(int(FPS)*time):
    frame = np.random.randint(0, 256, 
                              (height, width, 3), 
                              dtype=np.uint8)
    if ((index+2 < len(tempo_frames)-1) and (timestamp >= tempo_frames[index]) and (timestamp < tempo_frames[index+2])):
        print(timestamp)
        if (timestamp < tempo_frames[index+1]):
            cv2.circle(frame, (paint_x, paint_h), radius, (0, 0, 0), -1)
            print('add frame')       
    elif ((index+2 < len(tempo_frames)-1) and (timestamp >= tempo_frames[index+2])):
        index = index+2
    video.write(frame)

video.release()
# ffmpeg -i Demo_Full_1_noises.avi NOISES.mp4
