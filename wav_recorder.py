import pyaudio
import wave
from pyo import *
import librosa
import numpy as np
import librosa.display

sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 44100  # Record at 44100 samples per second
seconds = 10
filename = "output.wav"
chunk = 2048 #fs*5  # Record in chunks of 1024 samples


p = pyaudio.PyAudio()  # Create an interface to PortAudio


device = 'Pixel USB-C earbuds'
dev_list, dev_index =  pa_get_input_devices()
# ['Pixel USB-C earbuds', 'MacBook Pro Microphone']
dev = dev_index[dev_list.index(device)]



print('Recording')

stream = p.open(format=sample_format,
             channels=channels,
             rate=fs,
             input=True,
             input_device_index = dev,
             frames_per_buffer=chunk)

frames = []  # Initialize array to store frames

# Store data in chunks for 3 seconds
for i in range(0, int(fs / chunk * seconds)):
    data = stream.read(chunk, exception_on_overflow=False)
    frames.append(data)

# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()



# Save the recorded data as a WAV file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()
print('Finished recording')


# Understand tempo
y, sr = librosa.load('output.wav')
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print('Estimated tempo: {:.2f} beats per minute'.format(tempo/2))
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)

o_env = librosa.onset.onset_strength(y, sr=sr)
times = librosa.times_like(o_env, sr=sr)

import matplotlib.pyplot as plt
D = np.abs(librosa.stft(y))
plt.figure()
ax1 = plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                          x_axis='time', y_axis='log')
plt.title('Power spectrogram')
plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(times, o_env, label='Onset strength')
plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
            linestyle='--', label='Onsets')
plt.axis('tight')
plt.legend(frameon=True, framealpha=0.75)
plt.show()
