import pyaudio
import wave
from pyo import *
import librosa
import numpy as np
import librosa.display
import uuid

sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 44100  # Record at 44100 samples per second
seconds = 1

random_uuid = str(uuid.uuid4())
filename = "niubi%s.wav" % random_uuid
chunk = 2048 #fs*5  # Record in chunks of 1024 samples


p = pyaudio.PyAudio()  # Create an interface to PortAudio


device = 'MacBook Pro Microphone'
dev_list, dev_index =  pa_get_input_devices()
print(dev_list)
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