import pyaudio
import wave
import os
import sys
import time

# Audio settings
channels = 1
rate = 24000
chunk = 1024
record_seconds = 2
file_count = 0

recording_class = 'drone'
output_folder = os.path.join('recordings', recording_class)
os.makedirs(output_folder, exist_ok=True)


def list_input_devices():
    p = pyaudio.PyAudio()
    devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('maxInputChannels') > 0:
            devices.append((i, info.get('name')))
    p.terminate()
    return devices


def find_input_device_index(name_keyword):
    p = pyaudio.PyAudio()
    device_index = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('maxInputChannels') > 0 and name_keyword.lower() in info.get('name').lower():
            device_index = i
            break
    p.terminate()
    return device_index

# Print all available input devices at start
print("Available input devices:")
for i, name in list_input_devices():
    print(f"  ID {i}: {name}")

device_index = find_input_device_index("respeaker")  # adjust keyword here
if device_index is None:
    print("Desired input device not found. Exiting program.")
    sys.exit(1)  # Exit immediately with error code

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=chunk)

print("Recording... Press Ctrl+C to stop.")

try:
    while True:
        frames = []
        for _ in range(int(rate / chunk * record_seconds)):
            try:
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)
            except IOError as e:
                print(f"Warning: Buffer overflow ({e}). Skipping corrupted chunk.")
                frames.append(b'\x00' * chunk)  # Append silence if overflow occurs

        output_filename = os.path.join(output_folder, f"{recording_class}_{file_count}.wav")
        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        print(f"\033[92m Audio saved to \033[00m {output_filename}")
        file_count += 1

        # Optional: delay between recordings
        # time.sleep(0.5)

except KeyboardInterrupt:
    print("\nRecording stopped by user.")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
