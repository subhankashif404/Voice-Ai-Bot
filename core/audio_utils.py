import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import numpy as np
import queue

_recording_queue = queue.Queue()
_is_recording = False
_stream = None

def _callback(indata, frames, time, status):
    if status:
        pass
    _recording_queue.put(indata.copy())

def start_recording(fs=16000):
    global _is_recording, _stream
    if _is_recording: return
    
    # Check for available input devices
    try:
        devices = sd.query_devices()
        input_device = sd.query_devices(kind='input')
        if not input_device:
            raise RuntimeError("No input device found")
    except Exception as e:
        print(f"Error querying audio devices: {e}")
        return False

    while not _recording_queue.empty():
        _recording_queue.get()
        
    try:
        _is_recording = True
        _stream = sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=_callback)
        _stream.start()
        return True
    except Exception as e:
        _is_recording = False
        print(f"Failed to start stream: {e}")
        return False

def stop_recording(fs=16000):
    global _is_recording, _stream
    _is_recording = False
    if _stream:
        _stream.stop()
        _stream.close()
        _stream = None
        
    audio_data = []
    while not _recording_queue.empty():
        audio_data.append(_recording_queue.get())
        
    if not audio_data: 
        return None
        
    recording = np.concatenate(audio_data, axis=0)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp.name, fs, recording)
    return temp.name