# /// script
# dependencies = [
#   "bracketbot-ai @ /home/bracketbot/bracketbot-ai/dist/bracketbot_ai-0.0.1-py3-none-any.whl",
#   "sounddevice",
# ]
# ///

from bracketbot_ai import Transcriber
import sounddevice as sd
import time
import queue
import threading

def main():
    model = Transcriber(device=0, chunk_duration=2)
    
    audio_queue = queue.Queue()
    
    def audio_callback(indata, frames, time, status):
        audio_queue.put_nowait(indata)
    
    def process_audio():
        while True:
            try:
                audio_data = audio_queue.get_nowait()
                out = model(audio_data)
                if out.text.strip():
                    print(out.text)
                    print(out.speed)
                audio_queue.task_done()
            except queue.Empty:
                continue

    # Start processing thread
    processing_thread = threading.Thread(target=process_audio, daemon=True)
    processing_thread.start()

    with sd.InputStream(
        device=2,
        channels=1,
        samplerate=model.sample_rate,
        callback=audio_callback,
        blocksize=int(model.sample_rate * 2),
        dtype='float32'
    ):
        print("Listening...")
        while True:
            time.sleep(0.1)

if __name__ == "__main__":
    main()  