import argparse
import json
import os
import queue
import sys
import wave

import sounddevice as sd
from vosk import KaldiRecognizer, Model

# English model folder (download and unzip here)
ENGLISH_MODEL_PATH = "models/vosk-model-small-en-us-0.15"


def load_english_model() -> Model:
    if not os.path.isdir(ENGLISH_MODEL_PATH):
        raise FileNotFoundError(
            f"English model not found at: {ENGLISH_MODEL_PATH}\n\n"
            "Download this English model:\n"
            "  https://alphacephei.com/vosk/models\n"
            "Model name:\n"
            "  vosk-model-small-en-us-0.15\n\n"
            "Unzip it so the folder path is exactly:\n"
            f"  {ENGLISH_MODEL_PATH}"
        )
    return Model(ENGLISH_MODEL_PATH)


def transcribe_wav(model: Model, wav_path: str) -> str:
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    with wave.open(wav_path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()

        if channels != 1 or sampwidth != 2 or framerate != 16000:
            raise ValueError(
                "WAV must be PCM 16-bit, mono, 16000 Hz.\n"
                "Convert with:\n"
                f'  ffmpeg -i "{wav_path}" -ac 1 -ar 16000 -sample_fmt s16 output.wav'
            )

        rec = KaldiRecognizer(model, framerate)
        rec.SetWords(True)

        parts = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if result.get("text"):
                    parts.append(result["text"])

        final = json.loads(rec.FinalResult())
        if final.get("text"):
            parts.append(final["text"])

    return " ".join(parts).strip()


def transcribe_mic(model: Model, device=None):
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    samplerate = 16000
    rec = KaldiRecognizer(model, samplerate)
    rec.SetWords(True)

    print("Listening (English model)... Press Ctrl+C to stop.\n")

    try:
        with sd.RawInputStream(
            samplerate=samplerate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=callback,
            device=device,
        ):
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").strip()
                    if text:
                        print(text)
                else:
                    partial = json.loads(rec.PartialResult()).get("partial", "").strip()
                    if partial:
                        print(f"\r{partial}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nStopped.")
        final = json.loads(rec.FinalResult()).get("text", "").strip()
        if final:
            print("Final:", final)


def main():
    parser = argparse.ArgumentParser(
        description="Offline Speech-to-Text (English only) using Vosk"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mic", action="store_true", help="Transcribe from microphone")
    group.add_argument("--file", help="Transcribe from a WAV file (16kHz mono PCM16)")
    parser.add_argument("--device", type=int, default=None, help="Audio input device id (optional)")
    args = parser.parse_args()

    model = load_english_model()

    if args.mic:
        transcribe_mic(model, device=args.device)
    else:
        print(transcribe_wav(model, args.file))


if __name__ == "__main__":
    main()
