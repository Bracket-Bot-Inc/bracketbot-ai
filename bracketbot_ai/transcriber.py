#!/usr/bin/env python3
"""BracketBot AI Transcriber - SenseVoice RKNN inference for speech-to-text"""
import logging
import argparse
import sys
import time
import queue
import threading
from pathlib import Path
from datetime import datetime
from typing import Union, List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
import kaldi_native_fbank as knf
import sentencepiece as spm
import numpy as np
from rknnlite.api import RKNNLite

from bracketbot_ai.model_manager import ensure_model

SPEECH_SCALE = 0.5
RKNN_INPUT_LEN = 171

class SenseVoiceInferenceSession:
    def __init__(
        self,
        embedding_model_file,
        encoder_model_file,
        bpe_model_file,
        device_id=-1,
        intra_op_num_threads=4,
    ):
        logging.info(f"Loading model from {embedding_model_file}")

        self.embedding = np.load(embedding_model_file)
        logging.info(f"Loading model {encoder_model_file}")
        start = time.time()
        self.encoder = RKNNLite(verbose=False)
        self.encoder.load_rknn(encoder_model_file)
        self.encoder.init_runtime()

        logging.info(
            f"Loading {encoder_model_file} takes {time.time() - start:.2f} seconds"
        )
        self.blank_id = 0
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_model_file)

    def __call__(self, speech, language: int, use_itn: bool) -> np.ndarray:
        language_query = self.embedding[[[language]]]

        # 14 means with itn, 15 means without itn
        text_norm_query = self.embedding[[[14 if use_itn else 15]]]
        event_emo_query = self.embedding[[[1, 2]]]

        # scale the speech
        speech = speech * SPEECH_SCALE
        
        input_content = np.concatenate(
            [
                language_query,
                event_emo_query,
                text_norm_query,
                speech,
            ],
            axis=1,
        ).astype(np.float32)
        # pad [1, len, ...] to [1, RKNN_INPUT_LEN, ... ]
        input_content = np.pad(input_content, ((0, 0), (0, RKNN_INPUT_LEN - input_content.shape[1]), (0, 0)))
        start_time = time.time()
        encoder_out = self.encoder.inference(inputs=[input_content])[0]
        end_time = time.time()
        def unique_consecutive(arr):
            if len(arr) == 0:
                return arr
            # Create a boolean mask where True indicates the element is different from the previous one
            mask = np.append([True], arr[1:] != arr[:-1])
            out = arr[mask]
            out = out[out != self.blank_id]
            return out.tolist()
        
        #现在shape变成了1, n_vocab, n_seq. 这里axis需要改一下
        # hypos = unique_consecutive(encoder_out[0].argmax(axis=-1))
        hypos = unique_consecutive(encoder_out[0].argmax(axis=0))
        text = self.sp.DecodeIds(hypos)
        return text
# ```

# File: utils/frontend.py
# ```py
# -*- coding:utf-8 -*-
# @FileName  :frontend.py
# @Time      :2024/7/18 09:39
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com

class WavFrontend:
    """Conventional frontend structure for ASR."""

    def __init__(
        self,
        cmvn_file: str = None,
        fs: int = 16000,
        window: str = "hamming",
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        lfr_m: int = 7,
        lfr_n: int = 6,
        dither: float = 0,
        **kwargs,
    ) -> None:
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = fs
        opts.frame_opts.dither = dither
        opts.frame_opts.window_type = window
        opts.frame_opts.frame_shift_ms = float(frame_shift)
        opts.frame_opts.frame_length_ms = float(frame_length)
        opts.mel_opts.num_bins = n_mels
        opts.energy_floor = 0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False
        self.opts = opts

        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.cmvn_file = cmvn_file

        if self.cmvn_file:
            self.cmvn = self.load_cmvn()
        self.fbank_fn = None
        self.fbank_beg_idx = 0
        self.reset_status()

    def reset_status(self):
        self.fbank_fn = knf.OnlineFbank(self.opts)
        self.fbank_beg_idx = 0

    def fbank(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        waveform = waveform * (1 << 15)
        self.fbank_fn = knf.OnlineFbank(self.opts)
        self.fbank_fn.accept_waveform(self.opts.frame_opts.samp_freq, waveform.tolist())
        frames = self.fbank_fn.num_frames_ready
        mat = np.empty([frames, self.opts.mel_opts.num_bins])
        for i in range(frames):
            mat[i, :] = self.fbank_fn.get_frame(i)
        feat = mat.astype(np.float32)
        feat_len = np.array(mat.shape[0]).astype(np.int32)
        return feat, feat_len

    def lfr_cmvn(self, feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.lfr_m != 1 or self.lfr_n != 1:
            feat = self.apply_lfr(feat, self.lfr_m, self.lfr_n)

        if self.cmvn_file:
            feat = self.apply_cmvn(feat)

        feat_len = np.array(feat.shape[0]).astype(np.int32)
        return feat, feat_len

    def load_audio(self, filename: str) -> Tuple[np.ndarray, int]:
        data, sample_rate = sf.read(
            filename,
            always_2d=True,
            dtype="float32",
        )
        assert (
            sample_rate == 16000
        ), f"Only 16000 Hz is supported, but got {sample_rate}Hz"
        self.sample_rate = sample_rate
        data = data[:, 0]  # use only the first channel
        samples = np.ascontiguousarray(data)

        return samples, sample_rate

    @staticmethod
    def apply_lfr(inputs: np.ndarray, lfr_m: int, lfr_n: int) -> np.ndarray:
        LFR_inputs = []

        T = inputs.shape[0]
        T_lfr = int(np.ceil(T / lfr_n))
        left_padding = np.tile(inputs[0], ((lfr_m - 1) // 2, 1))
        inputs = np.vstack((left_padding, inputs))
        T = T + (lfr_m - 1) // 2
        for i in range(T_lfr):
            if lfr_m <= T - i * lfr_n:
                LFR_inputs.append(
                    (inputs[i * lfr_n : i * lfr_n + lfr_m]).reshape(1, -1)
                )
            else:
                # process last LFR frame
                num_padding = lfr_m - (T - i * lfr_n)
                frame = inputs[i * lfr_n :].reshape(-1)
                for _ in range(num_padding):
                    frame = np.hstack((frame, inputs[-1]))

                LFR_inputs.append(frame)
        LFR_outputs = np.vstack(LFR_inputs).astype(np.float32)
        return LFR_outputs

    def apply_cmvn(self, inputs: np.ndarray) -> np.ndarray:
        """
        Apply CMVN with mvn data
        """
        frame, dim = inputs.shape
        means = np.tile(self.cmvn[0:1, :dim], (frame, 1))
        vars = np.tile(self.cmvn[1:2, :dim], (frame, 1))
        inputs = (inputs + means) * vars
        return inputs

    def get_features(self, inputs: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(inputs, str):
            inputs, _ = self.load_audio(inputs)

        fbank, _ = self.fbank(inputs)
        feats = self.apply_cmvn(self.apply_lfr(fbank, self.lfr_m, self.lfr_n))
        return feats

    def load_cmvn(
        self,
    ) -> np.ndarray:
        with open(self.cmvn_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        means_list = []
        vars_list = []
        for i in range(len(lines)):
            line_item = lines[i].split()
            if line_item[0] == "<AddShift>":
                line_item = lines[i + 1].split()
                if line_item[0] == "<LearnRateCoef>":
                    add_shift_line = line_item[3 : (len(line_item) - 1)]
                    means_list = list(add_shift_line)
                    continue
            elif line_item[0] == "<Rescale>":
                line_item = lines[i + 1].split()
                if line_item[0] == "<LearnRateCoef>":
                    rescale_line = line_item[3 : (len(line_item) - 1)]
                    vars_list = list(rescale_line)
                    continue

        means = np.array(means_list).astype(np.float64)
        vars = np.array(vars_list).astype(np.float64)
        cmvn = np.array([means, vars])
        return cmvn
@dataclass
class TranscriptionResults:
    """Transcription results container"""
    text: str = ""
    segments: List[Dict] = field(default_factory=list)
    duration: float = 0.0
    speed: Dict[str, float] = field(default_factory=dict)
    chunk_id: Optional[int] = None
    timestamp: Optional[datetime] = None
    sequence_length: int = 0
    
    def __repr__(self): 
        return f"TranscriptionResults(text='{self.text[:50]}...', segments={len(self.segments)}, seq_len={self.sequence_length})"
    
    def __len__(self): 
        return len(self.segments)
    
    def __bool__(self):
        return bool(self.text.strip())

class Transcriber:
    """SenseVoice Speech-to-Text Transcriber"""
    
    def __init__(self, device=0, verbose=False, 
                 chunk_duration=4.0):
        self.device = device
        self.verbose = verbose
        
        # Audio parameters
        self.sample_rate = 16000
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * self.sample_rate)
        
        # Processing state
        self.chunk_count = 0
        self.recent_transcriptions = []
        self.max_recent = 5
        
        # Ensure SenseVoice model is available
        model_name = "SenseVoiceSmall"
        model_dir, self.model_path = ensure_model(model_name)
        self.frontend = WavFrontend(str(model_dir / "am.mvn"))
        self.asr = SenseVoiceInferenceSession(
            str(model_dir / "embedding.npy"),
            str(model_dir / "SenseVoiceSmall.rknn"),
            str(model_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model")
        )
    
    def __call__(self, chunk, language=0, use_itn=False):
        """Process single audio chunk using streaming frontend"""
        chunk = chunk.flatten()
        speed = {}
        # Get features from streaming frontend
        t1 = time.time()
        #chunk = chunk * SPEECH_SCALE
        features = self.frontend.get_features(chunk)
        if len(features) == 0:
            return TranscriptionResults(
                text="",
                speed=speed,
                timestamp=datetime.now()
            )
        
        if len(features.shape) == 2:
            features = features[np.newaxis, ...]

        speed['feature_extraction'] = (time.time() - t1) * 1000

        t1 = time.time()
        raw_text = self.asr(
            features,
            language=0,  # Auto-detect
            use_itn=False
        )
        speed['inference'] = (time.time() - t1) * 1000
        t1 = time.time()
        text = self._parse_output(raw_text)
        speed['postprocessing'] = (time.time() - t1) * 1000
        
        return TranscriptionResults(
            text=text,
            duration=features.shape[0] * 0.01,  # Assuming 10ms frame shift
            speed=speed,
            timestamp=datetime.now()
        )
    def _parse_output(self, raw_text):
        """Parse ASR output to extract clean text"""
        if not raw_text:
            return ""
            
        # Extract text after <|woitn|>
        if '<|woitn|>' in raw_text:
            parts = raw_text.split('<|woitn|>')
            text = parts[1].strip() if len(parts) > 1 else ""
            
            # Check if it's actual speech
            if '<|nospeech|>' in raw_text:
                return ""
            elif text:
                return text
            else:
                return ""
        else:
            # Fallback
            cleaned = raw_text.strip()
            if cleaned and cleaned != "[no speech]":
                return cleaned
            return ""

def main():
    import scipy.signal
    import sounddevice as sd
    import soundfile as sf
    parser = argparse.ArgumentParser(description="BracketBot AI Speech Transcriber")
    parser.add_argument('source', nargs='?', help='Audio file path or "stream" for continuous')
    parser.add_argument('--device', type=int, default=0, choices=[-1,0,1,2], help='NPU device')
    parser.add_argument('--chunk-duration', type=float, default=4.0, help='Chunk duration (seconds)')
    parser.add_argument('--min-frames', type=int, default=100, help='Minimum frames before inference')
    parser.add_argument('--scale', type=float, default=0.5, help='Input scaling factor')
    parser.add_argument('--language', type=int, default=0, help='Language code (0=auto)')
    parser.add_argument('--use-itn', action='store_true', help='Use inverse text normalization')
    parser.add_argument('--save-audio', action='store_true', help='Save audio chunks for debugging')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Update global scaling
    global SPEECH_SCALE
    SPEECH_SCALE = args.scale
    
    try:
        model = Transcriber(
            device=args.device, 
            verbose=args.verbose,
            chunk_duration=args.chunk_duration
        )
        
        # Set minimum frames for inference
        if model.frontend:
            model.frontend.min_frames_for_inference = args.min_frames
        
        if not args.source:
            args.source = "stream"
        
        if args.source == "stream":
            # Continuous transcription using frontend's streaming
            print("="*60)
            print("🎤 Continuous Speech Transcription (Streaming Frontend)")
            print("="*60)
            print(f"Audio Device: ReSpeaker Lite (Device {model.audio_device})")
            print(f"NPU Device: RK3588 (device {args.device})")
            print(f"Chunk Duration: {args.chunk_duration}s")
            print(f"Min Frames for Inference: {args.min_frames}")
            print(f"Input Scaling: {SPEECH_SCALE}")
            print()
            
            # Audio streaming callback using frontend's windowing
            audio_buffer = np.array([], dtype=np.float32)
            chunk_counter = 0
            
            def audio_callback(indata, frames, time, status):
                nonlocal audio_buffer, chunk_counter
                if status:
                    print(f"Audio status: {status}")
                
                # Append new audio data
                audio_buffer = np.concatenate([audio_buffer, indata[:, 0]])
                
                # Process when we have enough data for a chunk
                if len(audio_buffer) >= model.chunk_samples:
                    # Extract chunk
                    chunk = audio_buffer[:model.chunk_samples]
                    audio_buffer = audio_buffer[model.chunk_samples:]  # Remove processed chunk completely
                    
                    chunk_counter += 1
                    
                    # Add chunk to streaming frontend
                    try:
                        results = model(chunk)
                        if results and results.text.strip():
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            print(f"[{timestamp}] Chunk {chunk_counter}: {results.text}")
                            
                            if args.verbose and results.speed:
                                total_time = sum(results.speed.values())
                                print(f"  Speed: {total_time:.1f}ms")
                        elif args.verbose:
                            frames_ready = model.frontend.fbank_fn.num_frames_ready if model.frontend.fbank_fn else 0
                            print(f"Chunk {chunk_counter}: buffering ({frames_ready} frames)")
                    except Exception as e:
                        if args.verbose:
                            print(f"Transcription error: {e}")
            
            # Start streaming
            try:
                print("Starting audio stream... (Press Ctrl+C to stop)")
                print(f"Note: Inference will start after {args.min_frames} frames are accumulated")
                with sd.InputStream(
                    device=model.audio_device,
                    channels=1,
                    samplerate=model.sample_rate,
                    callback=audio_callback,
                    blocksize=int(model.sample_rate * 0.1),  # 100ms blocks
                    dtype='float32'
                ):
                    print("Listening...")
                    while True:
                        time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopping transcription...")
            except Exception as e:
                print(f"Streaming error: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        else:
            # Single file transcription
            print(f"\nTranscribing: {args.source}")
            print(f"Settings: language={args.language}, use_itn={args.use_itn}, scale={SPEECH_SCALE}")
            
            # Load audio file using soundfile
            try:
                audio_data, sample_rate = sf.read(args.source, dtype='float32')
                if len(audio_data.shape) > 1:
                    audio_data = audio_data[:, 0]  # Use first channel if stereo
                
                if sample_rate != 16000:
                    print(f"Warning: Audio sample rate is {sample_rate}Hz, expected 16000Hz")
                    print("Resampling to 16000Hz...")
                    audio_data = scipy.signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
                    sample_rate = 16000
                    print(f"Resampled audio: {len(audio_data)/sample_rate:.2f}s at {sample_rate}Hz")
                
                print(f"Loaded audio: {len(audio_data)/sample_rate:.2f}s at {sample_rate}Hz")
                
                # Process audio in chunks using streaming frontend
                all_results = []
                chunk_count = 0
                
                for i in range(0, len(audio_data), model.chunk_samples):
                    # Extract chunk
                    chunk_end = min(i + model.chunk_samples, len(audio_data))
                    chunk = audio_data[i:chunk_end]
                    
                    # Pad if chunk is too short
                    if len(chunk) < model.chunk_samples:
                        chunk = np.pad(chunk, (0, model.chunk_samples - len(chunk)), 'constant')
                    
                    chunk_count += 1
                    print(f"Processing chunk {chunk_count} ({i/sample_rate:.1f}s - {chunk_end/sample_rate:.1f}s)")
                    
                    try:
                        chunk = chunk.astype(np.float32)
                        print(chunk.shape)
                        results = model(chunk)
                        if results and results.text.strip():
                            all_results.append(results)
                            print(f"  Chunk {chunk_count}: '{results.text}'")
                        elif args.verbose:
                            frames_ready = model.frontend.fbank_fn.num_frames_ready if model.frontend.fbank_fn else 0
                            print(f"  Chunk {chunk_count}: buffering ({frames_ready} frames)")
                    except Exception as e:
                        print(f"  Error processing chunk {chunk_count}: {e}")
                
                # Combine results
                if all_results:
                    combined_text = " ".join(r.text for r in all_results if r.text.strip())
                    print(f"\nFull transcript: '{combined_text}'")
                    
                    if args.verbose:
                        total_speed = sum(sum(r.speed.values()) for r in all_results if r.speed)
                        print(f"\nTotal processing time: {total_speed:.1f}ms across {len(all_results)} chunks")
                        avg_speed = total_speed / len(all_results) if all_results else 0
                        print(f"Average per chunk: {avg_speed:.1f}ms")
                        
                        # Show timing breakdown by section
                        if all_results and hasattr(all_results[0], 'speed') and all_results[0].speed:
                            speed_keys = all_results[0].speed.keys()
                            for section in speed_keys:
                                section_total = sum(r.speed.get(section, 0) for r in all_results if r.speed)
                                section_avg = section_total / len(all_results) if all_results else 0
                                print(f"  {section}: {section_total:.1f}ms total, {section_avg:.1f}ms avg")
                else:
                    print("\nNo transcription results")
                
            except Exception as e:
                print(f"Error loading audio file: {e}")
                return 1
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 