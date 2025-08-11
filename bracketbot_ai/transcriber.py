#!/usr/bin/env python3
"""BracketBot AI Transcriber - SenseVoice RKNN inference for speech-to-text"""

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

class WavFrontend:
    # lovemefan, lovemefan@outlook.com https://huggingface.co/happyme531/SenseVoiceSmall-RKNN2/blob/main/sensevoice_rknn.py
    """Conventional frontend structure for ASR with streaming support."""

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
        
        # Minimum frames required before processing
        # SenseVoice needs sufficient context for good recognition, but max 167 speech frames
        self.min_frames_for_inference = 30  # Lower threshold to process more chunks

    def reset_status(self):
        self.fbank_fn = knf.OnlineFbank(self.opts)
        self.fbank_beg_idx = 0

    def accept_waveform(self, waveform: np.ndarray) -> bool:
        """
        Add new waveform data to the streaming frontend.
        Returns True if enough frames are available for inference.
        """
        waveform = waveform * (1 << 15)
        self.fbank_fn.accept_waveform(self.opts.frame_opts.samp_freq, waveform.tolist())
        return self.fbank_fn.num_frames_ready >= self.min_frames_for_inference

    def get_streaming_features(self) -> Tuple[np.ndarray, int]:
        """
        Get features from the streaming frontend if enough frames are available.
        Returns features and number of new frames processed.
        """
        frames_ready = self.fbank_fn.num_frames_ready
        
        if frames_ready < self.min_frames_for_inference:
            return None, 0
        
        # Get all available frames
        mat = np.empty([frames_ready, self.opts.mel_opts.num_bins], dtype=np.float32)
        for i in range(frames_ready):
            mat[i, :] = self.fbank_fn.get_frame(i)
        
        feat = mat.astype(np.float32)
        
        # Apply LFR and CMVN
        feat = self.apply_lfr(feat, self.lfr_m, self.lfr_n)
        if self.cmvn_file:
            feat = self.apply_cmvn(feat)
        
        # Ensure final output is float32
        feat = feat.astype(np.float32)
        
        # Pad or truncate to match SenseVoice input requirements
        # SenseVoice concatenates: language_query(1) + event_emo_query(2) + text_norm_query(1) + speech
        # Total must fit in RKNN_INPUT_LEN (171), so speech should be <= 167 frames
        max_speech_frames = 167
        current_frames = feat.shape[0]
        
        if current_frames < max_speech_frames:
            # Pad with zeros to reach reasonable size for speech recognition
            target_frames = min(max_speech_frames, max(current_frames, 100))  # At least 100 frames for context
            if target_frames > current_frames:
                padding = np.zeros((target_frames - current_frames, feat.shape[1]), dtype=np.float32)
                feat = np.vstack([feat, padding])
        elif current_frames > max_speech_frames:
            # Truncate to max allowed speech frames
            feat = feat[:max_speech_frames]
        
        # Update beginning index for next call
        new_frames = frames_ready - self.fbank_beg_idx
        self.fbank_beg_idx = frames_ready
        
        return feat, new_frames

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
        import soundfile as sf
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
        # Ensure inputs are float32
        inputs = inputs.astype(np.float32)
        frame, dim = inputs.shape
        
        # Convert CMVN data to float32 to avoid dtype promotion
        means = np.tile(self.cmvn[0:1, :dim], (frame, 1)).astype(np.float32)
        vars = np.tile(self.cmvn[1:2, :dim], (frame, 1)).astype(np.float32)
        
        # Apply CMVN and ensure result is float32
        result = (inputs + means) * vars
        return result.astype(np.float32)

    def get_features(self, inputs: Union[str, np.ndarray]) -> Tuple[np.ndarray, int]:
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
                 chunk_duration=4.0, context_chunk_count=2):
        self.device = device
        self.verbose = verbose
        self.context_chunk_count = context_chunk_count
        
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
        self.embedding = np.load(str(model_dir / "embedding.npy"))
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model"))
        self.frontend = WavFrontend(str(model_dir / "am.mvn"))
        self.rknn = None
        self._load_model()
    
    def _load_model(self):
        """Load SenseVoice models"""
        try:
            self.rknn = RKNNLite()
            ret = self.rknn.load_rknn(str(self.model_path))
            if ret != 0: 
                raise RuntimeError(f"Failed to load RKNN model: {ret}")
            
            core_mask = 0b111 if self.device == -1 else (1 << self.device)
            ret = self.rknn.init_runtime(core_mask=core_mask)
            if ret != 0: 
                raise RuntimeError(f"Failed to init RKNN runtime: {ret}")
            if self.verbose:
                print(f"✓ Transcriber loaded: {self.model_path.name} on NPU device {self.device}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load SenseVoice models: {e}")
    
    def __call__(self, chunk, language=0, use_itn=False):
        """Process single audio chunk using streaming frontend"""
        chunk = chunk.flatten()
        speed = {}
        t1 = time.time()
        
        # Reset frontend state to treat this chunk independently
        if self.chunk_count == 0:
            self.frontend.reset_status()
        self.chunk_count = (self.chunk_count + 1) % self.context_chunk_count
        
        # Apply input scaling
        speed['preprocessing'] = (time.time() - t1) * 1000
        
        # Add chunk to frontend's streaming buffer
        t1 = time.time()
        inference_ready = self.frontend.accept_waveform(chunk)
        speed['buffering'] = (time.time() - t1) * 1000
        
        if not inference_ready:
            # Not enough frames yet, return empty result
            return TranscriptionResults(
                text="",
                speed=speed,
                timestamp=datetime.now()
            )
        
        # Get features from streaming frontend
        t1 = time.time()
        try:
            features, new_frames = self.frontend.get_streaming_features()
            if features is None or new_frames == 0:
                return TranscriptionResults(
                    text="",
                    speed=speed,
                    timestamp=datetime.now()
                )
        except Exception as e:
            return TranscriptionResults(text=f"Feature extraction error: {e}", sequence_length=0)
        
        speed['feature_extraction'] = (time.time() - t1) * 1000
        
        # Run inference
        t1 = time.time()
        if len(features.shape) == 2:
            features = features[np.newaxis, ...]  # Add batch dimension
        
        # Verify features have reasonable shape for SenseVoice
        if len(features.shape) != 3 or features.shape[0] != 1:
            return TranscriptionResults(text=f"Feature shape error: expected (1, frames, 560), got {features.shape}")
        
        if features.shape[2] != 560:
            return TranscriptionResults(text=f"Feature dimension error: expected 560 features, got {features.shape[2]}")
        
        if features.shape[1] > 167:
            return TranscriptionResults(text=f"Too many frames: got {features.shape[1]}, max 167 for speech")
        
        # Features should already be float32 from the frontend pipeline
        assert features.dtype == np.float32, f"Expected float32 features, got {features.dtype}"
        
        try:
            # SenseVoice inference with proper embedding concatenation
            if self.embedding is not None:
                # Add language, event/emotion, and text normalization embeddings as in original
                language_query = self.embedding[[[language]]]
                # 14 means with itn, 15 means without itn
                text_norm_query = self.embedding[[[14 if use_itn else 15]]]
                event_emo_query = self.embedding[[[1, 2]]]
                
                # Concatenate embeddings with features (axis=1 is sequence dimension)
                input_content = np.concatenate([
                    language_query,      # 1 frame
                    event_emo_query,     # 2 frames  
                    text_norm_query,     # 1 frame
                    features      # 684 frames
                ], axis=1).astype(np.float32)
                
                
                # Pad to expected input length (171 from RKNN_INPUT_LEN)
                RKNN_INPUT_LEN = 171
                if input_content.shape[1] < RKNN_INPUT_LEN:
                    padding_needed = RKNN_INPUT_LEN - input_content.shape[1]
                    input_content = np.pad(input_content, ((0, 0), (0, padding_needed), (0, 0)))
                elif input_content.shape[1] > RKNN_INPUT_LEN:
                    input_content = input_content[:, :RKNN_INPUT_LEN, :]
                
                # Run RKNN inference
                outputs = self.rknn.inference(inputs=[input_content])
            else:
                # Fallback to direct inference if no embeddings
                outputs = self.rknn.inference(inputs=[features])
            
            # Basic decoding - just use the first output and decode tokens
            if len(outputs) > 0:
                encoder_out = outputs[0]
                
                # Simple argmax decoding
                def unique_consecutive(arr):
                    if len(arr) == 0:
                        return arr
                    mask = np.append([True], arr[1:] != arr[:-1])
                    out = arr[mask]
                    out = out[out != 0]  # Remove blank tokens (blank_id = 0)
                    return out.tolist()
                
                # Try different output shapes based on SenseVoice format
                if len(encoder_out.shape) == 3:
                    # Most likely shape is (1, vocab_size, sequence_length)
                    # Following the original: encoder_out[0].argmax(axis=0)
                    argmax_tokens = encoder_out[0].argmax(axis=0)
                    
                    hypos = unique_consecutive(argmax_tokens)
                else:
                    hypos = unique_consecutive(encoder_out.flatten())
                
                # Convert tokens to text using SentencePiece if available
                if self.sp is not None and len(hypos) > 0:
                    raw_text = self.sp.DecodeIds(hypos)
                else:
                    # Fallback - show token sequence
                    raw_text = f"<|{language}|><|woitn|>Tokens: {hypos[:20]}..."  # Show first 20 tokens
            else:
                raw_text = f"<|{language}|><|woitn|>No output from model"
                
        except Exception as e:
            return TranscriptionResults(text=f"Inference error: {e}", sequence_length=0)
        
        speed['inference'] = (time.time() - t1) * 1000
        
        # Parse output (postprocess)
        t1 = time.time()
        if not raw_text:
            text = ""
        elif '<|woitn|>' in raw_text:
            parts = raw_text.split('<|woitn|>')
            text = parts[1].strip() if len(parts) > 1 else ""
            # Check if it's actual speech
            if '<|nospeech|>' in raw_text:
                text = ""
            elif not text:
                text = ""
        else:
            # Fallback - return cleaned raw text
            cleaned = raw_text.strip()
            if cleaned and cleaned != "[no speech]":
                text = cleaned
            else:
                text = ""
        
        speed['postprocessing'] = (time.time() - t1) * 1000
        
        return TranscriptionResults(
            text=text,
            duration=new_frames * 0.01,  # Assuming 10ms frame shift
            speed=speed,
            timestamp=datetime.now()
        )
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'rknn') and self.rknn:
            self.rknn.release()

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