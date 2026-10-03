# -*- coding: utf-8 -*-
"""
Audio processing utilities: loading, trimming, normalization,
recording, device enumeration, and waveform extraction.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable
import os
import threading
import numpy as np
import soundfile as sf
import sounddevice as sd
from pydub import AudioSegment


class AudioProcessor:
    """Provides audio manipulation, recording, and analysis routines."""

    @staticmethod
    def load_audio(file_path: str | Path, target_sr: int = 24000) -> Tuple[np.ndarray, int]:
        """
        Load any common audio format, convert to mono and resample to target_sr.
        Returns (mono_audio_float32, sample_rate).
        """
        file_path = str(file_path)
        try:
            # First attempt direct load with soundfile
            data, sr = sf.read(file_path, dtype="float32")
        except Exception:
            # Fallback to pydub for mp3, m4a, etc.
            segment = AudioSegment.from_file(file_path)
            segment = segment.set_channels(1)
            segment = segment.set_frame_rate(target_sr)
            samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
            # Normalize integers to float32 [-1.0, 1.0]
            max_val = float(1 << (8 * segment.sample_width - 1))
            samples /= max_val
            return samples, target_sr

        # Convert to mono if multi-channel
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Simple linear resample if sample rate doesn't match
        if sr != target_sr:
            duration = len(data) / sr
            target_length = int(duration * target_sr)
            indices = np.linspace(0, len(data) - 1, target_length)
            data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
            sr = target_sr

        return data.astype(np.float32), sr

    @staticmethod
    def get_audio_duration(file_path: str | Path) -> float:
        """Return audio duration in seconds."""
        file_path = str(file_path)
        try:
            info = sf.info(file_path)
            return float(info.duration)
        except Exception:
            try:
                segment = AudioSegment.from_file(file_path)
                return len(segment) / 1000.0
            except Exception:
                return 0.0

    @staticmethod
    def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Normalize audio volume to peak level to avoid clipping and enhance presence."""
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            return (audio / peak) * target_peak
        return audio

    @staticmethod
    def trim_silence(audio: np.ndarray, sample_rate: int, threshold_db: float = -35.0) -> np.ndarray:
        """
        Trim leading and trailing silence based on energy threshold.
        """
        if len(audio) == 0:
            return audio

        # Calculate frame-wise RMS
        frame_length = int(sample_rate * 0.02)  # 20ms frame
        if frame_length == 0 or len(audio) < frame_length:
            return audio

        num_frames = len(audio) // frame_length
        rms = np.zeros(num_frames)
        for i in range(num_frames):
            frame = audio[i * frame_length : (i + 1) * frame_length]
            rms[i] = np.sqrt(np.mean(frame**2) + 1e-9)

        rms_db = 20 * np.log10(rms + 1e-9)
        active_frames = np.where(rms_db > threshold_db)[0]

        if len(active_frames) == 0:
            return audio

        start_sample = max(0, active_frames[0] * frame_length - int(sample_rate * 0.1))
        end_sample = min(len(audio), (active_frames[-1] + 1) * frame_length + int(sample_rate * 0.1))

        return audio[start_sample:end_sample]

    @staticmethod
    def extract_waveform_peaks(audio: np.ndarray, num_points: int = 150) -> List[float]:
        """
        Downsample audio to a fixed number of normalized peak values [0.0, 1.0]
        for visual waveform representation in the UI.
        """
        if len(audio) == 0:
            return [0.0] * num_points

        # Work with absolute amplitude
        abs_audio = np.abs(audio)
        chunk_size = max(1, len(abs_audio) // num_points)
        peaks = []

        for i in range(num_points):
            start = i * chunk_size
            end = min(len(abs_audio), (i + 1) * chunk_size)
            if start < len(abs_audio):
                val = float(np.max(abs_audio[start:end]))
            else:
                val = 0.0
            peaks.append(val)

        max_val = max(peaks) if peaks and max(peaks) > 1e-6 else 1.0
        return [min(1.0, p / max_val) for p in peaks]

    @staticmethod
    def save_audio(file_path: str | Path, audio: np.ndarray, sample_rate: int = 24000) -> bool:
        """Save numpy audio array to wav or other supported format."""
        file_path = str(file_path)
        try:
            # Ensure folder exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            # Clip between -1 and 1
            clipped = np.clip(audio, -1.0, 1.0).astype(np.float32)
            sf.write(file_path, clipped, sample_rate)
            return True
        except Exception as err:
            print(f"Error saving audio to {file_path}: {err}")
            return False

    @staticmethod
    def list_input_devices() -> List[Dict[str, any]]:
        """List available audio input recording devices."""
        devices = []
        try:
            for idx, dev in enumerate(sd.query_devices()):
                if dev.get("max_input_channels", 0) > 0:
                    devices.append({
                        "index": idx,
                        "name": dev.get("name", f"Eingabegerät {idx}"),
                        "channels": dev.get("max_input_channels"),
                        "default_sr": dev.get("default_samplerate", 44100)
                    })
        except Exception as err:
            print(f"Error querying input devices: {err}")
        return devices

    @staticmethod
    def list_output_devices() -> List[Dict[str, any]]:
        """List available audio playback devices."""
        devices = []
        try:
            for idx, dev in enumerate(sd.query_devices()):
                if dev.get("max_output_channels", 0) > 0:
                    devices.append({
                        "index": idx,
                        "name": dev.get("name", f"Ausgabegerät {idx}"),
                        "channels": dev.get("max_output_channels"),
                        "default_sr": dev.get("default_samplerate", 44100)
                    })
        except Exception as err:
            print(f"Error querying output devices: {err}")
        return devices


class AudioRecorder:
    """Background microphone audio recorder with live volume level callbacks."""

    def __init__(self, sample_rate: int = 24000, device_index: Optional[int] = None):
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.is_recording = False
        self._frames: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        self.level_callback: Optional[Callable[[float], None]] = None

    def start(self, level_callback: Optional[Callable[[float], None]] = None) -> bool:
        """Start microphone recording stream."""
        if self.is_recording:
            return False

        self.level_callback = level_callback
        self._frames.clear()
        self.is_recording = True

        def callback(indata, frames, time_info, status):
            if not self.is_recording:
                return
            audio_copy = indata.copy()
            with self._lock:
                self._frames.append(audio_copy)
            if self.level_callback:
                level = float(np.max(np.abs(audio_copy)))
                self.level_callback(min(1.0, level * 1.5))

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device_index,
                callback=callback
            )
            self._stream.start()
            return True
        except Exception as err:
            print(f"Error starting recording stream: {err}")
            self.is_recording = False
            return False

    def stop(self) -> Optional[np.ndarray]:
        """Stop microphone recording and return the recorded numpy array."""
        if not self.is_recording:
            return None

        self.is_recording = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        with self._lock:
            if not self._frames:
                return np.array([], dtype=np.float32)
            audio = np.concatenate(self._frames, axis=0).flatten()

        return audio.astype(np.float32)
