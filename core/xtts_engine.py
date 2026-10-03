# -*- coding: utf-8 -*-
"""
XTTS-v2 Voice Cloning and Speech Synthesis Engine.
Provides high-performance inference on CUDA/CPU with memory caching.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Sequence, Union
import os
import time
import torch
import numpy as np

from .config import XTTS_MODEL_DIR, DEFAULT_SAMPLE_RATE
from .audio_processor import AudioProcessor


class XTTSEngine:
    """Wrapper around XTTS-v2 for conditioning latent extraction and zero-shot synthesis."""

    def __init__(self, model_dir: Path = XTTS_MODEL_DIR, device: Optional[str] = None):
        self.model_dir = model_dir
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Any = None
        self.config: Any = None
        self.is_loaded = False

    def load_model(self, force_reload: bool = False) -> Tuple[bool, str]:
        """
        Load the XTTS v2 model into VRAM/RAM.
        Returns (success, message).
        """
        if self.is_loaded and not force_reload:
            return True, "Modell ist bereits geladen."

        config_path = self.model_dir / "config.json"
        model_path = self.model_dir / "model.pth"
        vocab_path = self.model_dir / "vocab.json"

        if not (config_path.exists() and model_path.exists() and vocab_path.exists()):
            return False, f"Modell-Dateien wurden in {self.model_dir} nicht gefunden. Bitte zuerst herunterladen."

        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts

            print(f"Loading XTTS-v2 configuration from {config_path}...")
            self.config = XttsConfig()
            self.config.load_json(str(config_path))

            print(f"Initializing XTTS model on {self.device}...")
            self.model = Xtts.init_from_config(self.config)
            self.model.load_checkpoint(
                self.config,
                checkpoint_dir=str(self.model_dir),
                checkpoint_path=str(model_path),
                vocab_path=str(vocab_path),
                eval=True,
                use_deepspeed=False
            )

            # Move model to target device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model.cuda()
            else:
                self.model.cpu()

            self.is_loaded = True
            return True, f"Modell erfolgreich auf {self.device.upper()} geladen."
        except Exception as err:
            self.is_loaded = False
            self.model = None
            self.config = None
            error_str = f"Fehler beim Laden des Modells: {str(err)}"
            print(error_str)
            return False, error_str

    def unload_model(self):
        """Free model memory from GPU."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.config is not None:
            del self.config
            self.config = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def extract_conditioning_latents(
        self, audio_paths: Union[str, Path, Sequence[Union[str, Path]]]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
        """
        Extract conditioning latents and speaker embedding from one or multiple reference audio files.
        Returns (gpt_cond_latent, speaker_embedding, status_message).
        """
        if not self.is_loaded or self.model is None:
            success, msg = self.load_model()
            if not success or self.model is None:
                return None, None, msg

        if isinstance(audio_paths, (list, tuple)):
            paths = [str(p) for p in audio_paths]
        else:
            paths = [str(audio_paths)]

        valid_paths = [p for p in paths if os.path.exists(p)]
        if not valid_paths:
            return None, None, "Keine der angegebenen Referenz-Audiodateien existiert."

        try:
            print(f"Extracting conditioning latents from {len(valid_paths)} audio file(s): {valid_paths}")
            # XTTS get_conditioning_latents accepts a list of audio files and pools speaker embeddings
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=valid_paths,
                gpt_cond_len=30,
                max_ref_length=60,
                sound_norm_refs=True
            )
            return gpt_cond_latent, speaker_embedding, f"Stimmen-Latents erfolgreich aus {len(valid_paths)} Datei(en) extrahiert."
        except Exception as err:
            error_str = f"Fehler bei Latent-Extraktion: {str(err)}"
            print(error_str)
            return None, None, error_str

    def synthesize(
        self,
        text: str,
        language: str = "de",
        gpt_cond_latent: Optional[torch.Tensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        reference_audio_path: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
        temperature: float = 0.75,
        repetition_penalty: float = 2.0,
        speed: float = 1.0,
        length_penalty: float = 1.0
    ) -> Tuple[Optional[np.ndarray], int, str]:
        """
        Synthesize speech from text using conditioning latents or reference audio.
        Returns (audio_array, sample_rate, status_message).
        """
        if not self.is_loaded or self.model is None:
            success, msg = self.load_model()
            if not success or self.model is None:
                return None, 0, msg

        # If latents not provided, extract them from reference audio
        if gpt_cond_latent is None or speaker_embedding is None:
            if not reference_audio_path:
                return None, 0, "Keine Stimmen-Latents und keine Referenz-Audiodatei angegeben."
            latents, spk_emb, msg = self.extract_conditioning_latents(reference_audio_path)
            if latents is None or spk_emb is None:
                return None, 0, msg
            gpt_cond_latent = latents
            speaker_embedding = spk_emb

        if gpt_cond_latent is None or speaker_embedding is None:
            return None, 0, "Konnte Stimmen-Latents nicht initialisieren."

        try:
            # Ensure tensors are on the correct device
            gpt_cond_latent = gpt_cond_latent.to(self.device)
            speaker_embedding = speaker_embedding.to(self.device)

            print(f"Synthesizing text ({len(text)} chars, lang={language}) on {self.device}...")
            start_time = time.time()

            if self.model is None:
                return None, 0, "Modell ist nicht im Speicher vorhanden."

            out = self.model.inference(
                text=text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=float(temperature),
                repetition_penalty=float(repetition_penalty),
                length_penalty=float(length_penalty),
                speed=float(speed),
                enable_text_splitting=True
            )

            elapsed = time.time() - start_time
            wav = np.array(out["wav"], dtype=np.float32)
            sr = 24000

            return wav, sr, f"Erfolgreich synthetisiert in {elapsed:.2f} Sekunden."
        except Exception as err:
            error_str = f"Fehler bei Sprachsynthese: {str(err)}"
            print(error_str)
            return None, 0, error_str
