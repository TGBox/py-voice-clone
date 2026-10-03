# -*- coding: utf-8 -*-
"""
Voice profile data model and storage manager.
Stores reference audio, extracted conditioning latents, and metadata.
"""

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any
import json
import shutil
import zipfile
import datetime
import re
import torch
import numpy as np

from .config import PROFILES_DIR
from .audio_processor import AudioProcessor


@dataclass
class VoiceProfile:
    id: str
    name: str
    created_at: str
    language: str
    description: str
    duration_sec: float
    sample_rate: int
    has_latents: bool
    folder_path: str
    sample_count: int = 1

    @property
    def sample_audio_path(self) -> Path:
        return Path(self.folder_path) / "sample.wav"

    @property
    def latents_path(self) -> Path:
        return Path(self.folder_path) / "latents.pth"

    @property
    def samples_dir(self) -> Path:
        return Path(self.folder_path) / "samples"

    @property
    def reference_sample_paths(self) -> List[Path]:
        """Return list of all stored reference sample paths, or fallback to sample.wav."""
        if self.samples_dir.exists():
            files = sorted(list(self.samples_dir.glob("*.wav")) + list(self.samples_dir.glob("*.mp3")))
            if files:
                return files
        return [self.sample_audio_path] if self.sample_audio_path.exists() else []

    @property
    def metadata_path(self) -> Path:
        return Path(self.folder_path) / "meta.json"


class VoiceProfileManager:
    """Manages the creation, retrieval, deletion, and import/export of voice profiles."""

    def __init__(self, base_dir: Path = PROFILES_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Convert a user-facing name to a safe filesystem identifier."""
        clean = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE).strip().lower()
        clean = re.sub(r"[-\s]+", "_", clean)
        return clean or "unnamed_voice"

    def list_profiles(self) -> List[VoiceProfile]:
        """Scan profile folder and return all valid voice profiles."""
        profiles = []
        for profile_dir in self.base_dir.iterdir():
            if profile_dir.is_dir():
                meta_file = profile_dir / "meta.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        profile = VoiceProfile(
                            id=data.get("id", profile_dir.name),
                            name=data.get("name", profile_dir.name),
                            created_at=data.get("created_at", ""),
                            language=data.get("language", "de"),
                            description=data.get("description", ""),
                            duration_sec=float(data.get("duration_sec", 0.0)),
                            sample_rate=int(data.get("sample_rate", 24000)),
                            has_latents=(profile_dir / "latents.pth").exists(),
                            folder_path=str(profile_dir),
                            sample_count=int(data.get("sample_count", 1))
                        )
                        profiles.append(profile)
                    except Exception as err:
                        print(f"Error reading profile in {profile_dir}: {err}")

        # Sort newest first
        profiles.sort(key=lambda p: p.created_at, reverse=True)
        return profiles

    def get_profile(self, profile_id: str) -> Optional[VoiceProfile]:
        """Get a single profile by id."""
        for p in self.list_profiles():
            if p.id == profile_id:
                return p
        return None

    def create_profile(
        self,
        name: str,
        audio_data: np.ndarray,
        sample_rate: int = 24000,
        gpt_cond_latent: Optional[torch.Tensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        language: str = "de",
        description: str = "",
        sample_count: int = 1,
        source_audio_paths: Optional[List[str | Path]] = None
    ) -> VoiceProfile:
        """
        Create a new voice profile folder, store sample.wav, individual samples, latents.pth, and meta.json.
        """
        slug = self._sanitize_name(name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_id = f"{slug}_{timestamp}"
        profile_dir = self.base_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        # Save primary combined audio sample
        sample_wav_path = profile_dir / "sample.wav"
        AudioProcessor.save_audio(sample_wav_path, audio_data, sample_rate)
        duration_sec = len(audio_data) / sample_rate if sample_rate > 0 else 0.0

        # Save individual source sample files if provided
        saved_samples_list = []
        if source_audio_paths:
            samples_subfolder = profile_dir / "samples"
            samples_subfolder.mkdir(parents=True, exist_ok=True)
            for idx, src_p in enumerate(source_audio_paths):
                src_path = Path(src_p)
                if src_path.exists():
                    clean_orig_name = self._sanitize_name(src_path.stem)
                    dest_name = f"sample_{idx+1:02d}_{clean_orig_name}{src_path.suffix.lower()}"
                    dest_path = samples_subfolder / dest_name
                    shutil.copy2(src_path, dest_path)
                    saved_samples_list.append(dest_name)
            sample_count = max(sample_count, len(saved_samples_list))

        # Save latents if provided
        has_latents = False
        if gpt_cond_latent is not None and speaker_embedding is not None:
            latents_path = profile_dir / "latents.pth"
            torch.save(
                {
                    "gpt_cond_latent": gpt_cond_latent.cpu(),
                    "speaker_embedding": speaker_embedding.cpu()
                },
                latents_path
            )
            has_latents = True

        # Save metadata
        iso_created = datetime.datetime.now().isoformat()
        meta = {
            "id": profile_id,
            "name": name,
            "created_at": iso_created,
            "language": language,
            "description": description,
            "duration_sec": duration_sec,
            "sample_rate": sample_rate,
            "has_latents": has_latents,
            "sample_count": sample_count,
            "samples": saved_samples_list
        }
        with open(profile_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return VoiceProfile(
            id=profile_id,
            name=name,
            created_at=iso_created,
            language=language,
            description=description,
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            has_latents=has_latents,
            folder_path=str(profile_dir),
            sample_count=sample_count
        )

    def load_latents(
        self, profile: VoiceProfile, device: str = "cuda"
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Load conditioning latents from profile's latents.pth and move to device.
        """
        if not profile.has_latents or not profile.latents_path.exists():
            return None

        try:
            checkpoint = torch.load(profile.latents_path, map_location=device, weights_only=False)
            gpt_cond = checkpoint.get("gpt_cond_latent")
            spk_emb = checkpoint.get("speaker_embedding")
            if gpt_cond is not None and spk_emb is not None:
                return gpt_cond.to(device), spk_emb.to(device)
        except Exception as err:
            print(f"Error loading latents for {profile.name}: {err}")
        return None

    def delete_profile(self, profile_id: str) -> bool:
        """Delete profile folder and all associated assets."""
        profile_dir = self.base_dir / profile_id
        if profile_dir.exists() and profile_dir.is_dir():
            shutil.rmtree(profile_dir, ignore_errors=True)
            return True
        return False

    def export_profile_zip(self, profile_id: str, dest_zip_path: str | Path) -> bool:
        """Export voice profile to a portable .zip file."""
        profile = self.get_profile(profile_id)
        if not profile:
            return False

        profile_dir = Path(profile.folder_path)
        try:
            with zipfile.ZipFile(dest_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in profile_dir.glob("*"):
                    if file_path.is_file():
                        zf.write(file_path, arcname=file_path.name)
            return True
        except Exception as err:
            print(f"Error exporting profile: {err}")
            return False

    def import_profile_zip(self, zip_path: str | Path) -> Optional[VoiceProfile]:
        """Import voice profile from a .zip file."""
        zip_path = Path(zip_path)
        if not zip_path.exists():
            return None

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                if "meta.json" not in zf.namelist():
                    return None
                meta_data = json.loads(zf.read("meta.json").decode("utf-8"))
                name = meta_data.get("name", "Imported Voice")
                slug = self._sanitize_name(name)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                new_id = f"{slug}_{timestamp}"
                target_dir = self.base_dir / new_id
                target_dir.mkdir(parents=True, exist_ok=True)
                zf.extractall(target_dir)

                # Update meta.json with new id and timestamp
                meta_data["id"] = new_id
                meta_data["created_at"] = datetime.datetime.now().isoformat()
                with open(target_dir / "meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta_data, f, indent=2, ensure_ascii=False)

                return self.get_profile(new_id)
        except Exception as err:
            print(f"Error importing profile zip: {err}")
            return None
