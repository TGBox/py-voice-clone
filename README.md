# VoiceClone AI Studio 🎙️

**VoiceClone AI Studio** ist eine moderne, Python-basierte Desktop-Anwendung mit grafischer Benutzeroberfläche (PySide6 / Qt 6) zum Erstellen und Klonen von Stimmenmodellen aus Audiodateien mittels modernster neuronaler Sprachsynthese (**Coqui XTTS-v2**) und NVIDIA CUDA-Beschleunigung (RTX 3060).

---

## ✨ Hauptfunktionen

- **🚀 Zero-Shot Stimmen-Klonung:**
  - Erstellen Sie ein dauerhaftes Stimmen-Profil aus nur 3 bis 15 Sekunden Referenz-Audio (Sprachnachricht, Interview, Hörbuch etc.).
  - Die neuronalen Stimmen-Merkmale (*Conditioning Latents & Speaker Embeddings*) werden einmalig extrahiert und blitzschnell wiederverwendet.

- **🎙️ Integrierte Mikrofon-Aufnahme:**
  - Nehmen Sie Ihre eigene Stimme direkt in der App über ein beliebiges Mikrofon auf.
  - Mit Live-Pegelanzeige (VU-Meter), Aufnahmedauer-Timer und automatischer Vorbereitung.

- **🔊 Leistungsstarke Sprachsynthese (TTS):**
  - Beliebige Texte mit der geklonten Stimme in **17 Sprachen** (Deutsch, Englisch, Französisch, Spanisch, Italienisch u. v. m.) sprechen lassen.
  - Extrem schnelle Generierung dank GPU-Beschleunigung auf der NVIDIA GeForce RTX 3060 (~3-fach schneller als Echtzeit).
  - Feineinstellung für Sprachgeschwindigkeit (*Speed*), Expressivität/Kreativität (*Temperature*) und Wiederholungsstrafe (*Repetition Penalty*).

- **🎛️ Integrierter Audio-Player mit Waveform:**
  - Direkte Wiedergabe generierter Sprache oder Referenz-Audio in der Benutzeroberfläche.
  - Interaktive Wellenform-Anzeige mit Timeline-Scrubbing, Loop-Wiedergabe, Lautstärkeregler und Audio-Export (WAV / MP3).

- **📚 Stimmen-Bibliothek:**
  - Übersicht aller gespeicherten Stimmen-Profile mit Such- und Filterfunktion.
  - Hörproben-Vorschau, Export und Import von Stimmen-Profilen als portable `.zip`-Archive.

- **🖥️ Vollbild- und Fenstermodus (F11):**
  - Nahtloser Wechsel zwischen rahmenlosem Vollbildmodus und normalem Fenstermodus über die Taste **F11** oder den Header-Button.

- **⚡ Hardware- & GPU-Monitor:**
  - Live-Anzeige der NVIDIA GPU-Auslastung und VRAM-Belegung in Echtzeit.
  - Vollständig lokaler Betrieb ohne externe Cloud-Abhängigkeiten.

---

## 🚀 Schnellstart

### 1. Starten über den Windows-Batch-Launcher (Empfohlen)

Doppelklicken Sie einfach auf die Datei:

```cmd
run.bat
```

### 2. Starten über das Terminal (PowerShell / CMD)

```powershell
.\.venv\Scripts\python.exe main.py
```

---

## 📂 Projektstruktur

```text
py-voice-clone/
│
├── core/                        # Backend-Logik
│   ├── config.py                # Pfade, Konstanten und unterstützte Sprachen
│   ├── audio_processor.py       # Audio-Laden, Normalisierung, Trimming, Mikrofon
│   ├── voice_profile.py         # Profil-Verwaltung, Metadaten, Latents-Speicherung
│   ├── xtts_engine.py           # XTTS-v2 Wrapper (GPU-Inferenz, Latent-Extraktion)
│   └── model_manager.py         # Modell-Prüfung, Download-Worker, Hardware-Telemetry
│
├── ui/                          # PySide6 Desktop-Oberfläche
│   ├── theme.py                 # Modernes Dark-Theme (QSS)
│   ├── main_window.py           # Hauptfenster mit Sidebar & F11 Vollbild-Logik
│   ├── components/              # Wiederverwendbare Widgets
│   │   ├── waveform_view.py     # Custom Waveform-Visualisierung
│   │   ├── audio_player.py      # Audio-Player mit Scrubbing & Export
│   │   └── mic_recorder.py      # Mikrofon-Aufnahmedialog mit VU-Meter
│   └── views/                   # Hauptansichten
│       ├── clone_view.py        # Stimme aus Datei / Mikrofon erstellen
│       ├── synth_view.py        # Text-to-Speech Studio
│       ├── library_view.py      # Gespeicherte Stimmen & Zip-Import/Export
│       └── settings_view.py     # Hardware-Monitor & Modell-Status
│
├── data/
│   ├── profiles/                # Gespeicherte Stimmen-Profile
│   ├── outputs/                 # Generierte Sprachdateien
│   └── recordings/              # Mikrofon-Aufnahmen
│
├── models/
│   └── xtts_v2/                 # Lokale neuronale Modellgewichte (~1.9 GB)
│
├── main.py                      # Einstiegspunkt der Anwendung
├── run.bat                      # 1-Klick-Starter für Windows
└── requirements.txt             # Getestete Python-Abhängigkeiten
```

---

## 💡 Tipps für optimale Sprachklonung

1. **Audioqualität:** Verwenden Sie eine saubere Audioaufnahme ohne laute Musik, Hall oder Störgeräusche im Hintergrund.
2. **Optimale Dauer:** 5 bis 12 Sekunden zusammenhängende, natürliche Sprache liefern das beste Ergebnis für das Speaker Embedding.
3. **Automatische Vorbereitung:** Nutzen Sie vor dem Klonen den Button *„Vorbereitung anwenden“*, um Stille am Anfang/Ende zu entfernen und den Pegel auf -1 dB zu normalisieren.
4. **Exportieren & Teilen:** Sie können erstellte Stimmen jederzeit als `.zip` exportieren und auf anderen Rechnern importieren.
