# VoiceClone AI Studio 🎙️

**VoiceClone AI Studio** ist eine moderne, Python-basierte Desktop-Anwendung mit grafischer Benutzeroberfläche (PySide6 / Qt 6) zum Erstellen und Klonen von Stimmenmodellen aus Audiodateien mittels modernster neuronaler Sprachsynthese (**Coqui XTTS-v2**) und NVIDIA CUDA-Beschleunigung (RTX 3060).

---

## ✨ Hauptfunktionen

- **🚀 Multi-Sample & Zero-Shot Stimmen-Klonung:**
  - Erstellen Sie ein dauerhaftes Stimmen-Profil aus einer oder **mehreren Audiodateien gleichzeitig** (Multi-Sample Studio).
  - Mehrere kurze Sprachaufnahmen (z. B. 3 bis 6 Sätze mit unterschiedlicher Betonung und Tonhöhe) erhöhen die Stimmtreue und Natürlichkeit drastisch.
  - Die neuronalen Stimmen-Merkmale (*Conditioning Latents & Speaker Embeddings*) werden aggregiert extrahiert und blitzschnell für spätere Synthesen wiederverwendet.

- **🎙️ Integrierte Mikrofon-Aufnahme:**
  - Nehmen Sie Ihre eigene Stimme direkt in der App über ein beliebiges Mikrofon auf.
  - Fügen Sie beliebig viele Mikrofon-Takes nacheinander zur Clip-Liste hinzu.
  - Mit Live-Pegelanzeige (VU-Meter), Aufnahmedauer-Timer und automatischer Vorbereitung.

- **🔊 Leistungsstarke Sprachsynthese (TTS):**
  - Beliebige Texte mit der geklonten Stimme in **17 Sprachen** (Deutsch, Englisch, Französisch, Spanisch, Italienisch u. v. m.) sprechen lassen.
  - Extrem schnelle Generierung dank GPU-Beschleunigung auf der NVIDIA GeForce RTX 3060 (~3-fach schneller als Echtzeit).
  - Feineinstellung für Sprachgeschwindigkeit (*Speed*), Expressivität/Kreativität (*Temperature*) und Wiederholungsstrafe (*Repetition Penalty*).

- **🎛️ Integrierter Audio-Player mit Waveform:**
  - Direkte Wiedergabe generierter Sprache oder kombinierter Referenz-Audio in der Benutzeroberfläche.
  - Interaktive Wellenform-Anzeige mit Timeline-Scrubbing, Loop-Wiedergabe, Lautstärkeregler und Audio-Export (WAV / MP3).

- **📚 Stimmen-Bibliothek:**
  - Übersicht aller gespeicherten Stimmen-Profile mit Clip-Zähler, Sprach- und Dauerkennung.
  - Such- und Filterfunktion, Hörproben-Vorschau sowie Export und Import von Stimmen-Profilen als portable `.zip`-Archive.

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

## 💡 Tipps für optimale Sprachklonung mit Multi-Sampling

1. **Mehrere verschiedene Aufnahmen kombinieren (Empfohlen):** Nutzen Sie 3 bis 6 kurze Aufnahmen (jeweils 5–10 Sekunden, Gesamtlänge idealerweise 15–45 Sekunden).
2. **Phonetische Vielfalt & Dynamik:** Sprechen Sie in den Clips unterschiedliche Sätze (Fragesätze, Aussagesätze, ruhigere und lebendigere Passagen). Dadurch lernt das neuronale Modell alle Phoneme und Ihre authentische Tonhöhen-Dynamik.
3. **Audioqualität & Raumakustik:** Verwenden Sie rauschfreie Aufnahmen ohne Hintergrundmusik, Nachhall oder Nebengeräusche.
4. **Automatische Vorbereitung:** Nutzen Sie den Button *„Vorbereitung auf alle Clips anwenden“*, um Stille an den Enden automatisch abzuschneiden und alle Clips einheitlich auf -1 dB zu normalisieren.
5. **Exportieren & Teilen:** Sie können erstellte Profile jederzeit als `.zip` sichern oder auf anderen Geräten weiterverwenden.
