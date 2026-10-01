"""The Likho speech engine: recordings in, two layers of text out.

from likho_engine import Engine, EngineSettings, LocalLanguage, transcribe

engine = Engine(EngineSettings())
transcript = transcribe(engine, "call.wav", LocalLanguage())
for line in transcript.segments:
    print(line.text_script, "|", line.text_roman)
"""

from likho_engine.config import EngineSettings
from likho_engine.engine import Detection, Engine, Line, StopRequested
from likho_engine.transcript import Decision, Language, LocalLanguage, Segment, Transcript, transcribe

__all__ = [
    "Decision",
    "Detection",
    "Engine",
    "EngineSettings",
    "Language",
    "Line",
    "LocalLanguage",
    "Segment",
    "StopRequested",
    "Transcript",
    "transcribe",
]
__version__ = "0.1.0"
