"""Speaks results out loud (macOS `say`) and can add the same speech to a saved video.

Live: `say` runs in the background so the video loop never waits for it.
Saved videos: every utterance is recorded with its video time; afterwards each one is rendered
to audio with `say -o` and mixed into the video at that time with ffmpeg, so a recorded demo
plays back with the voice in the right places.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path


class Speaker:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and shutil.which("say") is not None
        self._proc: subprocess.Popen | None = None
        self.log: list[tuple[float, str]] = []  # (video time, text) of everything said

    def say(self, text: str, time_s: float | None = None):
        if time_s is not None:
            self.log.append((time_s, text))
        if not self.enabled:
            return
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()  # newest result wins
        self._proc = subprocess.Popen(["say", text])


def add_speech_track(video_path: str, utterances: list[tuple[float, str]], out_path: str) -> bool:
    """Mix spoken `utterances` (time in seconds, text) into the video. Also re-encodes it as
    H.264 so it plays everywhere. Returns False (and leaves the video alone) if `say` or
    ffmpeg is missing."""
    if not utterances or shutil.which("say") is None or shutil.which("ffmpeg") is None:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        inputs, filters = ["-i", video_path], []
        for k, (t, text) in enumerate(utterances):
            clip = Path(tmp) / f"say{k}.aiff"
            subprocess.run(["say", "-o", str(clip), text], check=True)
            inputs += ["-i", str(clip)]
            ms = int(t * 1000)
            filters.append(f"[{k + 1}:a]adelay={ms}|{ms}[a{k}]")
        mix = "".join(f"[a{k}]" for k in range(len(utterances)))
        filters.append(f"{mix}amix=inputs={len(utterances)}:normalize=0[speech]")
        cmd = ["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(filters),
               "-map", "0:v", "-map", "[speech]", "-vf", "scale=-2:min(1080\\,ih)", "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p",
               "-c:a", "aac", out_path]
        subprocess.run(cmd, check=True)
    return True
