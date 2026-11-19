"""Instruments built from ReaSamplOmatic5000: multisampled instruments and drum pads.

Sample libraries name each zone by its root note: a MIDI number ("Super_48.aif")
or a note name ("Pad C3.wav"). One RS5k per zone covers the notes nearest its
root and repitches them from it, so chords play across the whole keyboard.
One-shots (drums, clicks, foley) get one RS5k per pad instead, each on its own
note. Hundreds of hits then share one open file per sound; as audio items each
hit holds its own, and past REAPER's open-file limit the rest render silent.
"""

from __future__ import annotations

import re
from pathlib import Path

from reaper_mcp.compose_tools import _set_shown
from reaper_mcp.reaper import RPR, get_project, held

RS5K = "ReaSamplOmatic5000 (Cockos)"
SUFFIXES = (".wav", ".aif", ".aiff", ".flac")
NOTE = re.compile(r"(?<![A-Za-z])([A-Ga-g])([#b]?)(-?\d)(?!\d)")
NUMBER = re.compile(r"(?<!\d)(\d{2,3})(?!\d)")
STEPS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

# RS5k parameter indices (stable across versions) and their scaling
VOLUME, MIN_VELOCITY_GAIN, NOTE_START, NOTE_END, PITCH_START, PITCH_END = 0, 2, 3, 4, 5, 6
MAX_VOICES, ATTACK, RELEASE, OBEY_NOTE_OFFS, LOOP = 8, 9, 10, 11, 12
MS_RANGE = 2000.0  # Attack and Release run 0..2000 ms
# RS5k "MODE" config: 0 = freely configurable shifted (pitch moves from the start
# value to the end value across the range), 1 = sample (ignores the note, the
# default after loading a file), 2 = note (semitone shifted).
FREELY_SHIFTED = "0"
PITCH_RANGE = 80.0  # pitch parameters run -80..+80 semitones


def root_of(name: str, *, middle_c: str) -> int | None:
    """A zone's root note from its file name. Note names read with middle C
    (MIDI 60) called middle_c: "C3" for Logic and Yamaha, "C4" for most others.
    Without a note name, a lone 2-3 digit number is taken as the MIDI note. With
    both ("10CC 27D#1"), a number naming the same pitch class wins: libraries
    mislabel octaves more often than MIDI numbers."""
    stem = Path(name).stem
    octave_offset = 60 - 12 * (int(middle_c[1:]) + 1)
    numbers = [int(n) for n in NUMBER.findall(stem) if 0 <= int(n) <= 127]
    found = NOTE.findall(stem)
    if found:
        letter, accidental, octave = found[-1]
        semitone = STEPS[letter.lower()] + {"#": 1, "b": -1}.get(accidental, 0)
        agreeing = [n for n in numbers if n % 12 == semitone % 12]
        if agreeing:
            return agreeing[-1]
        midi = 12 * (int(octave) + 1) + semitone + octave_offset
        return midi if 0 <= midi <= 127 else None
    return numbers[-1] if len(numbers) == 1 else None


def zones(roots: list[int]) -> list[tuple[int, int]]:
    """Note ranges for sorted roots: each zone runs to halfway to its neighbours,
    the first down to 0 and the last up to 127."""
    ranges = []
    for i, root in enumerate(roots):
        low = 0 if i == 0 else (roots[i - 1] + root) // 2 + 1
        high = 127 if i == len(roots) - 1 else (root + roots[i + 1]) // 2
        ranges.append((low, high))
    return ranges


@held
def load_multisample(  # noqa: PLR0913 -- an MCP tool's options are its keyword arguments
    *,
    track_index: int,
    folder: str,
    match: str | None = None,
    middle_c: str = "C3",
    attack_ms: float = 5.0,
    release_ms: float = 400.0,
    loop: bool = False,
    velocity_db: float = -24.0,
) -> dict:
    """
    Make a playable multisampled instrument on a track from a folder of samples
    named by root note ("Super_48.aif" or "Pad C3.wav"): one ReaSamplOmatic5000
    per root, each covering the notes nearest it and repitching from it.
    match: only files whose name contains this. middle_c: what the library calls
    MIDI 60 ("C3" for Logic/Alchemy, "C4" for most others). Notes stop on note-off
    with release_ms; loop repeats each sample while a note is held.
    velocity_db: the gain at the softest velocity (0 = velocity changes nothing).
    Replaces RS5k zones this tool added before.
    """
    root_dir = Path(folder).expanduser()
    if not root_dir.is_dir():
        return {"success": False, "error": f"not a folder: {folder}"}
    by_root: dict[int, Path] = {}
    skipped = []
    for path in sorted(root_dir.iterdir()):
        if path.suffix.lower() not in SUFFIXES or (match and match.lower() not in path.name.lower()):
            continue
        root = root_of(path.name, middle_c=middle_c)
        if root is None or root in by_root:
            skipped.append(path.name)
            continue
        by_root[root] = path
    if not by_root:
        return {"success": False, "error": f"no samples with a root note in their name in {folder}",
                "skipped": skipped[:10]}
    track = get_project().tracks[track_index]
    _remove_zones(track)
    roots = sorted(by_root)
    for root, (low, high) in zip(roots, zones(roots), strict=True):
        fx = RPR.TrackFX_AddByName(track.id, RS5K, False, 1)
        RPR.TrackFX_SetNamedConfigParm(track.id, fx, "FILE0", str(by_root[root]))
        RPR.TrackFX_SetNamedConfigParm(track.id, fx, "DONE", "")
        RPR.TrackFX_SetNamedConfigParm(track.id, fx, "MODE", FREELY_SHIFTED)
        RPR.TrackFX_SetNamedConfigParm(track.id, fx, "renamed_name", f"zone {root}")
        for param, value in ((NOTE_START, low / 127), (NOTE_END, high / 127),
                             (PITCH_START, (low - root + PITCH_RANGE) / (2 * PITCH_RANGE)),
                             # RS5k reaches the end pitch one note past the range end
                             (PITCH_END, (high + 1 - root + PITCH_RANGE) / (2 * PITCH_RANGE)),
                             (ATTACK, attack_ms / MS_RANGE), (RELEASE, release_ms / MS_RANGE),
                             (OBEY_NOTE_OFFS, 1.0), (LOOP, 1.0 if loop else 0.0)):
            RPR.TrackFX_SetParam(track.id, fx, param, value)
        _set_shown(track.id, fx, MIN_VELOCITY_GAIN, text=f"{velocity_db}")
    result = {"success": True, "track_index": track_index, "zones": len(roots),
              "roots": roots, "range": [0, 127]}
    if skipped:
        result["skipped"] = skipped[:10]
    return result


def _remove_zones(track, prefix: str = "zone ") -> None:
    for index in reversed(range(RPR.TrackFX_GetCount(track.id))):
        name = RPR.TrackFX_GetNamedConfigParm(track.id, index, "renamed_name", "", 256)[4]
        if str(name).startswith(prefix):
            RPR.TrackFX_Delete(track.id, index)


@held
def load_one_shots(
    *,
    track_index: int,
    pads: dict[str, str | dict],
    velocity_db: float = -24.0,
    release_ms: float = 30.0,
) -> dict:
    """
    A drum rack from one-shot samples: one ReaSamplOmatic5000 per pad, each
    playing its file whole, on one MIDI note, louder with velocity. Write the
    hits with write_notes (velocity = dynamics, small timing offsets = feel).
    pads: {"36": "/path/kick.aif"} or {"36": {"file": path, "pitch": semitones,
    "gain_db": 0}}. velocity_db: the gain at the softest velocity.
    Replaces pads this tool added before.
    """
    specs = {}
    for note, spec in pads.items():
        spec = {"file": spec} if isinstance(spec, str) else dict(spec)
        if not Path(spec["file"]).expanduser().is_file():
            return {"success": False, "error": f"pad {note}: no file {spec['file']}"}
        specs[int(note)] = spec
    track = get_project().tracks[track_index]
    _remove_zones(track, "pad ")
    for note, spec in sorted(specs.items()):
        fx = RPR.TrackFX_AddByName(track.id, RS5K, False, 1)
        RPR.TrackFX_SetNamedConfigParm(track.id, fx, "FILE0", str(Path(spec["file"]).expanduser()))
        RPR.TrackFX_SetNamedConfigParm(track.id, fx, "DONE", "")
        RPR.TrackFX_SetNamedConfigParm(track.id, fx, "MODE", FREELY_SHIFTED)
        RPR.TrackFX_SetNamedConfigParm(track.id, fx, "renamed_name", f"pad {note}")
        pitch = (float(spec.get("pitch", 0.0)) + PITCH_RANGE) / (2 * PITCH_RANGE)
        for param, value in ((NOTE_START, note / 127), (NOTE_END, note / 127),
                             (PITCH_START, pitch), (PITCH_END, pitch),
                             (ATTACK, 0.0), (RELEASE, release_ms / MS_RANGE),
                             (OBEY_NOTE_OFFS, 0.0), (LOOP, 0.0)):
            RPR.TrackFX_SetParam(track.id, fx, param, value)
        _set_shown(track.id, fx, MIN_VELOCITY_GAIN, text=f"{velocity_db}")
        _set_shown(track.id, fx, VOLUME, text=f"{spec.get('gain_db', 0.0)}")
    return {"success": True, "track_index": track_index, "pads": sorted(specs)}


TOOLS = (load_multisample, load_one_shots)
