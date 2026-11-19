"""Search the Apple Loops library through Logic's own index.

Logic and GarageBand keep every installed Apple Loop in one SQLite file,
~/Music/Audio Music Apps/Databases/LogicLoopsDatabaseV10.db, rebuilt by Logic's
"Reindex all loops". Its layout, read from the file itself:

- Loops: one row per loop. fileURL (file:// URL), lengthInSeconds, tempo,
  numberOfBeats, key (48 = C .. 59 = B, 0 = no key), keyType (1 major,
  2 minor, 3 neither, 4 both), fileType (1 loop, 2 one-shot sound effect), hasMidi (a loop
  that also carries MIDI).
- LoopsKeywords: per loop, instrumentType ("Vocals"), instrumentSubType
  ("Choir"), genre and comma-separated descriptors ("Dark,Relaxed,Processed").
- Keywords and Tagging: the same tags as ids; JamPacks and JamPackLoopRelation:
  the sound pack each loop came from.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from urllib.parse import unquote, urlparse

DATABASE = Path("~/Music/Audio Music Apps/Databases/LogicLoopsDatabaseV10.db").expanduser()
NOTES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
KEY_TYPES = {1: "", 2: "m"}
MAJOR, MINOR = 1, 2


def key_name(key: int, key_type: int) -> str | None:
    if not 48 <= key <= 59:
        return None
    return NOTES[key - 48] + KEY_TYPES.get(key_type, "")


def parse_key(text: str) -> tuple[int, int]:
    """"F minor", "Fm", "Ab" or "G# major" as (pitch class, MAJOR or MINOR)."""
    t = text.strip().replace("♯", "#").replace("♭", "b")
    minor = t.lower().endswith(("minor", "min", "m")) and not t.lower().endswith("major")
    name = t.split()[0].rstrip("m") if " " in t else (t[:-1] if minor else t)
    name = name[0].upper() + name[1:]
    flats = {"Db": "C#", "D#": "Eb", "Gb": "F#", "G#": "Ab", "A#": "Bb", "Cb": "B", "Fb": "E"}
    return NOTES.index(flats.get(name, name)), MINOR if minor else MAJOR


def compatible(key: int, key_type: int, target: tuple[int, int]) -> bool:
    """The loop's key is the target key or its relative major/minor."""
    if not 48 <= key <= 59:
        return False
    pitch, kind = target
    relative = (pitch + 3) % 12 if kind == MINOR else (pitch - 3) % 12
    if key_type == kind:
        return (key - 48) == pitch
    if key_type in (MAJOR, MINOR):
        return (key - 48) == relative
    return False


def _word(text: str) -> re.Pattern:
    """Matches at the start of a word: "rain" finds "Hard Rain", not "Train"."""
    return re.compile(r"(?<![a-z0-9])" + re.escape(text.lower()))


def find_loops(
    *,
    text: str | None = None,
    key: str | None = None,
    unpitched: bool = True,
    tempo: list[float] | None = None,
    seconds: list[float] | None = None,
    exclude: str | None = None,
    limit: int = 25,
) -> dict:
    """
    Search the installed Apple Loops (Logic's loop index) for samples to use.
    text: words that must all start a word in the file name, instrument type/subtype,
    genre, descriptors or sound pack ("crickets", "choir dark", "foley",
    "vocals female relaxed"). key: "F minor" keeps loops in that key or its
    relative major (Ab); unpitched also keeps loops without a key (drums,
    noise, foley). tempo and seconds: [min, max]. One-shot sound effects
    (nature, foley, animals) have no tempo or beats. exclude: words that must not
    appear. One row per loop, columns named once; key is like "Ab" or "Fm".
    Logic plays loops at the project tempo and key; here they play as recorded.
    """
    if not DATABASE.is_file():
        return {"success": False, "error": f"no Logic loop index at {DATABASE}"}
    target = parse_key(key) if key else None
    words = [_word(w) for w in (text or "").split()]
    banned = [_word(w) for w in (exclude or "").split()]
    con = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)
    try:
        packs: dict[int, str] = {}
        for loop_id, name in con.execute("""
                select r.loopId, j.jamPackName from JamPackLoopRelation r
                join JamPacks j on j.identifier = r.jamPackId"""):
            packs[loop_id] = f"{packs[loop_id]},{name}" if loop_id in packs else name
        rows = [(*row[1:], packs.get(row[0], "")) for row in con.execute("""
            select l.id, l.fileURL, l.lengthInSeconds, l.tempo, l.numberOfBeats, l.key, l.keyType,
                   l.fileType, coalesce(k.instrumentType, ''), coalesce(k.instrumentSubType, ''),
                   coalesce(k.genre, ''), coalesce(k.descriptors, '')
            from Loops l left join LoopsKeywords k on k.loopId = l.id
            where l.isWebLoop = 0""")]
    finally:
        con.close()
    found, total = [], 0
    for url, length, bpm, beats, k, k_type, file_type, kind, sub, genre, desc, pack in rows:
        path = unquote(urlparse(url).path)
        haystack = f"{Path(path).stem} {kind} {sub} {genre} {desc} {pack}".lower()
        if not all(w.search(haystack) for w in words) or any(w.search(haystack) for w in banned):
            continue
        if target and not (compatible(k, k_type, target) or (unpitched and key_name(k, k_type) is None)):
            continue
        if tempo and not (tempo[0] <= (bpm or 0) <= tempo[1]):
            continue
        if seconds and not (seconds[0] <= length <= seconds[1]):
            continue
        total += 1
        if len(found) < limit:
            found.append([path, round(length, 2), bpm if file_type == 1 else None,
                          beats if file_type == 1 else None, key_name(k, k_type),
                          f"{kind}/{sub}".strip("/"), genre, desc, pack])
    return {"success": True, "matches": total,
            "columns": ["path", "seconds", "tempo", "beats", "key", "type", "genre", "descriptors",
                        "pack"],
            "loops": found}


TOOLS = (find_loops,)
