import sqlite3

from reaper_mcp import loop_library as lib


def test_keys():
    assert lib.key_name(53, 2) == "Fm"
    assert lib.key_name(56, 1) == "Ab"
    assert lib.key_name(0, 0) is None
    assert lib.parse_key("F minor") == (5, lib.MINOR)
    assert lib.parse_key("Fm") == (5, lib.MINOR)
    assert lib.parse_key("G# major") == (8, lib.MAJOR)
    assert lib.parse_key("Db") == (1, lib.MAJOR)
    f_minor = lib.parse_key("F minor")
    assert lib.compatible(53, 2, f_minor)       # F minor
    assert lib.compatible(56, 1, f_minor)       # Ab major, relative
    assert not lib.compatible(53, 1, f_minor)   # F major
    assert not lib.compatible(0, 0, f_minor)


def test_find_loops(tmp_path, monkeypatch):
    db = tmp_path / "loops.db"
    con = sqlite3.connect(db)
    con.executescript("""
        create table Loops (id integer, fileURL text, lengthInSeconds float, tempo integer,
            numberOfBeats integer, key integer, keyType integer, fileType integer, isWebLoop integer);
        create table LoopsKeywords (loopId integer, instrumentType text, instrumentSubType text,
            genre text, descriptors text);
        create table JamPacks (identifier text, jamPackName text);
        create table JamPackLoopRelation (jamPackId text, loopId integer);
        insert into Loops values (1, 'file:///L/Dark%20Choir.caf', 6.0, 80, 8, 53, 2, 1, 0),
                                 (2, 'file:///L/Crickets.caf', 9.4, 0, 0, 0, 0, 2, 0),
                                 (3, 'file:///L/Bright%20Choir.caf', 6.0, 80, 8, 55, 1, 1, 0);
        insert into LoopsKeywords values (1, 'Vocals', 'Choir', 'Orchestral', 'Dark'),
                                         (2, 'Sound Effect', 'Animals', '', ''),
                                         (3, 'Vocals', 'Choir', 'Orchestral', 'Cheerful');
        insert into JamPacks values ('v', 'Jam Pack Voices');
        insert into JamPackLoopRelation values ('v', 1);
    """)
    con.commit()
    con.close()
    monkeypatch.setattr(lib, "DATABASE", db)
    r = lib.find_loops(text="choir", key="F minor")
    assert r["matches"] == 1
    assert r["loops"][0][0] == "/L/Dark Choir.caf"
    assert r["loops"][0][-1] == "Jam Pack Voices"
    crickets = lib.find_loops(text="animals")["loops"][0]
    assert crickets[2] is None and crickets[4] is None  # one-shot: no tempo, no key
    assert lib.find_loops(text="choir", exclude="cheerful")["matches"] == 1
