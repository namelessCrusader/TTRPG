"""Sourced numbers for wave 13 (lane A-S).

The numbers that decide results (wood and rock strength, brine properties,
JANAF equilibria, body and bone numbers, ...) were looked up in real sources and
stored with their citation in tests/data/sourced_numbers.json. This module is
the one door to that file:

    from w13_sourced import get          # or load the file by path
    value, status, citation = get("wood.bending_mor")

  value     whatever the entry stores (a number, a list or a dict); None when
            the number could not be found
  status    "sourced" or "not_found"
  citation  authors, title, year and URL as one string (for a not_found entry:
            what was searched and the leads that were not opened)

An unknown key raises KeyError. Nothing here touches the network: the file is
read from disk once, on the first call, and never at import. Other lanes call
`get` inside try/except, so a missing file is an OSError they can catch.

`entry(key)` returns the whole record (note, alt, derived, vs_spec) as a copy,
and `keys()` lists the keys. A test that checks a (u) number reads the status
through `get` and says which status it saw (acceptance tags, plan section 9).
"""
import copy
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(os.path.dirname(_HERE), "tests", "data", "sourced_numbers.json")
_CACHE = {}


def _entries():
    if "entries" not in _CACHE:
        with open(PATH) as f:
            _CACHE["entries"] = json.load(f)["entries"]
    return _CACHE["entries"]


def get(key):
    """(value, status, citation) of one key; KeyError when there is no such key."""
    e = _entries()[key]
    return copy.deepcopy(e["value"]), e["status"], e["citation"]


def entry(key):
    """The whole stored record of one key, as a copy."""
    return copy.deepcopy(_entries()[key])


def keys():
    """Every key, in file order."""
    return list(_entries())
