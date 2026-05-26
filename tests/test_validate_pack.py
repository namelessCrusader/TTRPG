"""Tests for world pack validation."""

from pathlib import Path

from src.sim.validate_pack import KNOWN_PACKS, validate_all_packs, validate_pack

WORLDS = Path(__file__).resolve().parents[1] / "worlds"


def test_tavern_pack_validates():
    errors = validate_pack(WORLDS / "tavern", smoke_ticks=5)
    assert errors == [], errors


def test_all_known_packs_validate():
    results = validate_all_packs(WORLDS, smoke_ticks=5)
    assert set(results.keys()) == set(KNOWN_PACKS)
    failures = {name: errs for name, errs in results.items() if errs}
    assert not failures, failures


def test_scenario_ambient_refs_valid():
    """Scenario force_ambient ids must exist in pack ambient_events."""
    for name in ("castle", "spicy", "magic_duel", "tavern"):
        errors = validate_pack(WORLDS / name, smoke_ticks=0)
        amb_errs = [e for e in errors if "force_ambient" in e]
        assert amb_errs == [], amb_errs
