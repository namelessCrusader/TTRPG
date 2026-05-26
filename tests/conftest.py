"""Shared pytest fixtures."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@pytest.fixture(autouse=True)
def _pygame_isolation(request):
    """Reset pygame between tests that touch the display."""
    if "play_client" not in request.node.nodeid and "player_spawn" not in request.node.nodeid:
        yield
        return
    try:
        import pygame
        if pygame.get_init():
            pygame.quit()
    except Exception:
        pass
    yield
    try:
        import pygame
        if pygame.get_init():
            pygame.quit()
    except Exception:
        pass
