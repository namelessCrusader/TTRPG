"""Headless pygame smoke tests for PlayClient (no display required)."""

from __future__ import annotations

import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from src.sim.game_loop import GameLoop
from src.sim.lm_adapter import MockLMAdapter
from src.sim.play_client import PlayClient, _EVENT_TURN_DONE, _EVENT_UI_MSG
from src.sim.schemas import EntityKind
from src.sim.world_loader import load_world_pack


def _boot_client(view: str = "voxel") -> tuple[PlayClient, GameLoop, str]:
    pygame.init()
    world = load_world_pack("worlds/voxel_tavern")
    loop = GameLoop(world, adapter=MockLMAdapter())
    pid = next(
        eid for eid, e in world.spatial.entities.items() if e.kind == EntityKind.PLAYER
    )
    loop.player_id = pid
    client = PlayClient(loop, view=view, adapter_label="mock", window_w=960, window_h=640)
    pygame.display.set_mode((client.win_w, client.win_h), pygame.RESIZABLE)
    client._font = pygame.font.SysFont("monospace", 14)
    client._font_sm = pygame.font.SysFont("monospace", 11)
    return client, loop, pid


def _pump(client: PlayClient) -> None:
    for event in pygame.event.get():
        if event.type == _EVENT_UI_MSG:
            client._push_message(event.text, getattr(event, "color", "normal"))
        elif event.type == _EVENT_TURN_DONE:
            client._turn_busy = False
            client._on_world_changed()


def _frame(client: PlayClient, screen: pygame.Surface) -> None:
    screen.fill((12, 14, 22))
    client._draw_world(screen)
    client._draw_hud(screen)
    client._draw_input(screen)
    pygame.display.flip()
    _pump(client)


def test_hud_buttons_and_instant_move():
    client, loop, pid = _boot_client("voxel")
    screen = pygame.display.get_surface()
    p0 = loop.world.spatial.entities[pid].position

    _frame(client, screen)
    assert client._buttons, "expected HUD buttons"

    client._activate_button("map")
    _frame(client, screen)
    assert client._messages

    client._activate_button("move north")
    _frame(client, screen)
    p1 = loop.world.spatial.entities[pid].position
    assert (p1.x, p1.y) != (p0.x, p0.y), f"move north did not change position: {p0} -> {p1}"

    pygame.quit()


def test_zoom_rotate_and_fullscreen_toggle():
    client, _loop, _pid = _boot_client("voxel")
    screen = pygame.display.get_surface()

    client._zoom_view(0.15)
    client._view_yaw = math.pi / 4
    _frame(client, screen)
    assert client._view_zoom > 1.0

    client._toggle_fullscreen(screen)
    screen = pygame.display.get_surface()
    client._sync_display_geometry(screen)
    _frame(client, screen)

    client._toggle_fullscreen(screen)
    screen = pygame.display.get_surface()
    _frame(client, screen)
    assert client.view_w >= 1 and client.view_h >= 1

    pygame.quit()


def test_gl_software_draw_no_crash():
    client, _loop, _pid = _boot_client("gl")
    screen = pygame.display.get_surface()
    assert client._gl_force_software
    _frame(client, screen)
    pygame.quit()
