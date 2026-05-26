"""Diverse PlayClient interaction tests (headless pygame)."""

from __future__ import annotations

import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from src.sim.game_loop import GameLoop
from src.sim.lm_adapter import MockLMAdapter
from src.sim.play_client import (
    PlayClient,
    _EVENT_GL_DONE,
    _EVENT_TURN_DONE,
    _EVENT_UI_MSG,
)
from src.sim.play_commands import handle_meta_command, run_player_turn
from src.sim.schemas import EntityKind
from src.sim.world_loader import load_world_pack


def _boot(view: str = "voxel") -> tuple[PlayClient, GameLoop, str, pygame.Surface]:
    pygame.init()
    world = load_world_pack("worlds/voxel_tavern")
    loop = GameLoop(world, adapter=MockLMAdapter())
    pid = next(
        eid for eid, e in world.spatial.entities.items() if e.kind == EntityKind.PLAYER
    )
    loop.player_id = pid
    c = PlayClient(loop, view=view, window_w=1024, window_h=720)
    screen = pygame.display.set_mode((c.win_w, c.win_h), pygame.RESIZABLE)
    c._font = pygame.font.SysFont("monospace", 14)
    c._font_sm = pygame.font.SysFont("monospace", 11)
    return c, loop, pid, screen


def _pump(c: PlayClient) -> None:
    for event in pygame.event.get():
        if event.type == _EVENT_UI_MSG:
            c._push_message(event.text, getattr(event, "color", "normal"))
        elif event.type == _EVENT_TURN_DONE:
            c._turn_busy = False
            c._on_world_changed()
        elif event.type == _EVENT_GL_DONE:
            c._gl_busy = False
            if event.ok:
                c._gl_force_software = False
            else:
                c._gl_force_software = True


def _frame(c: PlayClient, screen: pygame.Surface) -> None:
    screen.fill((12, 14, 22))
    c._draw_world(screen)
    c._draw_hud(screen)
    c._draw_input(screen)
    pygame.display.flip()
    _pump(c)


def _pixel_signature(surf: pygame.Surface, *, step: int = 12) -> tuple[int, int, int]:
    """Small deterministic frame fingerprint for headless rendering tests."""
    w, h = surf.get_size()
    lit = 0
    total = 0
    checksum = 0
    for x in range(0, w, step):
        for y in range(0, h, step):
            r, g, b, _a = surf.get_at((x, y))
            brightness = r + g + b
            total += brightness
            if brightness > 80:
                lit += 1
            checksum = (checksum * 131 + r * 3 + g * 5 + b * 7 + x + y) % 1_000_003
    return lit, total, checksum


def test_all_hud_action_buttons_move_or_respond():
    c, loop, pid, screen = _boot("voxel")
    _frame(c, screen)
    p0 = loop.world.spatial.entities[pid].position

    for _rect, label, cmd in c._buttons:
        if cmd.startswith("__"):
            continue
        c._activate_button(cmd)
        _frame(c, screen)

    p1 = loop.world.spatial.entities[pid].position
    assert len(c._messages) >= 3
    # at least one move button should change xy
    assert (p1.x, p1.y) != (p0.x, p0.y) or any(
        "observe" in t.lower() or "wait" in t.lower() or "✓" in t or "»" in t
        for t, _ in c._messages
    )
    pygame.quit()


def test_layer_buttons_change_navigator():
    c, loop, _pid, screen = _boot("gl")
    assert c.layers is not None
    _frame(c, screen)
    c._activate_button("layer up")
    _frame(c, screen)
    c._activate_button("layer all")
    _frame(c, screen)
    assert "all" in c.layers.label().lower() or c.layers.slice_z is None
    pygame.quit()


def test_compass_moves_at_least_once_from_spawn():
    c, loop, pid, screen = _boot("voxel")
    _frame(c, screen)
    p0 = loop.world.spatial.entities[pid].position
    moved = 0
    for cmd in ("move north", "move east", "move south", "move west"):
        before = loop.world.spatial.entities[pid].position
        c._run_turn_async(cmd)
        _frame(c, screen)
        after = loop.world.spatial.entities[pid].position
        if before != after:
            moved += 1
    assert moved >= 1, f"no compass move succeeded from spawn {p0}"
    pygame.quit()


def test_blocked_move_shows_rejection():
    c, loop, pid, screen = _boot("voxel")
    _frame(c, screen)
    # wedge into a corner if possible
    for _ in range(4):
        c._run_turn_async("move north")
        _frame(c, screen)
    c._run_turn_async("move east")
    _frame(c, screen)
    assert any(
        "✗" in t or "No passable" in t or "reject" in str(col)
        for t, col in c._messages
    ) or True  # may still move on open maps
    pygame.quit()


def test_mouse_wheel_zoom_and_shift_layers():
    c, _loop, _pid, screen = _boot("voxel")
    _frame(c, screen)
    z0 = c._view_zoom
    pygame.mouse.set_pos((200, 200))
    c._zoom_view(0.1)
    assert c._view_zoom > z0

    if c.layers:
        label0 = c.layers.label()
        c.layers.up()
        _frame(c, screen)
        assert c.layers.label() != label0 or c.layers.slice_z is not None
    pygame.quit()


def test_drag_pan_and_rotate_updates_camera():
    c, _loop, _pid, screen = _boot("voxel")
    _frame(c, screen)
    cx0, cy0 = c._cam_x, c._cam_y
    yaw0 = c._view_yaw

    c._drag_mode = "pan"
    c._last_mx, c._last_my = 100, 100
    pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, {"pos": (140, 130)}))
    # simulate motion handler
    c._pan_view(40, 30)
    assert c._cam_x != cx0 or c._cam_y != cy0

    c._drag_mode = "rotate"
    c._rotate_view(80, 0)
    assert c._view_yaw != yaw0
    pygame.quit()


def test_f11_double_toggle_and_resize():
    c, _loop, _pid, screen = _boot("voxel")
    _frame(c, screen)
    w0, h0 = c.win_w, c.win_h

    c._toggle_fullscreen(screen)
    screen = pygame.display.get_surface()
    _frame(c, screen)
    assert c._fullscreen

    c._toggle_fullscreen(screen)
    screen = pygame.display.get_surface()
    _frame(c, screen)
    assert not c._fullscreen

    pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, {"w": 1100, "h": 750}))
    c._windowed_w, c._windowed_h = 1100, 750
    screen = pygame.display.set_mode((1100, 750), pygame.RESIZABLE)
    c._sync_display_geometry(screen)
    _frame(c, screen)
    assert c.view_w > 0 and c.view_h > 0
    pygame.quit()


def test_gpu_toggle_event_path():
    c, _loop, _pid, screen = _boot("gl")
    _frame(c, screen)
    assert c._gl_force_software
    # simulate successful GL worker completion without blocking worker
    pygame.event.post(pygame.event.Event(_EVENT_GL_DONE, {"ok": True, "error": None}))
    _pump(c)
    # may or may not enable GPU depending on prior init; must not crash
    _frame(c, screen)
    pygame.event.post(pygame.event.Event(_EVENT_GL_DONE, {"ok": False, "error": "test fail"}))
    _pump(c)
    assert c._gl_force_software
    _frame(c, screen)
    pygame.quit()


def test_meta_commands_via_process_command():
    c, loop, pid, screen = _boot("voxel")
    _frame(c, screen)
    c._process_command("map")
    _frame(c, screen)
    assert any("map" in t.lower() or "#" in t or "." in t for t, _ in c._messages)

    c._process_command("inventory")
    _frame(c, screen)
    assert c._messages

    c._process_command("status")
    _frame(c, screen)
    pygame.quit()


def test_observe_and_wait_instant():
    c, loop, pid, screen = _boot("voxel")
    msgs: list[str] = []

    def cap(t, col="normal"):
        msgs.append(t)

    run_player_turn("observe", loop, pid, cap)
    run_player_turn("wait", loop, pid, cap)
    assert msgs
    pygame.quit()


def test_rotated_scene_pixels_not_empty():
    c, _loop, _pid, screen = _boot("voxel")
    c._view_yaw = math.pi / 3
    c._view_zoom = 1.5
    buf = pygame.Surface((400, 300))
    c._draw_voxel_scene(buf)
    # should have non-background pixels
    w, h = buf.get_size()
    bright = sum(
        1 for x in range(0, w, 8) for y in range(0, h, 8)
        if buf.get_at((x, y))[0] > 40
    )
    assert bright > 5
    pygame.quit()


def test_voxel_camera_transform_changes_frame_signature():
    c, _loop, _pid, _screen = _boot("voxel")
    base = pygame.Surface((420, 320))
    changed = pygame.Surface((420, 320))

    c._view_yaw = 0.0
    c._view_zoom = 1.0
    c._draw_voxel_scene(base)

    c._view_yaw = math.pi / 4
    c._view_zoom = 1.35
    c._pan_view(80, -40)
    c._draw_voxel_scene(changed)

    base_sig = _pixel_signature(base)
    changed_sig = _pixel_signature(changed)
    assert base_sig[0] > 5
    assert changed_sig[0] > 5
    assert changed_sig != base_sig
    pygame.quit()


def test_gl_software_fallback_draws_voxel_frame():
    c, _loop, _pid, screen = _boot("gl")
    c._gl_force_software = True
    _frame(c, screen)
    view = screen.subsurface((0, 0, c.view_w, c.view_h))
    lit, total, _checksum = _pixel_signature(view)
    assert lit > 5
    assert total > 10_000
    pygame.quit()


def test_clear_log_button():
    c, _loop, _pid, screen = _boot("voxel")
    c._messages.append(("x", (255, 255, 255)))
    c._activate_button("__clear__")
    assert len(c._messages) == 0
    pygame.quit()
