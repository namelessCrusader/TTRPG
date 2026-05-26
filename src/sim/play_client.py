"""
Unified pygame play client — iso, voxel, or GL view; one event loop and command set.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

try:
    import pygame
except ImportError as exc:
    raise ImportError("pygame is required") from exc

# Custom events — handlers run on the main pygame thread.
_EVENT_UI_MSG = pygame.USEREVENT + 10
_EVENT_TURN_DONE = pygame.USEREVENT + 11
_EVENT_GL_DONE = pygame.USEREVENT + 12

from .play_commands import (
    handle_meta_command,
    handle_persistence_command,
    help_lines,
    run_player_turn,
)
from .schemas import SpatialGrid
from .voxel_layers import LayerNavigator
from .voxel_draw import (
    C_BG_VOID,
    draw_layer_banner,
    draw_voxel_isometric,
    snap_camera_to,
)

# Pygame colour palette (shared)
C_TEXT = (210, 215, 225)
C_TEXT_DIM = (120, 130, 150)
C_MSG_AMBIENT = (160, 200, 180)
C_MSG_PLAYER = (120, 180, 255)
C_MSG_NPC = (255, 210, 120)
C_MSG_REJECT = (255, 100, 100)
C_MSG_GOOD = (90, 230, 180)
C_INPUT_BG = (22, 26, 38)
C_INPUT_BORDER = (60, 80, 110)
C_HUD_BG = (28, 34, 52)
C_BG = (12, 14, 22)
C_GL_EMPTY = (32, 42, 58)

_PY_COLORS = {
    "normal": C_TEXT,
    "dim": C_TEXT_DIM,
    "ambient": C_MSG_AMBIENT,
    "player": C_MSG_PLAYER,
    "npc": C_MSG_NPC,
    "reject": C_MSG_REJECT,
    "good": C_MSG_GOOD,
}


class PlayClient:
    """
    Single pygame shell for Mark-1 worlds.

    view: ``iso`` (2D tile art), ``voxel`` (isometric blocks), ``gl`` (GPU).
    """

    HUD_W = 300
    INPUT_H = 44
    MSG_LINES = 12
    HUD_MSG_H = 200
    MENU_TOP = 72

    def __init__(
        self,
        game_loop,
        *,
        view: str = "auto",
        adapter_label: str = "LM",
        window_w: int = 1280,
        window_h: int = 800,
        session=None,
    ) -> None:
        self.loop = game_loop
        self.session = session
        self.world = game_loop.world
        self.player_id = game_loop.player_id
        self.adapter_label = adapter_label
        self.grid: SpatialGrid = self.world.spatial

        if view == "auto":
            from .cli import resolve_pygame_view
            class _A:
                gl = voxel = iso = False
                view = "auto"
            view = resolve_pygame_view(_A(), self.grid)

        self.view = view
        self.win_w = window_w
        self.win_h = window_h
        self._windowed_w = window_w
        self._windowed_h = window_h
        self.view_w = max(1, window_w - self.HUD_W)
        self.view_h = max(1, window_h - self.INPUT_H)
        self._view_buffer: Optional[pygame.Surface] = None

        self.layers: Optional[LayerNavigator] = None
        if self.grid.use_voxels or self.grid.depth > 1 or view in ("voxel", "gl"):
            self.layers = LayerNavigator(
                depth=max(1, self.grid.depth),
                slice_z=0 if self.grid.depth > 1 else None,
            )
            # GL: show full volume by default (single Z slice looks empty)
            if view == "gl":
                self.layers.show_all()

        self._cam_x = 0
        self._cam_y = 0
        self._view_yaw = 0.0
        self._view_zoom = 1.0
        self._ZOOM_MIN = 0.35
        self._ZOOM_MAX = 3.0
        self._messages: deque[tuple[str, tuple]] = deque(maxlen=150)
        self._input_text = ""
        self._drag_mode: Optional[str] = None  # pan | orbit | rotate
        self._last_mx = self._last_my = 0
        self._hud_scroll = 0
        self._hover_button = -1
        self._buttons: list[tuple[pygame.Rect, str, str]] = []
        self._font: Optional[pygame.font.Font] = None
        self._font_sm: Optional[pygame.font.Font] = None
        self._world_lock = threading.Lock()
        self._turn_busy = False
        self._gl_busy = False
        self._fullscreen = False
        self._gl_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mark1-gl")
        self._gl_frame_future = None

        self._iso = None
        self._gl = None
        self._gl_ctx = None
        self._gl_fbo = None
        # The GPU renderer is still experimental on Linux/pygame; boot GL mode
        # into the known-good voxel renderer so the window is never black.
        # Press G in the window to attempt the GPU backend explicitly.
        self._gl_force_software = view == "gl"

        if view == "iso":
            from .iso_renderer import IsoRenderer, build_tile_cache
            build_tile_cache()
            self._iso = IsoRenderer(game_loop, window_w=window_w, window_h=window_h)
            self._iso.layers = self.layers
            self._iso._voxel_mode = bool(self.grid.use_voxels)
        if view in ("voxel", "gl") and self.layers:
            self._snap_camera()
        if view == "voxel" and self.layers:
            z = self.layers.slice_z
            self._cam_x, self._cam_y = snap_camera_to(
                self.grid, self.player_id, z=z,
            )

    def _init_gl_backend(self) -> None:
        from .voxel_gl.client import require_gl
        from .voxel_gl.renderer import VoxelGLRenderer
        import moderngl

        require_gl()

        if self._gl_ctx is not None:
            return

        # Keep ModernGL fully offscreen. Creating/borrowing a window GL context
        # after opening a normal pygame display can make pygame blits disappear.
        self._gl_ctx = moderngl.create_standalone_context()
        self._gl = VoxelGLRenderer(self._gl_ctx)
        self._gl.focus_grid(self.grid)
        self._resize_gl_fbo(max(1, self.view_w), max(1, self.view_h))
        self._snap_gl_target()
        self._gl.rebuild_instances(self.grid, self.layers)
        if self._gl._instance_count == 0 and not self.grid.use_voxels:
            self._messages.append(
                ("No voxels in this world — try: ./run_sim.sh --pygame --gl voxel_tavern", C_MSG_AMBIENT)
            )

    def _resize_gl_fbo(self, w: int, h: int) -> None:
        if self._gl_ctx is None:
            return
        if self._gl_fbo is not None:
            self._gl_fbo.release()
        tex = self._gl_ctx.texture((max(1, w), max(1, h)), 3)
        depth = self._gl_ctx.depth_renderbuffer((max(1, w), max(1, h)))
        self._gl_fbo = self._gl_ctx.framebuffer([tex], depth)
        self._gl_tex = tex

    def _snap_gl_target(self) -> None:
        if self._gl is None:
            return
        if self.player_id and self.player_id in self.grid.entities:
            p = self.grid.entities[self.player_id]
            import numpy as np
            self._gl.camera.target = np.array(
                [p.position.x, p.position.z, p.position.y], dtype=np.float32
            )

    def _push_message(self, text: str, color: str = "normal") -> None:
        """Append to the log (main thread only)."""
        rgb = _PY_COLORS.get(color, C_TEXT)
        for line in str(text).splitlines():
            if line.strip():
                self._messages.append((line, rgb))

    def _emit(self, text: str, color: str = "normal") -> None:
        """Thread-safe: queue a message for the next frame."""
        pygame.event.post(
            pygame.event.Event(_EVENT_UI_MSG, {"text": str(text), "color": color})
        )

    def _snap_camera(self) -> None:
        if self.view == "iso" and self._iso:
            self._iso._snap_camera()
            self._cam_x = self._iso._cam_x
            self._cam_y = self._iso._cam_y
        elif self.view == "voxel" and self.layers:
            z = self.layers.slice_z
            self._cam_x, self._cam_y = snap_camera_to(
                self.grid, self.player_id, z=z,
            )
        elif self.view == "gl":
            self._snap_gl_target()

    def _on_world_changed(self) -> None:
        with self._world_lock:
            self._snap_camera()
            if self._gl and self.layers:
                self._gl.rebuild_instances(self.grid, self.layers)

    def _process_command(self, raw: str) -> None:
        if handle_persistence_command(
            raw,
            session=self.session,
            world=self.world,
            loop=self.loop,
            emit=self._push_message,
        ):
            self.world = self.loop.world
            if self.session is not None:
                self.session.world = self.world
                self.session.loop = self.loop
                self.player_id = self.loop.player_id
            self._on_world_changed()
            return
        if handle_meta_command(
            raw,
            world=self.world,
            loop=self.loop,
            player_id=self.player_id,
            layers=self.layers,
            emit=self._push_message,
            pygame_quit=True,
        ):
            if raw.lower().strip().startswith("layer"):
                self._on_world_changed()
            return
        self._run_turn_async(raw)

    def _sync_iso_camera(self) -> None:
        if self._iso is not None:
            self._iso._cam_x = self._cam_x
            self._iso._cam_y = self._cam_y

    def _hud_messages_y(self) -> int:
        return max(self.MENU_TOP, self.view_h - self.HUD_MSG_H)

    def _activate_button(self, command: str) -> None:
        """Run a HUD/menu command without requiring text input."""
        if command == "__clear__":
            self._messages.clear()
            return
        if command == "__toggle_gpu__":
            self._request_gpu_toggle()
            return
        if handle_meta_command(
            command,
            world=self.world,
            loop=self.loop,
            player_id=self.player_id,
            layers=self.layers,
            emit=self._push_message,
            pygame_quit=True,
        ):
            if command.lower().strip().startswith("layer"):
                self._on_world_changed()
            self._push_message(f"✓ {command}", "good")
            return

        self._run_turn_async(command)

    def _is_instant_command(self, command: str) -> bool:
        """Commands that bypass the LM and should run synchronously."""
        from .intent_router import try_parse_compass_step, try_parse_trivial_actions

        low = command.lower().strip()
        if low.startswith("layer ") or low in (
            "map", "m", "inventory", "inv", "i", "status", "log", "goals", "help",
            "chronicle", "metrics", "lm", "checkpoint", "stream",
            "quit", "exit", "q",
        ) or low.startswith("save") or low.startswith("load "):
            return True
        if try_parse_trivial_actions(command, self.player_id):
            return True
        ent = self.grid.entities.get(self.player_id) if self.player_id else None
        if ent is not None and try_parse_compass_step(command, self.player_id, ent):
            return True
        return False

    def _run_turn_async(self, command: str) -> None:
        if self._turn_busy:
            self._push_message("(still processing previous action…)", "dim")
            return
        self._turn_busy = True
        self._push_message(f"» {command}", "normal")

        if self._is_instant_command(command):
            try:
                with self._world_lock:
                    run_player_turn(
                        command, self.loop, self.player_id, self._push_message,
                        on_world_changed=None,
                        session=self.session,
                        story_mode=getattr(self.session, "story_mode", False),
                    )
            except Exception as exc:
                self._push_message(f"Error: {exc}", "reject")
            finally:
                self._turn_busy = False
                self._on_world_changed()
            return

        def work() -> None:
            try:
                with self._world_lock:
                    run_player_turn(
                        command, self.loop, self.player_id, self._push_message,
                        on_world_changed=None,
                        session=self.session,
                        story_mode=getattr(self.session, "story_mode", False),
                    )
            finally:
                pygame.event.post(pygame.event.Event(_EVENT_TURN_DONE))

        threading.Thread(target=work, daemon=True).start()

    def _request_gpu_toggle(self) -> None:
        if self._gl_busy:
            self._push_message("GPU toggle already in progress…", "dim")
            return
        if not self._gl_force_software:
            self._gl_force_software = True
            self._push_message("Software voxel renderer (stable).", "ambient")
            return
        self._gl_busy = True
        self._push_message("Trying GPU renderer (may take a few seconds)…", "dim")

        def work() -> None:
            ok, err = False, None
            try:
                self._init_gl_backend()
                if self._gl and self._gl_fbo:
                    w, h = max(1, self.view_w), max(1, self.view_h)
                    ok = self._test_gl_frame(w, h)
                    if not ok:
                        err = "GPU frame empty or unreadable"
            except Exception as exc:
                err = str(exc)
            pygame.event.post(
                pygame.event.Event(_EVENT_GL_DONE, {"ok": ok, "error": err})
            )

        self._gl_executor.submit(work)

    def _capture_gl_frame(self, w: int, h: int):
        """Run on GL worker thread; returns (rgb bytes array, w, h) or None."""
        if not self._gl or not self._gl_fbo or not self._gl_ctx:
            return None
        try:
            import numpy as np
            self._gl_fbo.use()
            self._gl.draw(viewport_size=(w, h))
            self._gl_ctx.finish()
            raw = bytes(self._gl_fbo.read(components=3, alignment=1))
            if len(raw) != w * h * 3:
                return None
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            arr = np.flipud(arr)
            if int(arr.max()) < 55 or int((arr.max(axis=2) > 55).sum()) < 200:
                return None
            return arr, w, h
        except Exception:
            return None

    def _test_gl_frame(self, w: int, h: int) -> bool:
        if not self._gl or not self._gl_fbo or not self._gl_ctx:
            return False
        self._gl_fbo.use()
        self._gl.draw(viewport_size=(w, h))
        self._gl_ctx.finish()
        raw = bytes(self._gl_fbo.read(components=3, alignment=1))
        if len(raw) != w * h * 3:
            return False
        import numpy as np
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        visible = int((arr.max(axis=2) > 55).sum())
        return int(arr.max()) >= 55 and visible >= 200

    def _button_at(self, mx: int, my: int) -> Optional[int]:
        for i, (rect, _label, _cmd) in enumerate(self._buttons):
            if rect.collidepoint(mx, my):
                return i
        return None

    def _pan_view(self, dx: int, dy: int) -> None:
        # Pan in scene space (compensate for screen-space yaw rotation).
        yaw = self._view_yaw
        c, s = math.cos(yaw), math.sin(yaw)
        rdx = dx * c - dy * s
        rdy = dx * s + dy * c
        self._cam_x -= int(rdx)
        self._cam_y -= int(rdy)
        self._sync_iso_camera()

    def _rotate_view(self, dx: int, dy: int = 0) -> None:
        """Smooth orbit: horizontal drag = yaw, vertical = slight zoom."""
        self._view_yaw += dx * 0.004
        if dy:
            self._zoom_view(-dy * 0.002)
        self._last_mx += dx
        self._last_my += dy

    def _zoom_view(self, delta: float) -> None:
        """delta > 0 zooms in, < 0 zooms out."""
        if delta == 0:
            return
        factor = math.exp(delta)
        self._view_zoom = max(
            self._ZOOM_MIN,
            min(self._ZOOM_MAX, self._view_zoom * factor),
        )
        if self.view == "gl" and self._gl and not self._gl_force_software:
            self._gl.camera.distance = max(
                6.0,
                min(80.0, self._gl.camera.distance / factor),
            )

    def _gl_pan_target(self, dx: int, dy: int) -> None:
        if not self._gl:
            return
        import math
        yaw = self._gl.camera.yaw
        right = math.cos(yaw)
        forward = math.sin(yaw)
        scale = 0.08
        self._gl.camera.target[0] += (dx * right - dy * forward) * scale
        self._gl.camera.target[2] += (dx * forward + dy * right) * scale

    def _layer_key(self, event: pygame.event.Event) -> None:
        if not self.layers:
            return
        if event.key == pygame.K_PAGEUP:
            self.layers.up()
        elif event.key == pygame.K_PAGEDOWN:
            self.layers.down()
        elif event.key == pygame.K_HOME:
            self.layers.ground()
        elif event.key == pygame.K_END:
            self.layers.top()
        elif event.key == pygame.K_0:
            self.layers.show_all()
        else:
            return
        self._snap_camera()
        if self._gl:
            self._gl.rebuild_instances(self.grid, self.layers)
        self._emit(f"Layer: {self.layers.label()}", "ambient")

    def _handle_key(self, event: pygame.event.Event) -> bool:
        if event.key == pygame.K_ESCAPE:
            return False
        if event.key in (pygame.K_q, pygame.K_e):
            step = math.pi / 2 if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else math.pi / 12
            self._view_yaw += step if event.key == pygame.K_e else -step
            return True
        if event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self._zoom_view(0.12)
            return True
        if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self._zoom_view(-0.12)
            return True
        if event.key == pygame.K_r:
            self._view_yaw = 0.0
            self._view_zoom = 1.0
            self._snap_camera()
            self._push_message("Camera reset.", "dim")
            return True
        if event.key == pygame.K_F11:
            return True  # handled in run() where screen exists
        if self.view == "gl" and event.key == pygame.K_g:
            self._request_gpu_toggle()
            return True
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            text = self._input_text.strip()
            self._input_text = ""
            if text:
                self._process_command(text)
            return True
        if event.key == pygame.K_BACKSPACE:
            self._input_text = self._input_text[:-1]
            return True
        if event.key in (
            pygame.K_PAGEUP, pygame.K_PAGEDOWN,
            pygame.K_HOME, pygame.K_END, pygame.K_0,
        ):
            self._layer_key(event)
            return True
        if event.unicode and event.unicode.isprintable():
            self._input_text += event.unicode
        return True

    def _sync_display_geometry(self, screen: Optional[pygame.Surface] = None) -> None:
        """Refresh layout sizes after set_mode / resize."""
        surf = screen or pygame.display.get_surface()
        if surf is None:
            return
        self.win_w = max(self.HUD_W + 64, surf.get_width())
        self.win_h = max(self.INPUT_H + 64, surf.get_height())
        self.view_w = max(1, self.win_w - self.HUD_W)
        self.view_h = max(1, self.win_h - self.INPUT_H)
        self._view_buffer = None
        if self.view == "gl":
            self._resize_gl_fbo(self.view_w, self.view_h)

    def _toggle_fullscreen(self, screen: pygame.Surface) -> None:
        try:
            if self._fullscreen:
                pygame.display.set_mode(
                    (self._windowed_w, self._windowed_h),
                    pygame.RESIZABLE,
                )
                self._fullscreen = False
            else:
                self._windowed_w, self._windowed_h = self.win_w, self.win_h
                pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                self._fullscreen = True
            screen = pygame.display.get_surface()
            self._sync_display_geometry(screen)
            label = (
                "Fullscreen on (F11 to exit)"
                if self._fullscreen
                else "Windowed (F11 for fullscreen)"
            )
            self._push_message(label, "dim")
        except pygame.error as exc:
            self._fullscreen = False
            self._push_message(f"Fullscreen failed: {exc}", "reject")

    def _draw_voxel_scene(self, target: pygame.Surface) -> None:
        """Render voxels with a yaw-aware projection (not a bitmap rotation)."""
        draw_voxel_isometric(
            target,
            self.grid,
            self.layers,
            cam_x=self._cam_x,
            cam_y=self._cam_y,
            player_id=self.player_id,
            view_yaw=self._view_yaw,
            view_zoom=self._view_zoom,
            font=self._font_sm,
        )
        if self.layers:
            draw_layer_banner(
                target, self.layers, self._font_sm, world_name=self.world.name
            )
        self._draw_action_hint(target)

    def _draw_action_hint(self, view: pygame.Surface) -> None:
        """Brief viewport hint when the last message was a rejected action."""
        if not self._font_sm or not self._messages:
            return
        text, color = self._messages[-1]
        if color != C_MSG_REJECT and "✗" not in text:
            return
        hint = self._font_sm.render(text[:64], True, color)
        pygame.draw.rect(
            view, (18, 22, 34),
            (8, view.get_height() - hint.get_height() - 14, hint.get_width() + 12, hint.get_height() + 8),
            border_radius=4,
        )
        view.blit(hint, (14, view.get_height() - hint.get_height() - 10))

    def _draw_world(self, screen: pygame.Surface) -> None:
        view = screen.subsurface((0, 0, self.view_w, self.view_h))
        with self._world_lock:
            self._draw_world_locked(view)

    def _draw_world_locked(self, view: pygame.Surface) -> None:
        if self.view == "iso" and self._iso:
            self._iso._render_world(view)
            if self.layers:
                draw_layer_banner(view, self.layers, self._font_sm, world_name=self.world.name)
        elif self.view == "voxel":
            self._draw_voxel_scene(view)
        elif self.view == "gl":
            self._draw_gl_view(view)

    def _draw_gl_view(self, view: pygame.Surface) -> None:
        """Render GPU frame to pygame (with iso fallback if GPU read fails)."""
        w, h = max(1, self.view_w), max(1, self.view_h)

        if self._gl_busy:
            self._draw_software_gl_fallback(view, "Loading GPU renderer…")
            return

        if self._gl_force_software:
            self._draw_software_gl_fallback(view, "Software voxel · F11 fullscreen · G try GPU")
            return

        if not self._gl or not self._gl_fbo:
            self._draw_software_gl_fallback(view, "GL backend not initialized")
            return

        if self._gl._instance_count == 0:
            self._draw_software_gl_fallback(
                view,
                "No solid voxels — use ./run_sim.sh --pygame --gl voxel_tavern",
                (255, 220, 140),
            )
            return

        blitted = False
        if self._gl_frame_future is not None and self._gl_frame_future.done():
            try:
                payload = self._gl_frame_future.result(timeout=0)
            except Exception as exc:
                self._gl_force_software = True
                self._draw_software_gl_fallback(view, f"GPU error: {exc}", (255, 120, 120))
                self._gl_frame_future = None
                return
            self._gl_frame_future = None
            if payload is None:
                self._gl_force_software = True
                self._draw_software_gl_fallback(
                    view, "GPU frame failed — software fallback", (200, 220, 255),
                )
                return
            arr, pw, ph = payload
            view.blit(
                pygame.image.frombuffer(arr.tobytes(), (pw, ph), "RGB").copy(),
                (0, 0),
            )
            blitted = True

        if self._gl_frame_future is None:
            self._gl_frame_future = self._gl_executor.submit(
                self._capture_gl_frame, w, h
            )

        if not blitted:
            self._draw_software_gl_fallback(view, "GPU rendering…", (200, 220, 255))
            return

        if self.layers:
            draw_layer_banner(
                view, self.layers, self._font_sm, world_name=self.world.name
            )

    def _draw_software_gl_fallback(
        self,
        view: pygame.Surface,
        reason: str,
        color: tuple = (200, 220, 255),
    ) -> None:
        if self.layers:
            self.layers.show_all()
        if self._cam_x == 0 and self._cam_y == 0:
            self._cam_x, self._cam_y = snap_camera_to(
                self.grid, self.player_id, z=None,
            )
        self._draw_voxel_scene(view)
        self._draw_gl_overlay(view, reason, color)

    def _draw_gl_overlay(self, view: pygame.Surface, text: str, color: tuple) -> None:
        if not self._font_sm:
            return
        y = 12
        for chunk in text.replace(" — ", ". ").split(". "):
            line = chunk.strip()
            if not line:
                continue
            view.blit(self._font_sm.render(line + ".", True, color), (12, y))
            y += 18

    def _draw_hud(self, screen: pygame.Surface) -> None:
        hud = screen.subsurface((self.view_w, 0, self.HUD_W, self.view_h))
        hud.fill(C_HUD_BG)
        self._buttons.clear()
        y = 10
        view_label = {"iso": "2D Iso", "voxel": "Voxel", "gl": "Voxel GL"}.get(self.view, self.view)
        if self._font:
            lines = [
                f"Mark-1 · {view_label}",
                self.world.name,
                f"LM: {self.adapter_label[:26]}",
            ]
            if self.layers:
                lines.append(f"View: {self.layers.label()}")
            for line in lines:
                hud.blit(self._font.render(line, True, C_TEXT_DIM), (10, y))
                y += 18

        msg_y = self._hud_messages_y()
        hud.set_clip(pygame.Rect(0, self.MENU_TOP, self.HUD_W, max(0, msg_y - self.MENU_TOP)))

        y = self.MENU_TOP + 8
        y = self._draw_menu_section(
            hud,
            y,
            "Panels",
            [
                ("Map", "map"),
                ("Inventory", "inventory"),
                ("Status", "status"),
                ("Log", "log"),
                ("Goals", "goals"),
                ("Clear log", "__clear__"),
            ],
            msg_y=msg_y,
        )

        if self.layers:
            y = self._draw_menu_section(
                hud,
                y + 4,
                "Z Layers",
                [
                    ("Layer up", "layer up"),
                    ("Layer down", "layer down"),
                    ("Ground", "layer ground"),
                    ("Top", "layer top"),
                    ("All layers", "layer all"),
                ],
                msg_y=msg_y,
            )

        y = self._draw_menu_section(
            hud,
            y + 4,
            "Actions",
            [
                ("Look around", "observe"),
                ("Wait", "wait"),
                ("Move north", "move north"),
                ("Move south", "move south"),
                ("Move east", "move east"),
                ("Move west", "move west"),
            ],
            msg_y=msg_y,
        )

        if self.view == "gl":
            self._draw_menu_section(
                hud,
                y + 4,
                "Render",
                [("Try GPU / Toggle (G)", "__toggle_gpu__")],
                msg_y=msg_y,
            )

        hud.set_clip(None)

        if self._font_sm:
            pygame.draw.line(hud, (50, 60, 85), (6, msg_y - 2), (self.HUD_W - 6, msg_y - 2), 1)
            hud.blit(self._font_sm.render("Messages", True, C_TEXT_DIM), (8, msg_y + 4))
            my = msg_y + 20
            for text, color in list(self._messages)[-self.MSG_LINES :]:
                if my + 14 > self.view_h:
                    break
                limit = 72 if color == C_MSG_REJECT else 52
                hud.blit(self._font_sm.render(text[:limit], True, color), (8, my))
                my += 15

    def _draw_menu_section(
        self,
        hud: pygame.Surface,
        y: int,
        title: str,
        items: list[tuple[str, str]],
        *,
        msg_y: int,
    ) -> int:
        """Draw full-width HUD buttons and register screen-space click targets."""
        if not self._font_sm:
            return y

        pad = 8
        row_h = 22
        gap = 3
        title_y = y - self._hud_scroll
        if self.MENU_TOP <= title_y < msg_y:
            hud.blit(self._font_sm.render(title, True, C_TEXT_DIM), (10, title_y))
        y += 16

        btn_index = len(self._buttons)
        for label, command in items:
            vis_y = y - self._hud_scroll
            rect = pygame.Rect(pad, vis_y, self.HUD_W - pad * 2, row_h)
            screen_rect = pygame.Rect(
                self.view_w + pad, vis_y, self.HUD_W - pad * 2, row_h,
            )
            if rect.bottom >= self.MENU_TOP and rect.top < msg_y:
                fill = (58, 72, 108) if btn_index == self._hover_button else (38, 48, 72)
                pygame.draw.rect(hud, fill, rect, border_radius=3)
                pygame.draw.rect(hud, (90, 110, 150), rect, 1, border_radius=3)
                txt = self._font_sm.render(label, True, C_TEXT)
                hud.blit(
                    txt,
                    (rect.x + 8, rect.y + (rect.h - txt.get_height()) // 2),
                )
            self._buttons.append((screen_rect, label, command))
            btn_index += 1
            y += row_h + gap
        return y + 4

    def _draw_input(self, screen: pygame.Surface) -> None:
        bar = screen.subsurface((0, self.view_h, self.win_w, self.INPUT_H))
        bar.fill(C_INPUT_BG)
        pygame.draw.line(bar, C_INPUT_BORDER, (0, 0), (self.win_w, 0), 1)
        if self._font:
            bar.blit(self._font.render(f"> {self._input_text}_", True, C_TEXT), (12, 10))

    def run(self) -> None:
        pygame.init()
        titles = {"iso": "Isometric", "voxel": "Voxel", "gl": "Voxel GL"}
        pygame.display.set_caption(
            f"Mark-1 · {titles.get(self.view, 'Play')} · {self.world.name}"
        )
        screen = pygame.display.set_mode((self.win_w, self.win_h), pygame.RESIZABLE)
        clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("monospace", 14)
        self._font_sm = pygame.font.SysFont("monospace", 11)

        if self.view == "gl" and not self._gl_force_software:
            self._init_gl_backend()

        if self.view == "gl" and self._gl:
            self._snap_camera()
            self._resize_gl_fbo(max(1, self.view_w), max(1, self.view_h))
            self._gl.rebuild_instances(self.grid, self.layers)
            n = self._gl._instance_count
            self._emit(
                f"GL: {n} voxels, {self.layers.label() if self.layers else 'no layers'}. "
                "Drag=orbit, wheel=zoom.",
                "ambient",
            )
        elif self.view == "gl":
            self._emit(
                "GL mode: showing software voxel renderer. Press G to try GPU.",
                "ambient",
            )

        self._emit(f"Loaded {self.world.name}.", "ambient")
        if getattr(self.session, "story_mode", False):
            from .session_chronicle import format_scene_opening
            for line in format_scene_opening(self.world):
                self._push_message(line, "ambient")
        self._push_message(
            "Viewport: drag=pan · right-drag=rotate · wheel=zoom · Shift+wheel=layers",
            "dim",
        )
        self._push_message(
            "Keys: +/- zoom · Q/E rotate · Shift+Q/E = 90° · R reset camera",
            "dim",
        )
        for line in help_lines(self.layers is not None)[:5]:
            self._emit(line, "dim")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    if self._fullscreen:
                        screen = pygame.display.get_surface()
                        self._sync_display_geometry(screen)
                    else:
                        self._windowed_w, self._windowed_h = max(640, event.w), max(480, event.h)
                        screen = pygame.display.set_mode(
                            (self._windowed_w, self._windowed_h),
                            pygame.RESIZABLE,
                        )
                        self._sync_display_geometry(screen)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11:
                        self._toggle_fullscreen(screen)
                        screen = pygame.display.get_surface()
                    running = self._handle_key(event)
                elif event.type == _EVENT_UI_MSG:
                    self._push_message(event.text, getattr(event, "color", "normal"))
                elif event.type == _EVENT_TURN_DONE:
                    self._turn_busy = False
                    self._on_world_changed()
                elif event.type == _EVENT_GL_DONE:
                    self._gl_busy = False
                    if event.ok:
                        self._gl_force_software = False
                        if self._gl and self.layers:
                            with self._world_lock:
                                self._gl.rebuild_instances(self.grid, self.layers)
                        self._push_message("GPU renderer enabled.", "good")
                    else:
                        self._gl_force_software = True
                        detail = getattr(event, "error", None) or "unknown error"
                        self._push_message(
                            f"GPU unavailable — staying on software renderer ({detail})",
                            "reject",
                        )
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    btn = event.button
                    if mx >= self.view_w:
                        if btn == 1:
                            hit = self._button_at(mx, my)
                            if hit is not None:
                                self._activate_button(self._buttons[hit][2])
                        continue
                    gl_orbit = (
                        self.view == "gl"
                        and not self._gl_force_software
                        and self._gl is not None
                    )
                    if btn == 1:
                        self._drag_mode = "orbit" if gl_orbit else "pan"
                        self._last_mx, self._last_my = mx, my
                    elif btn in (3, 2):
                        self._drag_mode = "gl_pan" if gl_orbit else "rotate"
                        self._last_mx, self._last_my = mx, my
                elif event.type == pygame.MOUSEBUTTONUP and event.button in (1, 2, 3):
                    self._drag_mode = None
                elif event.type == pygame.MOUSEMOTION:
                    mx, my = pygame.mouse.get_pos()
                    if mx >= self.view_w:
                        self._hover_button = self._button_at(mx, my) or -1
                    else:
                        self._hover_button = -1
                    if self._drag_mode and mx < self.view_w:
                        dx, dy = mx - self._last_mx, my - self._last_my
                        if self._drag_mode == "pan":
                            self._pan_view(dx, dy)
                        elif self._drag_mode == "rotate":
                            self._rotate_view(dx, dy)
                        elif self._drag_mode == "gl_pan":
                            self._gl_pan_target(dx, dy)
                        elif self._drag_mode == "orbit" and self._gl:
                            self._gl.camera.yaw += dx * 0.01
                            self._gl.camera.pitch = max(
                                0.15,
                                min(1.4, self._gl.camera.pitch + dy * 0.01),
                            )
                        self._last_mx, self._last_my = mx, my
                elif event.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    mods = pygame.key.get_mods()
                    if mx >= self.view_w:
                        self._hud_scroll = max(
                            0,
                            self._hud_scroll - event.y * 24,
                        )
                    elif mods & pygame.KMOD_SHIFT and self.layers:
                        if event.y > 0:
                            self.layers.up()
                        elif event.y < 0:
                            self.layers.down()
                        self._on_world_changed()
                    else:
                        self._zoom_view(event.y * 0.08)

            screen.fill(C_BG)
            self._draw_world(screen)
            self._draw_hud(screen)
            self._draw_input(screen)
            pygame.display.flip()
            clock.tick(60)

        self._gl_executor.shutdown(wait=False)
        if self._gl_fbo:
            self._gl_fbo.release()
        if self._gl_ctx:
            self._gl_ctx.release()
        pygame.quit()


def run_view_only(world, *, width: int = 1280, height: int = 800) -> None:
    """Map viewer without game loop."""
    grid = world.spatial
    layers = LayerNavigator(depth=max(1, grid.depth), slice_z=0)
    cam_x, cam_y = snap_camera_to(grid, None)
    pygame.init()
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()
    view_h = height - 36
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_PAGEUP:
                    layers.up()
                elif event.key == pygame.K_PAGEDOWN:
                    layers.down()
                elif event.key == pygame.K_HOME:
                    layers.ground()
                elif event.key == pygame.K_END:
                    layers.top()
                elif event.key == pygame.K_0:
                    layers.show_all()
            elif event.type == pygame.VIDEORESIZE:
                width, height = event.w, event.h
                screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                view_h = height - 36
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    layers.up()
                elif event.y < 0:
                    layers.down()
        view = pygame.Surface((width, view_h))
        draw_voxel_isometric(view, grid, layers, cam_x=cam_x, cam_y=cam_y)
        draw_layer_banner(view, layers, font, world_name=world.name)
        screen.fill(C_BG)
        screen.blit(view, (0, 0))
        screen.blit(
            font.render("PgUp/Dn layers · Esc quit", True, C_TEXT_DIM),
            (12, height - 28),
        )
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
