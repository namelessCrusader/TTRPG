"""
Isometric 2.5D renderer for the simulation engine.

Replaces the ASCII TUI with a Pygame-based isometric view styled after
pixel-art dungeon games. The simulation backend (game_loop, compiler,
relational graph) is completely unchanged — this is purely a new visual
front-end that reads from the same WorldState.

Tile geometry:
    TILE_W = 64 px   (diamond width)
    TILE_H = 32 px   (diamond height, half of width for classic 2:1 iso)
    WALL_H = 36 px   (visible height of wall face below the diamond top)

Coordinate system:
    World (gx, gy) → screen (sx, sy):
        sx = (gx - gy) * TILE_W // 2
        sy = (gx + gy) * TILE_H // 2
    Final screen position adds camera offset and viewport centre.

Rendering order (painter's algorithm):
    Sort tiles by (gx + gy) ascending so far tiles are drawn before near ones.

Usage:
    python3 -m src.sim.iso_renderer [--ollama] [--model ...] [--world ...]
"""

from __future__ import annotations

import sys
import textwrap
from collections import deque
from pathlib import Path
from typing import Optional

import pygame

# ---------------------------------------------------------------------------
# Tile geometry constants
# ---------------------------------------------------------------------------

TILE_W: int = 64
TILE_H: int = 32
WALL_H: int = 36

# How many tiles to see on either side of the player (viewport radius).
VIEW_RADIUS: int = 12

# ---------------------------------------------------------------------------
# Colour palette — styled after the teal/cyan dungeon screenshot
# ---------------------------------------------------------------------------

C_BG          = (10,  22,  40)   # dark navy background
C_FLOOR_TOP   = (29, 210, 180)   # bright teal floor diamond
C_FLOOR_EDGE  = (12, 140, 120)   # slightly darker grid edge
C_FLOOR_DARK  = (18, 160, 138)   # floor tile when "outside" FOV (dim)
C_WALL_TOP    = (18, 175, 152)   # wall top face (slightly darker than floor)
C_WALL_LEFT   = (10, 100,  88)   # wall left face (darker — shadow side)
C_WALL_RIGHT  = (14, 130, 112)   # wall right face (lighter — lit side)
C_WALL_EDGE   = ( 8,  80,  68)   # wall outline
C_DOOR_FRAME  = ( 5,  32,  64)   # door arch frame
C_DOOR_FILL   = (15,  50,  80)   # door interior
C_PLAYER      = (255, 220,  60)  # player diamond — golden yellow
C_NPC         = (255, 110,  90)  # NPC diamond — coral
C_CREATURE    = (180, 100, 255)  # creature — purple
C_ITEM        = (160, 230, 200)  # item on floor — mint
C_ITEM_EQUIP  = (255, 200, 120)  # equipped item slot colour in HUD
C_STAIR_UP    = (255, 255, 140)  # stair up
C_STAIR_DOWN  = (100, 140, 255)  # stair down
C_PORTAL      = (220, 100, 255)  # portal tile — vivid purple
C_FOG         = ( 18,  30,  50)  # tiles in fog of war
C_TEXT        = (210, 235, 230)  # main UI text
C_TEXT_DIM    = (110, 140, 135)  # dimmed text (timestamps, etc.)
C_INPUT_BG    = ( 15,  30,  55)  # text-input box background
C_INPUT_CARET = (29, 210, 180)   # caret / cursor colour
C_HUD_BG      = ( 8,  18,  35)   # right-hand HUD background
C_HUD_BORDER  = (20, 110,  95)   # HUD border line
C_MSG_GOOD    = ( 90, 230, 180)  # OK action message
C_MSG_BAD     = (255, 120,  90)  # rejected action message
C_MSG_NPC     = (180, 160, 255)  # NPC speech
C_MSG_AMBIENT = (150, 200, 160)  # ambient / world event
C_HP_FULL     = ( 80, 220, 130)  # health bar full
C_HP_LOW      = (230,  80,  80)  # health bar low
C_CARRY_OK    = ( 80, 200, 160)  # carry-weight bar ok
C_CARRY_HEAVY = (230, 180,  50)  # carry-weight bar heavy
C_CARRY_OVER  = (230,  80,  80)  # carry-weight bar over-encumbered

# ---------------------------------------------------------------------------
# Tile surface cache — built once at startup
# ---------------------------------------------------------------------------

_tile_cache: dict[str, pygame.Surface] = {}


def _iso_diamond(w: int, h: int) -> list[tuple[int, int]]:
    """Return the 4 corner points of an isometric diamond."""
    return [(w // 2, 0), (w, h // 2), (w // 2, h), (0, h // 2)]


def _make_floor(top: tuple, edge: tuple) -> pygame.Surface:
    surf = pygame.Surface((TILE_W, TILE_H), pygame.SRCALPHA)
    pts = _iso_diamond(TILE_W, TILE_H)
    pygame.draw.polygon(surf, top, pts)
    pygame.draw.polygon(surf, edge, pts, 1)
    return surf


def _make_wall(
    top: tuple, left: tuple, right: tuple, edge: tuple
) -> pygame.Surface:
    """Full 3-D block: top diamond + left wall face + right wall face."""
    h_total = TILE_H + WALL_H
    surf = pygame.Surface((TILE_W, h_total), pygame.SRCALPHA)

    # Top diamond
    top_pts = _iso_diamond(TILE_W, TILE_H)
    pygame.draw.polygon(surf, top, top_pts)
    pygame.draw.polygon(surf, edge, top_pts, 1)

    # Right face (south-east)
    rf = [
        (TILE_W,      TILE_H // 2),
        (TILE_W // 2, TILE_H),
        (TILE_W // 2, h_total),
        (TILE_W,      TILE_H // 2 + WALL_H),
    ]
    pygame.draw.polygon(surf, right, rf)
    pygame.draw.polygon(surf, edge, rf, 1)

    # Left face (south-west)
    lf = [
        (0,           TILE_H // 2),
        (TILE_W // 2, TILE_H),
        (TILE_W // 2, h_total),
        (0,           TILE_H // 2 + WALL_H),
    ]
    pygame.draw.polygon(surf, left, lf)
    pygame.draw.polygon(surf, edge, lf, 1)

    return surf


def _make_door_wall(
    top: tuple, left: tuple, right: tuple,
    edge: tuple, door_frame: tuple, door_fill: tuple,
) -> pygame.Surface:
    """Wall block with a door arch cut into the right (south-east) face."""
    surf = _make_wall(top, left, right, edge).copy()

    # Door arch sits in the right face
    dw = TILE_W // 4
    dh = WALL_H - 4
    dx = TILE_W - dw - 4
    dy = TILE_H // 2 + 4

    arch_pts = [
        (dx,        dy + dh),
        (dx,        dy + 6),
        (dx + dw // 2, dy),
        (dx + dw,   dy + 6),
        (dx + dw,   dy + dh),
    ]
    pygame.draw.polygon(surf, door_fill, arch_pts)
    pygame.draw.polygon(surf, door_frame, arch_pts, 2)

    return surf


def _make_floor_dark(top: tuple, edge: tuple) -> pygame.Surface:
    """Dim floor tile for out-of-FOV tiles."""
    # Blend top colour toward the fog colour
    def blend(c1: tuple, c2: tuple, t: float) -> tuple:
        return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))

    dim_top = blend(top, C_FOG, 0.72)
    return _make_floor(dim_top, edge)


def _make_entity_diamond(colour: tuple, size: int = 14) -> pygame.Surface:
    """Small diamond representing an entity on the map."""
    surf = pygame.Surface((size * 2, size), pygame.SRCALPHA)
    pts = _iso_diamond(size * 2, size)
    pygame.draw.polygon(surf, colour, pts)
    dark = tuple(max(0, c - 60) for c in colour)
    pygame.draw.polygon(surf, dark, pts, 1)
    return surf


def _make_item_dot(colour: tuple, size: int = 6) -> pygame.Surface:
    """Tiny dot representing an item on the floor."""
    surf = pygame.Surface((size * 2, size), pygame.SRCALPHA)
    pts = _iso_diamond(size * 2, size)
    pygame.draw.polygon(surf, colour, pts)
    return surf


def build_tile_cache() -> None:
    """Pre-render all tile surface variants into _tile_cache (mutates in place)."""
    _tile_cache.update({
        "floor":        _make_floor(C_FLOOR_TOP, C_FLOOR_EDGE),
        "floor_dim":    _make_floor_dark(C_FLOOR_TOP, C_FLOOR_EDGE),
        "floor_fog":    _make_floor(C_FOG, C_FOG),
        "wall":         _make_wall(C_WALL_TOP, C_WALL_LEFT, C_WALL_RIGHT, C_WALL_EDGE),
        "door":         _make_door_wall(C_WALL_TOP, C_WALL_LEFT, C_WALL_RIGHT,
                                        C_WALL_EDGE, C_DOOR_FRAME, C_DOOR_FILL),
        "stairs_up":    _make_floor(C_STAIR_UP, C_FLOOR_EDGE),
        "stairs_down":  _make_floor(C_STAIR_DOWN, C_FLOOR_EDGE),
        "portal":       _make_floor(C_PORTAL, (160, 60, 200)),
        "ent_player":   _make_entity_diamond(C_PLAYER, 16),
        "ent_npc":      _make_entity_diamond(C_NPC, 14),
        "ent_creature": _make_entity_diamond(C_CREATURE, 14),
        "item":         _make_item_dot(C_ITEM, 7),
    })


# ---------------------------------------------------------------------------
# Facing-direction arrow
# ---------------------------------------------------------------------------

# Isometric projection of each compass direction's world vector into screen
# (sx, sy) deltas.  world_to_iso(dx, dy) - world_to_iso(0, 0) gives:
#   sx_delta = (dx - dy) * TILE_W // 2
#   sy_delta = (dx + dy) * TILE_H // 2
def _facing_iso_delta(dx: int, dy: int) -> tuple[float, float]:
    sx = (dx - dy) * TILE_W / 2
    sy = (dx + dy) * TILE_H / 2
    mag = (sx ** 2 + sy ** 2) ** 0.5
    if mag == 0:
        return (0.0, 0.0)
    return sx / mag, sy / mag   # unit vector in screen space


def _draw_facing_arrow(
    surf: "pygame.Surface",
    ent: object,   # EntityState — avoid circular import, duck-typed
    cx: int,
    cy: int,
    length: int = 14,
) -> None:
    """Draw a short directional tick from entity centre, showing facing.

    ``cx, cy`` is the screen-space entity origin (centre of diamond base).
    """
    from .schemas import FACING_VECTORS
    facing_val = ent.facing.value  # type: ignore[attr-defined]
    dx, dy = FACING_VECTORS[facing_val]
    fx, fy = _facing_iso_delta(dx, dy)
    ex = int(cx + fx * length)
    ey = int(cy + fy * length)
    # Arrow colour: bright white for player, dimmer for others
    is_player_arrow = getattr(ent, "entity_id", "") and getattr(ent, "entity_id", "").startswith("ent_player")
    arrow_col = (255, 255, 255) if is_player_arrow else (220, 220, 180)
    pygame.draw.line(surf, arrow_col, (cx, cy), (ex, ey), 2)
    # Arrowhead: two short lines
    perp_x, perp_y = -fy, fx
    tip_x, tip_y = ex, ey
    pygame.draw.line(
        surf, arrow_col,
        (tip_x, tip_y),
        (int(tip_x - fx * 5 + perp_x * 4), int(tip_y - fy * 5 + perp_y * 4)),
        2,
    )
    pygame.draw.line(
        surf, arrow_col,
        (tip_x, tip_y),
        (int(tip_x - fx * 5 - perp_x * 4), int(tip_y - fy * 5 - perp_y * 4)),
        2,
    )


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def world_to_iso(gx: int, gy: int) -> tuple[int, int]:
    """Convert grid coordinates to isometric screen offset (before camera)."""
    sx = (gx - gy) * TILE_W // 2
    sy = (gx + gy) * TILE_H // 2
    return sx, sy


# ---------------------------------------------------------------------------
# IsoRenderer
# ---------------------------------------------------------------------------


class IsoRenderer:
    """
    Full-screen Pygame isometric renderer.

    Wraps a GameLoop instance.  On each frame it reads from world.spatial
    and renders the isometric view.  Player input (text) is routed to
    game_loop.step().
    """

    HUD_W      = 280      # right-side HUD panel width
    INPUT_H    = 44       # bottom input bar height
    MSG_LINES  = 10       # how many message lines to keep visible
    FONT_SIZE  = 14
    FONT_SMALL = 12

    def __init__(
        self,
        game_loop,
        window_w: int = 1100,
        window_h: int = 680,
    ) -> None:
        self.loop = game_loop
        self.world = game_loop.world
        self.player_id = game_loop.player_id

        self.win_w = window_w
        self.win_h = window_h
        self.view_w = window_w - self.HUD_W
        self.view_h = window_h - self.INPUT_H

        # Camera offset (screen pixels) — follows player
        self._cam_x: int = 0
        self._cam_y: int = 0

        # Message log — (text, colour) tuples, newest at end
        self._messages: deque[tuple[str, tuple]] = deque(maxlen=120)

        # Text input state
        self._input_text: str = ""
        self._input_active: bool = True
        self._caret_tick: int = 0

        # Pending NPC results to display from last step
        self._last_npc_msgs: list[str] = []

        # Viewport scroll (not camera — for the message panel)
        self._msg_scroll: int = 0

        # Fog of war set from last projection
        self._visible: set = set()

        grid = self.world.spatial
        self._voxel_mode = bool(grid.use_voxels)
        self.layers = None
        if self._voxel_mode or grid.depth > 1:
            from .voxel_layers import LayerNavigator
            self.layers = LayerNavigator(
                depth=max(1, grid.depth),
                slice_z=0 if grid.depth > 1 else None,
            )

        # Pygame font handles (set in run())
        self._font: Optional[pygame.font.Font] = None
        self._font_sm: Optional[pygame.font.Font] = None
        self._font_title: Optional[pygame.font.Font] = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption(
            f"Mark-1  ·  {self.world.name}  ·  Isometric View"
        )
        screen = pygame.display.set_mode(
            (self.win_w, self.win_h),
            pygame.RESIZABLE,
        )
        clock = pygame.time.Clock()

        build_tile_cache()

        # Fonts: try a monospace, fall back to default
        mono_candidates = [
            "JetBrainsMonoNL-Regular.ttf",
            "FiraCode-Regular.ttf",
            "DejaVuSansMono.ttf",
            "UbuntuMono-Regular.ttf",
            None,
        ]
        self._font = None
        for fc in mono_candidates:
            try:
                self._font = pygame.font.Font(fc, self.FONT_SIZE)
                break
            except Exception:
                continue
        if self._font is None:
            self._font = pygame.font.SysFont("monospace", self.FONT_SIZE)
        self._font_sm   = pygame.font.SysFont("monospace", self.FONT_SMALL)
        self._font_title = pygame.font.SysFont("monospace", self.FONT_SIZE + 2, bold=True)

        self._snap_camera()
        self._add_message(
            f"Loaded world: {self.world.name}. Type a command and press Enter.",
            C_MSG_AMBIENT,
        )
        if self._voxel_mode and self.layers is not None:
            self._add_message(
                f"Voxel world — {self.layers.label()}. PgUp/PgDn: Z layer.",
                C_TEXT_DIM,
            )
        self._add_message(
            "Commands: inventory / map / status / log / goals / quit"
            + (" / layer" if self.layers else ""),
            C_TEXT_DIM,
        )

        running = True
        while running:
            dt = clock.tick(60)
            self._caret_tick += dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.VIDEORESIZE:
                    self.win_w = event.w
                    self.win_h = event.h
                    self.view_w = self.win_w - self.HUD_W
                    self.view_h = self.win_h - self.INPUT_H

                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event)

                elif event.type == pygame.MOUSEWHEEL:
                    self._msg_scroll = max(0, self._msg_scroll - event.y * 2)

            self._render(screen)
            pygame.display.flip()

        pygame.quit()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _handle_key(self, event: pygame.event.Event) -> bool:
        if event.key == pygame.K_ESCAPE:
            return False  # quit

        if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
            text = self._input_text.strip()
            self._input_text = ""
            if text:
                self._process_command(text)
            return True

        if event.key == pygame.K_BACKSPACE:
            self._input_text = self._input_text[:-1]
            return True

        if event.key == pygame.K_PAGEUP:
            if self.layers is not None:
                self.layers.up()
                self._snap_camera()
                self._add_message(f"Layer: {self.layers.label()}", C_MSG_AMBIENT)
            else:
                self._msg_scroll = min(
                    len(self._messages) - self.MSG_LINES,
                    self._msg_scroll + 3,
                )
            return True

        if event.key == pygame.K_PAGEDOWN:
            if self.layers is not None:
                self.layers.down()
                self._snap_camera()
                self._add_message(f"Layer: {self.layers.label()}", C_MSG_AMBIENT)
            else:
                self._msg_scroll = max(0, self._msg_scroll - 3)
            return True

        if self.layers is not None and event.key == pygame.K_HOME:
            self.layers.ground()
            self._snap_camera()
            self._add_message(f"Layer: {self.layers.label()}", C_MSG_AMBIENT)
            return True

        if self.layers is not None and event.key == pygame.K_END:
            self.layers.top()
            self._snap_camera()
            self._add_message(f"Layer: {self.layers.label()}", C_MSG_AMBIENT)
            return True

        if self.layers is not None and event.key == pygame.K_0:
            self.layers.show_all()
            self._snap_camera()
            self._add_message(f"Layer: {self.layers.label()}", C_MSG_AMBIENT)
            return True

        # Printable chars
        if event.unicode and event.unicode.isprintable():
            self._input_text += event.unicode

        return True

    def _process_command(self, raw: str) -> None:
        lower = raw.lower().strip()

        # Meta commands (no game-loop step)
        if lower in ("quit", "exit", "q"):
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            return

        if lower in ("inventory", "inv", "i"):
            from .narrator import render_inventory
            player = self.world.spatial.entities.get(self.player_id)
            if player:
                text = render_inventory(player, self.world)
                for line in text.splitlines():
                    self._add_message(line, C_TEXT)
            return

        if lower in ("map", "m"):
            if self.layers is not None and self._voxel_mode:
                from .voxel_layers import render_layer_ascii
                z = self.layers.slice_z if self.layers.is_slice_mode else (
                    self.world.spatial.entities[self.player_id].position.z
                    if self.player_id in self.world.spatial.entities
                    else 0
                )
                for line in render_layer_ascii(
                    self.world.spatial, z, player_id=self.player_id
                ).splitlines():
                    self._add_message(line, C_TEXT_DIM)
            else:
                self._add_message("[map view — see the isometric world]", C_TEXT_DIM)
            return

        if self.layers is not None:
            from .voxel_layers import layer_help_lines
            layer_msg = self.layers.handle_repl_command(raw)
            if layer_msg is not None:
                self._snap_camera()
                self._add_message(f"Layer: {layer_msg}", C_MSG_AMBIENT)
                return

        if lower in ("status",):
            from .narrator import render_world_summary
            for line in render_world_summary(self.world).splitlines()[:12]:
                self._add_message(line, C_TEXT)
            return

        if lower == "log":
            from .narrator import render_event_log
            lines = render_event_log(self.world)
            for line in (lines[-20:] if lines else ["(no events yet)"]):
                self._add_message(line, C_TEXT_DIM)
            return

        if lower == "goals":
            goals = getattr(self.world, "goals", [])
            if not goals:
                self._add_message("No goals defined in this world.", C_TEXT_DIM)
            for g in goals:
                status = "[done]" if g.completed else "[pending]"
                self._add_message(f"  {status} {g.title}: {g.description}", C_TEXT)
            return

        if lower == "help":
            help_lines = [
                "── Commands ──────────────────────────",
                "  inventory / inv   show carry & slots",
                "  status            world summary",
                "  log               event history",
                "  goals             quest progress",
                "  quit              exit",
                "── Actions ───────────────────────────",
                "  Type anything natural, e.g.:",
                "  'say hello to the guard'",
                "  'equip iron sword'",
                "  'store knife in satchel'",
                "  'kiss elara'",
            ]
            for line in help_lines:
                self._add_message(line, C_TEXT_DIM)
            return

        # Game action — route to game loop
        self._add_message(f"» {raw}", C_TEXT)
        try:
            result = self.loop.step(raw)
            self._handle_step_result(result)
            self._snap_camera()
        except Exception as exc:
            self._add_message(f"[ERROR] {exc}", C_MSG_BAD)

    def _handle_step_result(self, result) -> None:
        from .narrator import render_event

        # Main player action narration
        if result.event:
            narration = (
                result.narration
                or render_event(result.event, self.world)
            )
            # Split NPC reply if embedded in narration (contains newline)
            lines = narration.splitlines()
            for i, line in enumerate(lines):
                col = C_MSG_NPC if (i > 0 and ":" in line) else C_MSG_GOOD
                self._add_message(f"  {line}", col)
        elif not result.validation.valid:
            detail = result.validation.rejection_detail or str(result.validation.rejection_reason)
            self._add_message(f"  ✗ {detail}", C_MSG_BAD)
        else:
            self._add_message(f"  {result.status}", C_TEXT_DIM)

        # Ambient events
        for amb in result.ambient_events:
            hint = amb.narrative_hint or "(ambient event)"
            self._add_message(f"  ~ {hint}", C_MSG_AMBIENT)

        # NPC reactions
        for npc_res in result.npc_results:
            if npc_res.event:
                npc_line = render_event(npc_res.event, self.world)
                for line in npc_line.splitlines():
                    col = C_MSG_NPC if ":" in line else C_TEXT_DIM
                    self._add_message(f"    {line}", col)

        # Goal completions
        for g in result.completed_goals:
            self._add_message(f"  ★ Quest complete: {g.title}!", C_MSG_GOOD)

        # Scroll to bottom after a step
        self._msg_scroll = 0

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _snap_camera(self) -> None:
        """Centre camera on the player."""
        from .voxel_draw import snap_camera_to

        grid = self.world.spatial
        z = self.layers.slice_z if self.layers is not None else None
        if self._voxel_mode:
            self._cam_x, self._cam_y = snap_camera_to(grid, self.player_id, z=z)
            return
        player = grid.entities.get(self.player_id)
        if player is None:
            return
        sx, sy = world_to_iso(player.position.x, player.position.y)
        self._cam_x = sx
        self._cam_y = sy

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, screen: pygame.Surface) -> None:
        screen.fill(C_BG)

        # Surfaces for compositing
        view_surf = pygame.Surface((self.view_w, self.view_h))
        view_surf.fill(C_BG)

        hud_surf  = pygame.Surface((self.HUD_W, self.win_h))
        hud_surf.fill(C_HUD_BG)

        self._render_world(view_surf)
        self._render_hud(hud_surf)
        self._render_input(screen)

        screen.blit(view_surf, (0, 0))
        screen.blit(hud_surf,  (self.view_w, 0))

        # Vertical HUD divider
        pygame.draw.line(
            screen, C_HUD_BORDER,
            (self.view_w, 0), (self.view_w, self.win_h - self.INPUT_H), 2
        )

    def _world_to_screen(self, gx: int, gy: int) -> tuple[int, int]:
        """World grid → pixel position inside the view surface."""
        sx, sy = world_to_iso(gx, gy)
        cx = self.view_w // 2 + sx - self._cam_x
        cy = self.view_h // 3 + sy - self._cam_y
        return cx, cy

    def _render_world(self, surf: pygame.Surface) -> None:
        grid = self.world.spatial

        if self._voxel_mode and self.layers is not None:
            from .voxel_draw import draw_layer_banner, draw_voxel_isometric
            draw_voxel_isometric(
                surf,
                grid,
                self.layers,
                cam_x=self._cam_x,
                cam_y=self._cam_y,
                player_id=self.player_id,
            )
            draw_layer_banner(surf, self.layers, self._font_sm, world_name=self.world.name)
            return

        # Build portal coord set for the active region (for tile highlighting)
        active_rid = self.world.active_region_id
        portal_coords: set[tuple[int, int]] = set()
        for portal in self.world.portals:
            if portal.from_region == active_rid:
                portal_coords.add((portal.from_coord.x, portal.from_coord.y))
            if portal.bidirectional and portal.to_region == active_rid:
                portal_coords.add((portal.to_coord.x, portal.to_coord.y))

        # Compute visibility from player projection (fast path)
        from .projection import project
        from .spatial import visible_from
        player = grid.entities.get(self.player_id)
        if player:
            self._visible = visible_from(
                grid, player.position,
                player.sight_range,
                facing=player.facing,
                fov_degrees=player.fov_degrees,
            )

        # Collect entity positions for quick lookup
        ent_at: dict[tuple[int, int], object] = {}
        for e in grid.entities.values():
            if e.alive:
                ent_at[(e.position.x, e.position.y)] = e

        # Collect ground items
        item_at: dict[tuple[int, int], bool] = {}
        for obj in grid.objects.values():
            if obj.position is not None:
                item_at[(obj.position.x, obj.position.y)] = True

        # Determine render range centred on player
        px = player.position.x if player else grid.width // 2
        py = player.position.y if player else grid.height // 2
        r = VIEW_RADIUS + 2

        from .schemas import Coord as _Coord, TerrainType

        # Sort by painter's order: (gx + gy) ascending
        cells = [
            (gx, gy)
            for gx in range(max(0, px - r), min(grid.width, px + r + 1))
            for gy in range(max(0, py - r), min(grid.height, py + r + 1))
        ]
        cells.sort(key=lambda c: c[0] + c[1])

        for gx, gy in cells:
            coord = _Coord(x=gx, y=gy)
            tile = grid.tile_at(coord)
            is_visible = coord in self._visible
            sx, sy = self._world_to_screen(gx, gy)

            is_portal = (gx, gy) in portal_coords

            # Pick tile surface
            if not tile.passable:
                if tile.terrain and tile.terrain.value == "door_closed":
                    tile_surf = _tile_cache["door"]
                elif tile.terrain and tile.terrain.value == "door_open":
                    tile_surf = _tile_cache["floor"]
                else:
                    tile_surf = _tile_cache["wall"]
                # Wall surfaces are taller — blit anchored to top
                blit_y = sy - WALL_H
            else:
                if is_portal:
                    tile_surf = _tile_cache["portal"]
                elif tile.terrain and tile.terrain.value == "stairs_up":
                    tile_surf = _tile_cache["stairs_up"]
                elif tile.terrain and tile.terrain.value == "stairs_down":
                    tile_surf = _tile_cache["stairs_down"]
                elif not is_visible:
                    tile_surf = _tile_cache["floor_dim"]
                else:
                    tile_surf = _tile_cache["floor"]
                blit_y = sy

            blit_x = sx - TILE_W // 2

            # Fog of war: skip completely unknown tiles
            if not is_visible and tile.passable:
                # Draw dim floor to show explored areas
                surf.blit(tile_surf, (blit_x, blit_y))
                continue

            surf.blit(tile_surf, (blit_x, blit_y))

            # Ground items (only in visible area)
            if is_visible and (gx, gy) in item_at:
                isf = _tile_cache["item"]
                surf.blit(
                    isf,
                    (sx - isf.get_width() // 2, sy - TILE_H // 4 - isf.get_height()),
                )

            # Entities
            if (gx, gy) in ent_at:
                ent = ent_at[(gx, gy)]
                if ent.entity_id == self.player_id:
                    esf = _tile_cache["ent_player"]
                elif ent.kind.value == "creature":
                    esf = _tile_cache["ent_creature"]
                else:
                    esf = _tile_cache["ent_npc"]

                ey = sy - TILE_H // 4 - esf.get_height()
                surf.blit(esf, (sx - esf.get_width() // 2, ey))

                # Facing arrow: small line pointing in the entity's facing direction.
                # Drawn from the entity's screen-centre outward in iso-projected space.
                _draw_facing_arrow(surf, ent, sx, sy - TILE_H // 4)

                # Name tag above entity (only for nearby visible entities)
                dist = abs(gx - px) + abs(gy - py)
                if is_visible and dist <= 6 and self._font_sm:
                    name_surf = self._font_sm.render(
                        ent.name, True, C_TEXT_DIM
                    )
                    nx = sx - name_surf.get_width() // 2
                    ny = ey - name_surf.get_height() - 2
                    surf.blit(name_surf, (nx, ny))

    def _render_hud(self, surf: pygame.Surface) -> None:
        pad = 10
        y = 8

        def txt(s: str, colour: tuple, font=None, x: int = pad) -> int:
            f = font or self._font
            if f is None:
                return y
            rendered = f.render(s, True, colour)
            surf.blit(rendered, (x, y))
            return rendered.get_height() + 2

        # Title + region
        world_name = self.world.name[:26]
        y += txt(f" {world_name}", C_INPUT_CARET, self._font_title)
        active_region = self.world.regions.get(self.world.active_region_id)
        region_label = active_region.name if active_region else self.world.active_region_id
        floor_lv = active_region.floor_level if active_region else 0
        floor_str = (f"floor {floor_lv:+d}" if floor_lv != 0 else "ground floor")
        y += txt(f" {region_label[:24]}  [{floor_str}]", C_TEXT_DIM)
        y += txt(f" tick {self.world.tick}  ·  {len(self.world.regions)} region(s)", C_TEXT_DIM)
        y += 6

        # Horizontal rule
        pygame.draw.line(surf, C_HUD_BORDER, (pad, y), (self.HUD_W - pad, y), 1)
        y += 8

        # Player vitals
        player = self.world.spatial.entities.get(self.player_id)
        if player:
            y += txt("PLAYER", C_TEXT_DIM)
            y += txt(f"  {player.name}", C_TEXT, self._font_title)

            # HP bar
            hp_frac = player.health / max(1, player.max_health)
            y += 2
            self._bar(surf, pad, y, self.HUD_W - pad * 2, 10,
                      hp_frac, C_HP_FULL, C_HP_LOW, f"HP {player.health}/{player.max_health}")
            y += 16

            # Carry weight bar
            from .compiler import current_carry_weight
            carried = current_carry_weight(player, self.world.spatial)
            carry_frac = carried / max(0.01, player.carry_capacity)
            col = C_CARRY_OVER if carry_frac > 1.0 else (
                C_CARRY_HEAVY if carry_frac > 0.75 else C_CARRY_OK
            )
            self._bar(surf, pad, y, self.HUD_W - pad * 2, 10,
                      min(1.0, carry_frac), col, col,
                      f"WGT {carried:.1f}/{player.carry_capacity:.0f}kg")
            y += 16

            # Facing compass + FOV
            fov_label = f"{int(player.fov_degrees)}° fov"
            y += txt(
                f"  facing {player.facing.value:<9}  {fov_label}",
                C_INPUT_CARET,
            )

            # Status conditions
            if player.conditions:
                cond_str = "  " + ", ".join(player.conditions.keys())
                y += txt(cond_str[:32], C_MSG_BAD)
            else:
                y += txt(f"  {player.emotional_state.value} · {player.alertness.value}", C_TEXT_DIM)

            y += 6
            pygame.draw.line(surf, C_HUD_BORDER, (pad, y), (self.HUD_W - pad, y), 1)
            y += 8

            # Equipped slots
            y += txt("EQUIPPED", C_TEXT_DIM)
            if player.equipped_slots:
                for slot_name, oid in sorted(player.equipped_slots.items()):
                    obj = self.world.spatial.objects.get(oid)
                    obj_n = obj.name if obj else str(oid)
                    slot_label = slot_name.replace("_", " ")[:9]
                    line = f"  [{slot_label:<9}] {obj_n[:16]}"
                    y += txt(line, C_ITEM_EQUIP)
            else:
                y += txt("  (nothing equipped)", C_TEXT_DIM)

            y += 6
            pygame.draw.line(surf, C_HUD_BORDER, (pad, y), (self.HUD_W - pad, y), 1)
            y += 8

        # Visible entities
        y += txt("NEARBY", C_TEXT_DIM)
        from .schemas import Coord as _Coord
        if player:
            px, py2 = player.position.x, player.position.y
            nearby = [
                e for e in self.world.spatial.entities.values()
                if e.alive and e.entity_id != self.player_id
                and abs(e.position.x - px) + abs(e.position.y - py2) <= 8
                and _Coord(x=e.position.x, y=e.position.y) in self._visible
            ]
            nearby.sort(key=lambda e: abs(e.position.x - px) + abs(e.position.y - py2))
            for e in nearby[:6]:
                dist = abs(e.position.x - px) + abs(e.position.y - py2)
                col = C_NPC if e.kind.value == "npc" else C_CREATURE
                y += txt(f"  {e.name[:18]}  d={dist}", col)
                y += txt(f"    {e.emotional_state.value} · {e.alertness.value}", C_TEXT_DIM)
        if y < 300:
            y = 300

        pygame.draw.line(surf, C_HUD_BORDER, (pad, y), (self.HUD_W - pad, y), 1)
        y += 8

        # Message log
        y += txt("LOG", C_TEXT_DIM)
        msgs = list(self._messages)
        start = max(0, len(msgs) - self.MSG_LINES - self._msg_scroll)
        visible_msgs = msgs[start: start + self.MSG_LINES]
        for msg_text, msg_col in visible_msgs:
            # Word-wrap long messages
            wrapped = textwrap.wrap(msg_text, width=28) or [msg_text]
            for line in wrapped:
                if self._font_sm and y + 16 < surf.get_height() - 4:
                    rendered = self._font_sm.render(line, True, msg_col)
                    surf.blit(rendered, (pad, y))
                    y += rendered.get_height() + 1

        # Scroll indicator
        if self._msg_scroll > 0:
            si = self._font_sm.render(
                f"▲ {self._msg_scroll} more above", True, C_TEXT_DIM
            ) if self._font_sm else None
            if si:
                surf.blit(si, (pad, surf.get_height() - 20))

    def _bar(
        self, surf: pygame.Surface,
        x: int, y: int, w: int, h: int,
        frac: float,
        full_col: tuple, low_col: tuple,
        label: str = "",
    ) -> None:
        """Draw a horizontal progress bar with label."""
        pygame.draw.rect(surf, C_BG, (x, y, w, h), border_radius=2)
        fill_w = max(0, min(w, int(w * frac)))
        col = full_col if frac > 0.4 else low_col
        if fill_w > 0:
            pygame.draw.rect(surf, col, (x, y, fill_w, h), border_radius=2)
        pygame.draw.rect(surf, C_HUD_BORDER, (x, y, w, h), 1, border_radius=2)
        if label and self._font_sm:
            ls = self._font_sm.render(label, True, C_TEXT)
            surf.blit(ls, (x + 3, y + (h - ls.get_height()) // 2))

    def _render_input(self, screen: pygame.Surface) -> None:
        iy = self.win_h - self.INPUT_H
        pygame.draw.rect(
            screen, C_INPUT_BG,
            (0, iy, self.win_w, self.INPUT_H)
        )
        pygame.draw.line(screen, C_HUD_BORDER, (0, iy), (self.win_w, iy), 1)

        if self._font:
            prompt = self._font.render("» ", True, C_INPUT_CARET)
            screen.blit(prompt, (10, iy + (self.INPUT_H - prompt.get_height()) // 2))
            text_surf = self._font.render(self._input_text, True, C_TEXT)
            screen.blit(text_surf, (10 + prompt.get_width(), iy + (self.INPUT_H - text_surf.get_height()) // 2))

            # Blinking caret
            if (self._caret_tick // 500) % 2 == 0:
                caret_x = 10 + prompt.get_width() + text_surf.get_width() + 2
                caret_y = iy + 8
                pygame.draw.line(
                    screen, C_INPUT_CARET,
                    (caret_x, caret_y),
                    (caret_x, caret_y + self.INPUT_H - 16), 2
                )

        # Help hint right-aligned
        if self._font_sm:
            hint = self._font_sm.render(
                "Enter=send  PgUp/Dn=scroll log  ESC=quit", True, C_TEXT_DIM
            )
            screen.blit(
                hint,
                (self.win_w - hint.get_width() - 10,
                 iy + (self.INPUT_H - hint.get_height()) // 2)
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_message(self, text: str, colour: tuple) -> None:
        self._messages.append((text, colour))
        self._msg_scroll = 0  # auto-scroll to latest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Isometric Renderer — Mark-1 Sim")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--real-lm",  action="store_true")
    group.add_argument("--ollama",   action="store_true")
    group.add_argument("--torch",    action="store_true")
    group.add_argument("--outlines", metavar="GGUF_PATH")
    parser.add_argument("--model",       default=None)
    parser.add_argument(
        "--torch-path",
        dest="torch_path",
        default=None,
        metavar="DIR",
        help="Local HF weights directory (use with --torch)",
    )
    parser.add_argument("--ollama-url",  default="http://localhost:11434")
    parser.add_argument("--world",       default="default", metavar="WORLD")
    parser.add_argument("--width",       type=int, default=1100)
    parser.add_argument("--height",      type=int, default=680)
    return parser.parse_args()


if __name__ == "__main__":
    import sys
    import warnings

    warnings.warn(
        "iso_renderer CLI merged into src.sim.play — use: "
        "python3 -m src.sim.play --pygame --view iso",
        DeprecationWarning,
        stacklevel=1,
    )
    from .play import main as play_main

    argv = ["--pygame", "--view", "iso", *sys.argv[1:]]
    play_main(argv)
