"""Z-layer navigation and ASCII slice rendering."""

from src.sim.schemas import Coord, SpatialGrid, VoxelCell, VoxelMaterial
from src.sim.voxel.build import build_tavern_volume
from src.sim.voxel_layers import LayerNavigator, render_layer_ascii


def test_layer_navigator_commands():
    nav = LayerNavigator(depth=8, slice_z=2)
    assert nav.handle_repl_command("layer") == "layer z=2 / 7"
    nav.handle_repl_command("layer up")
    assert nav.slice_z == 3
    nav.handle_repl_command("layer down")
    assert nav.slice_z == 2
    nav.show_all()
    assert nav.slice_z is None


def test_render_layer_ascii_has_bounds():
    g = SpatialGrid(width=6, height=4, depth=3, use_voxels=True)
    build_tavern_volume(g)
    text = render_layer_ascii(g, 1)
    assert "Layer z=1" in text
    assert "#" in text or "." in text
