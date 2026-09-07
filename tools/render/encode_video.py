"""Blender headless: PNG sequence -> mp4 (Blender ships its own ffmpeg).
Usage: blender --background --python encode_video.py -- <png_dir> <out_mp4> [fps]
"""
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
PNGS, OUT = argv[0], argv[1]
FPS = int(argv[2]) if len(argv) > 2 else 12

files = sorted(f for f in os.listdir(PNGS) if f.endswith(".png"))
sc = bpy.context.scene
sc.render.resolution_x, sc.render.resolution_y = 640, 360
sc.render.fps = FPS
sc.frame_start, sc.frame_end = 1, len(files)
sc.render.image_settings.file_format = "FFMPEG"
sc.render.ffmpeg.format = "MPEG4"
sc.render.ffmpeg.codec = "H264"
sc.render.ffmpeg.constant_rate_factor = "MEDIUM"
sc.render.filepath = OUT

ed = sc.sequence_editor_create()
strip = ed.sequences.new_image("clip", os.path.join(PNGS, files[0]), 1, 1)
for f in files[1:]:
    strip.elements.append(f)
if len(argv) > 3 and argv[3]:                    # optional voiced track: a WAV
    ed.sequences.new_sound("voices", argv[3], 2, 1)   # already mixed & aligned
    sc.render.ffmpeg.audio_codec = "AAC"
    snd_frames = int(len(files))                 # keep video length; audio clips
bpy.ops.render.render(animation=True)
print("wrote", OUT)
