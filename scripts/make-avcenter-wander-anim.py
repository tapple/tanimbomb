from math import sin, cos, pi
import animDump
import numpy as np

def enumerate_waypoints():
    RADIUS = 1.5/2
    PHASE = 0
    POINTS = 6
    RINGS = 2
    for r in range(1, RINGS+1):
        for i in range(POINTS):
            angle = 2 * pi / POINTS * (i + PHASE)
            offset = np.array([0, cos(angle), sin(angle), 0]) * r * RADIUS
            angle2 = 2 * pi / POINTS * (i+1 + PHASE)
            offset2 = np.array([0, cos(angle2), sin(angle2), 0]) * r * RADIUS
            for s in range(r):
                interp = (offset2 * s + offset * (r - s)) / r
                yield f"{r}{6*i+3*s:02d}", interp

def enumerate_headings():
    PHASE = 0
    POINTS = 12
    for i in range(POINTS):
        t = i / POINTS
        x = sin(pi*t)  # pi not tau because quaternion double cover
        w = cos(pi*t)
        if w < 0:
            x = -x
        angle = pi / POINTS * (i + PHASE)
        offset = np.array([0, 0, 0, x])
        yield f"{3*i:02d}", offset

def make_avatar_center_pos_anim(name, pos):
    anim = animDump.KeyframeMotion(priority=2, easeIn=2.0, easeOut=0.1, loop_start=0, loop_end=0, duration=0)
    anim.ensure_joint('Avatar Center', locKeysF=[pos])
    anim.serialize_filename(f'AvCP{name}.anim')

def make_avatar_center_pos_anims():
    for name, pos in enumerate_waypoints():
        make_avatar_center_pos_anim(name, pos)
    make_avatar_center_pos_anim("000", [0,0,0,0])


def make_avatar_center_rot_anim(name, rot):
    anim = animDump.KeyframeMotion(priority=2, easeIn=0.5, easeOut=0.1, loop_start=0, loop_end=0, duration=0)
    anim.ensure_joint('Avatar Center', rotKeysF=[rot])
    anim.serialize_filename(f'AvCR{name}.anim')

def make_avatar_center_rot_anims():
    for name, rot in enumerate_headings():
        make_avatar_center_rot_anim(name, rot)

# for name, pos in enumerate_waypoints():
#     print(f"{name}: {pos}")
# make_avatar_center_pos_anims()

# for name, pos in enumerate_headings():
#     print(f"{name}: {pos}")
make_avatar_center_rot_anims()
