import animDump
import numpy

def enumerate_waypoints():
    RADIUS = 1.5/2
    PHASE = 0
    POINTS = 6
    RINGS = 2
    for r in range(1, RINGS+1):
        for i in range(POINTS):
            angle = 2 * numpy.pi / POINTS * (i + PHASE)
            offset = numpy.array([0, numpy.cos(angle), numpy.sin(angle), 0]) * r * RADIUS
            angle2 = 2 * numpy.pi / POINTS * (i+1 + PHASE)
            offset2 = numpy.array([0, numpy.cos(angle2), numpy.sin(angle2), 0]) * r * RADIUS
            for s in range(r):
                interp = (offset2 * s + offset * (r - s)) / r
                yield f"{r}{6*i+3*s:02d}", interp

def make_avatar_center_pos_anim(name, pos):
    anim = animDump.KeyframeMotion(priority=2, easeIn=2.0, easeOut=0.1, loop_start=0, loop_end=0, duration=0)
    anim.ensure_joint('Avatar Center', locKeysF=[pos])
    anim.serialize_filename(f'AvCP{name}.anim')

def make_avatar_center_pos_anims():
    for name, pos in enumerate_waypoints():
        make_avatar_center_pos_anim(name, pos)
    make_avatar_center_pos_anim("000", [0,0,0,0])

# for name, pos in enumerate_waypoints():
#     print(f"{name}: {pos}")
make_avatar_center_pos_anims()
