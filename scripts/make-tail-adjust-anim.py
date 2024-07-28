import animDump
import numpy

def make_avatar_center_adjust(x, y, z):
    anim = animDump.KeyframeMotion(priority=6, easeIn=0.1, easeOut=0.1, loop_start=0, loop_end=0, duration=0.1)
    anim.new_joint('Avatar Center', locKeysF=[[0.0, x*2.5, y*2.5, z*2.5], [1.0, 0.0, 0.0, 0.0]])
    # anim.serialize_filename('Avatar_Center_%d_%d_%d.anim' % (x, y, z))
    anim.serialize_filename('Avatar Center %.1f %.1f %.1f.anim' % (x*2.5, y*2.5, z*2.5))

def make_avatar_center_bg():
    anim = animDump.KeyframeMotion(priority=0, easeIn=0.0, easeOut=0.0)
    anim.new_joint('Avatar Center', locKeysF=[[0.0, 0.0, 0.0, 0.0]], rotKeysF=[[0.0, 0.0, 0.0, 0.0]])
    anim.serialize_filename('Avatar Center zero bg.anim')

def make_tail_adjust(z, y):
    anim = animDump.KeyframeMotion(priority=6, easeIn=0.0, easeOut=0.0)
    anim.new_joint('mTail1', locKeysF=[[0.0, -y-0.116, 0.0, z+0.047]])
    anim.serialize_filename('tail_adjust_z%+.2f_y%+.2f.anim' % (z, y))


def make_tail_length(scale):
    anim = animDump.KeyframeMotion(priority=6, easeIn=0.0, easeOut=0.0)
    anim.new_joint('mTail2', locKeysF=[[0.0, -0.197*scale, 0.0, 0.0]])
    anim.new_joint('mTail3', locKeysF=[[0.0, -0.168*scale, 0.0, 0.0]])
    anim.new_joint('mTail4', locKeysF=[[0.0, -0.142*scale, 0.0, 0.0]])
    anim.new_joint('mTail5', locKeysF=[[0.0, -0.112*scale, 0.0, 0.0]])
    anim.new_joint('mTail6', locKeysF=[[0.0, -0.094*scale, 0.0, 0.0]])
    anim.serialize_filename('tail_length_x%+.2f.anim' % (scale))

"""
for z in numpy.arange(-0.50, -0.01, 0.02):
    # for y in numpy.arange(-0.20, -0.01, 0.02):
    #     make_tail_adjust(z, y)
    for y in numpy.arange(0.92, 1.21, 0.02):  # split so that 0 has + sign
        make_tail_adjust(z, y)
for z in numpy.arange(0.0, 0.31, 0.02):
    # for y in numpy.arange(-0.20, -0.01, 0.02):
    #     make_tail_adjust(z, y)
    for y in numpy.arange(0.92, 1.21, 0.02):  # split so that 0 has + sign
        make_tail_adjust(z, y)
"""

"""
for scale in range(1, 81):
    make_tail_length(scale*0.05)
"""

for x in range(-2, 3):
    for y in range(-2, 3):
        for z in range(-2, 3):
            make_avatar_center_adjust(x, y, z)
make_avatar_center_bg()
