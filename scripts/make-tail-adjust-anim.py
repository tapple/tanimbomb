import animDump
#import numpy

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
for z in numpy.arange(-0.42, -0.51, -0.02):
    for y in numpy.arange(-0.20, -0.01, 0.02):
        make_tail_adjust(z, y)
    for y in numpy.arange(0.00, 0.11, 0.02):  # split so that 0 has + sign
        make_tail_adjust(z, y)
"""

for scale in range(1, 81):
    make_tail_length(scale*0.05)
