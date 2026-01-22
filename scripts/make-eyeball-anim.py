#! /bin/env python

import animDump
import numpy
from numpy.linalg import norm
from quaternion import from_rotation_vector

YL = numpy.array([0.1, -1, 0])
ZL = numpy.array([0.0,  0, 1])
YL /= norm(YL)
ZL /= norm(ZL)
YR = YL * [-1,  1, -1]
ZR = ZL * [ 1, -1,  1]

dx = numpy.deg2rad(5)
dy = numpy.deg2rad(5)

for x in range(-2, 3):
    for y in range(-2, 3):
        anim = animDump.KeyframeMotion(priority=6, easeIn=0.1, easeOut=0.1, loop_start=0, loop_end=0, duration=0.1)
        anim.ensure_joint('mFaceForeheadLeft' , rotKeysQ=[(0, from_rotation_vector(YL*dx*x) * from_rotation_vector(ZL*dy*y))])
        anim.ensure_joint('mFaceForeheadRight', rotKeysQ=[(0, from_rotation_vector(YR*dx*x) * from_rotation_vector(ZR*dy*y))])
        anim.serialize_filename(f'dog_eye_{x+2}_{y+2}.anim')
