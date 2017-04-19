#!/usr/bin/env python

import sys
import struct

# http://stackoverflow.com/questions/442188/readint-readbyte-readstring-etc-in-python
#
class BinaryStream:
    def __init__(self, base_stream):
        self.base_stream = base_stream

    def readBytes(self, length):
        return self.base_stream.read(length)

    def writeBytes(self, value):
        self.base_stream.write(value)

    def unpack(self, fmt):
        return struct.unpack(fmt, self.readBytes(struct.calcsize(fmt)))

    def pack(self, fmt, *data):
        return self.writeBytes(struct.pack(fmt, *data))

    # http://stackoverflow.com/questions/32774910/clean-way-to-read-a-null-terminated-c-style-string-from-a-file
    def readCString(self):
        buf = bytearray()
        while True:
            b = self.base_stream.read(1)
            if b is None or b == '\0':
                return str(buf)
            else:
                buf.append(b)

    def writeCString(self, string):
        self.writeBytes(string)
        self.writeBytes("\0")

class JointMotion(object):
    KEY_SIZE = 8

    @property
    def rotKeyCount(self):
        return len(self.rotKeys) / JointMotion.KEY_SIZE

    @property
    def locKeyCount(self):
        return len(self.locKeys) / JointMotion.KEY_SIZE


class JointConstraintSharedData(object):
    pass

class KeyframeMotion(object):
    def deserialize(self, file):
        stream = BinaryStream(file)
        (self.version, self.subVersion, self.priority, self.duration) = stream.unpack("HHif")
        self.emote = stream.readCString()
        (self.loopIn, self.loopOut, self.loop, self.easeIn, self.easeOut, self.handPosture, jointCount) = stream.unpack("ffiffii")
        self.joints = list()
        for jointNum in range(jointCount):
            joint = JointMotion()
            self.joints.append(joint)
            joint.name = stream.readCString()
            (joint.priority, rotKeyCount) = stream.unpack("ii")
            joint.rotKeys = stream.readBytes(rotKeyCount * JointMotion.KEY_SIZE)
            (locKeyCount,) = stream.unpack("i")
            joint.locKeys = stream.readBytes(locKeyCount * JointMotion.KEY_SIZE)
        (constraintCount,) = stream.unpack("i")
        self.constraints = list()
        for constraintNum in range(constraintCount):
            constraint = JointConstraintSharedData()
            self.constraints.append(constraint)
            (constraint.chainLength, constraint.type) = stream.unpack("BB")
            (constraint.sourceVolume, constraint.sourceOffsetX, constraint.sourceOffsetY, constraint.sourceOffsetZ,
                constraint.targetVolume, constraint.targetOffsetX, constraint.targetOffsetY, constraint.targetOffsetZ,
                constraint.targetDirX, constraint.targetDirY, constraint.targetDirZ,
                constraint.easeInStart, constraint.easeInStop, constraint.easeOutStart, constraint.easeOutStop) = stream.unpack("16s3f16s3f3f4f")

    def serialize(self, file):
        stream = BinaryStream(file)
        stream.pack("HHif", self.version, self.subVersion, self.priority, self.duration)
        stream.writeCString(self.emote)
        stream.pack("ffiffii", self.loopIn, self.loopOut, self.loop, self.easeIn, self.easeOut, self.handPosture, len(self.joints))
        for joint in self.joints:
            stream.writeCString(joint.name)
            stream.pack("ii", joint.priority, joint.rotKeyCount)
            stream.writeBytes(joint.rotKeys)
            stream.pack("i", joint.locKeyCount)
            stream.writeBytes(joint.locKeys)
        stream.pack("i", len(self.constraints))
        for constraint in self.constraints:
            stream.pack("BB", constraint.chainLength, constraint.type)
            stream.pack("16s3f16s3f3f4f",
                constraint.sourceVolume, constraint.sourceOffsetX, constraint.sourceOffsetY, constraint.sourceOffsetZ,
                constraint.targetVolume, constraint.targetOffsetX, constraint.targetOffsetY, constraint.targetOffsetZ,
                constraint.targetDirX, constraint.targetDirY, constraint.targetDirZ,
                constraint.easeInStart, constraint.easeInStop, constraint.easeOutStart, constraint.easeOutStop)

    def dump(self):
        print "version: %d.%d" % (self.version, self.subVersion)
        print "priority: %d" % (self.priority,)
        print "duration: %f" % (self.duration,)
        print 'emote: "%s"' % (self.emote,)
        print 'loop: %d (%f - %f)' % (self.loop, self.loopIn, self.loopOut)
        print 'ease: %f - %f' % (self.easeIn, self.easeOut)
        print 'joints: %d' % (len(self.joints),)
        for joint in self.joints:
            print '\tP%d %dR %dL: %s' % (joint.priority, joint.rotKeyCount, joint.locKeyCount, joint.name)
        print 'constraints: %d' % (len(self.constraints),)
        for constraint in self.constraints:
            print "\tchain: %d type: %d\n\t\t%s + <%f, %f, %f> ->\n\t\t%s + <%f, %f, %f> at <%f, %f, %f>\n\t\tease: %f, %f - %f, %f" % (constraint.chainLength, constraint.type,
                constraint.sourceVolume, constraint.sourceOffsetX, constraint.sourceOffsetY, constraint.sourceOffsetZ,
                constraint.targetVolume, constraint.targetOffsetX, constraint.targetOffsetY, constraint.targetOffsetZ,
                constraint.targetDirX, constraint.targetDirY, constraint.targetDirZ,
                constraint.easeInStart, constraint.easeInStop, constraint.easeOutStart, constraint.easeOutStop)

    def summarize(self, name):
        rotJointCount = 0
        locJointCount = 0
        for joint in self.joints:
            if (joint.rotKeyCount > 0): rotJointCount += 1
            if (joint.locKeyCount > 0): locJointCount += 1
        print '%s: %dR %dL' % (name, rotJointCount, locJointCount)

"""
print sys.argv
file = open(sys.argv[1], 'rb')
anim = KeyframeMotion()
anim.deserialize(file)
anim.dump()
"""

for arg in sys.argv[1:]:
    file = open(arg, 'rb')
    anim = KeyframeMotion()
    anim.deserialize(file)
    anim.summarize(arg)




