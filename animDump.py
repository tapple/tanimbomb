#!/usr/bin/env python

import argparse
import fnmatch
import struct
import copy
import sys

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
        print '%s: P%d %dR %dL %dC' % (name, self.priority,
                rotJointCount, locJointCount, len(self.constraints))

class AnimTransform(object):
    def __init__(self):
        pass

    def __call__(self, anim):
        pass

class SetAnimProperty(AnimTransform):
    def __init__(self, value, key):
        self.value = value
        self.key = key

    def __call__(self, anim):
        setattr(anim, self.key, self.value)


class TransformJointsMatching(AnimTransform):
    def __init__(self, *globs, **kwargs):
        self.dropGlobs = list()
        self.keepGlobs = list()
        for glob in globs:
            if glob.startswith('+'):
                self.keepGlobs.append(glob[1:])
            else:
                self.dropGlobs.append(glob);
        self.jointTransform = kwargs['jointTransform']

    def __repr__(self):
        return "TransformJointsMatching(%r, %r, %r)" % (self.dropGlobs,
                self.keepGlobs, self.jointTransform)

    def _shouldDropJointNamed(self, jointName):
        for dropGlob in self.dropGlobs:
            if (fnmatch.fnmatch(jointName, dropGlob)):
                for keepGlob in self.keepGlobs:
                    if (fnmatch.fnmatch(jointName, keepGlob)):
                        return False
                return True
        return False

    def __call__(self, anim):
        # iterate over a copy since joints may be removed
        for joint in anim.joints[:]:
            if self._shouldDropJointNamed(joint.name):
                self.jointTransform(anim, joint)

def dropLocationKeyframes(anim, joint):
    joint.locKeys = ""

def dropRotationKeyframes(anim, joint):
    joint.rotKeys = ""

def dropPriority(anim, joint):
    joint.priority = anim.priority

def dropJoint(anim, joint):
    anim.joints.remove(joint)

class DropEmptyJoints(AnimTransform):
    def __call__(self, anim):
        anim.joints = [joint for joint in anim.joints if
                (joint.rotKeyCount > 0 or joint.locKeyCount > 0)]



class AddConstraint(AnimTransform):
    def __init__(self, sourceVolume, targetVolume, chainLength):
        self.constraint = JointConstraintSharedData()
        self.constraint.chainLength = int(chainLength)
        self.constraint.type = 0 # 0: point, 1: plane
        self.constraint.sourceVolume = sourceVolume

        self.constraint.sourceOffsetX = 0
        self.constraint.sourceOffsetY = 0
        self.constraint.sourceOffsetZ = 0

        self.constraint.targetVolume = targetVolume
        self.constraint.targetOffsetX = 0
        self.constraint.targetOffsetY = 0
        self.constraint.targetOffsetZ = 0

        self.constraint.targetDirX = 0
        self.constraint.targetDirY = 0
        self.constraint.targetDirZ = 0

        self.constraint.easeInStart = -1
        self.constraint.easeInStop = 0
        self.constraint.easeOutStart = 10
        self.constraint.easeOutStop = 10

    def __call__(self, anim):
        anim.constraints.append(self.constraint)

class DropConstraints(AnimTransform):
    def __call__(self, anim):
        anim.constraints = list()




def _ensure_value(namespace, name, value):
    if getattr(namespace, name, None) is None:
        setattr(namespace, name, value)
    return getattr(namespace, name)

class AppendObjectAction(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest,
                 func,
                 nargs=0,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None,
                 **kwargs):
        self.func = func
        self.kwargs = kwargs
        super(AppendObjectAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        #print '%r %r %r' % (namespace, values, self.kwargs)
        item = self.func(*values, **self.kwargs)
        items = copy.copy(_ensure_value(namespace, self.dest, []))
        items.append(item)
        setattr(namespace, self.dest, items)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description='Manipulate Secondlife .anim files')
    parser.add_argument('files', type=argparse.FileType('r'), nargs='+',
                        help='anim files to dump or process')
    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('--outputfiles', '-o', type=argparse.FileType('w'),
            nargs='*')

    parser.add_argument('--pri', action=AppendObjectAction,
            dest='actions', func=SetAnimProperty, nargs=1,
            key='priority', type=int)
    parser.add_argument('--ease-in', action=AppendObjectAction,
            dest='actions', func=SetAnimProperty, nargs=1,
            key='easeIn', type=float)
    parser.add_argument('--ease-out', action=AppendObjectAction,
            dest='actions', func=SetAnimProperty, nargs=1,
            key='easeOut', type=float)

    parser.add_argument('--drop-loc', action=AppendObjectAction,
            dest='actions', func=TransformJointsMatching, nargs='*',
            jointTransform=dropLocationKeyframes)
    parser.add_argument('--drop-rot', action=AppendObjectAction,
            dest='actions', func=TransformJointsMatching, nargs='*',
            jointTransform=dropRotationKeyframes)
    parser.add_argument('--drop-pri', action=AppendObjectAction,
            dest='actions', func=TransformJointsMatching, nargs='*',
            jointTransform=dropPriority)
    parser.add_argument('--drop-joint', action=AppendObjectAction,
            dest='actions', func=TransformJointsMatching, nargs='*',
            jointTransform=dropJoint)
    parser.add_argument('--drop-empty-joints', action=AppendObjectAction,
            dest='actions', func=DropEmptyJoints)

    parser.add_argument('--add-constraint', action=AppendObjectAction,
            dest='actions', func=AddConstraint, nargs=3)
    parser.add_argument('--drop-constraints', action=AppendObjectAction,
            dest='actions', func=DropConstraints)

    args = parser.parse_args()
    _ensure_value(args, 'actions', [])

    if (args.verbose >= 2):
        print args

    if args.outputfiles is None:
        # summarize all files
        for file in args.files:
            anim = KeyframeMotion()
            anim.deserialize(file)
            anim.summarize(file.name)
            if (args.verbose > 0):
                anim.dump()

    else:
        # convert files
        if (len(args.files) != len(args.outputfiles)):
            print "different number of input and output files"
            sys.exit();
        for inputFile,outputFile in zip(args.files, args.outputfiles):
            anim = KeyframeMotion()
            anim.deserialize(inputFile)
            for action in args.actions:
                action(anim)
            anim.serialize(outputFile)
            anim.summarize(outputFile.name)
            if (args.verbose > 0):
                anim.dump()








