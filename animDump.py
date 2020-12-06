#!/usr/bin/env python3

import argparse
import fnmatch
import struct
import copy
import sys
import numpy as np


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


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
            if b is None or b == b'\0':
                return buf
            else:
                buf.extend(b)

    def writeCString(self, string):
        self.writeBytes(string)
        self.writeBytes(b"\0")


class JointMotion(object):
    U16 = np.dtype('<H')
    U16MAX = 65535
    KEY_SIZE = 4
    LOC_MAX = 5

    def __init__(self, name='', priority=0, *,
            rotKeys=None, locKeys=None, rotKeysF=None, locKeysF=None):
        self.name = name
        self.priority = priority
        self.rotKeys = rotKeys or []
        self.locKeys = locKeys or []
        if rotKeysF:
            self.rotKeysF = rotKeysF
        if locKeysF:
            self.locKeysF = locKeysF

    @property
    def rotKeys(self):
        return self._rotKeys

    @rotKeys.setter
    def rotKeys(self, value):
        self._rotKeys = np.array(np.clip(value, 0, self.U16MAX),
            self.U16).reshape((-1, self.KEY_SIZE))

    @property
    def locKeys(self):
        return self._locKeys

    @locKeys.setter
    def locKeys(self, value):
        self._locKeys = np.array(np.clip(value, 0, self.U16MAX),
            self.U16).reshape((-1, self.KEY_SIZE))

    @property
    def rotKeysF(self):
        return self.keys_int_to_float(self.rotKeys)

    @rotKeysF.setter
    def rotKeysF(self, value):
        self.rotKeys = self.keys_float_to_int(value)

    @property
    def locKeysF(self):
        return self.keys_int_to_float(self.locKeys, self.LOC_MAX)

    @locKeysF.setter
    def locKeysF(self, value):
        self.locKeys = self.keys_float_to_int(value, self.LOC_MAX)

    @classmethod
    def keys_int_to_float(cls, keys, scale=1.0, dur=1.0):
        m = array([dur, 2*scale, 2*scale, 2*scale]) / cls.U16MAX
        b = array([0, -scale, -scale, -scale])
        return keys * m + b

    @classmethod
    def keys_float_to_int(cls, keys, scale=1.0, dur=1.0):
        m = array([dur, 2*scale, 2*scale, 2*scale]) / cls.U16MAX
        b = array([0, -scale, -scale, -scale])
        return (keys - b) / m

    @classmethod
    def deserialize_keys(cls, stream, keyCount):
        return np.fromfile(stream.base_stream, cls.U16, keyCount*cls.KEY_SIZE
            ).reshape((keyCount, cls.KEY_SIZE))

    @classmethod
    def serialize_keys(cls, stream, keys):
        keys.tofile(stream.base_stream)

    def deserialize(self, stream):
        self.name = stream.readCString().decode('ascii')
        (self.priority, rotKeyCount) = stream.unpack("ii")
        self._rotKeys = self.deserialize_keys(stream, rotKeyCount)
        (locKeyCount,) = stream.unpack("i")
        self._locKeys = self.deserialize_keys(stream, locKeyCount)
        return self

    def serialize(self, stream):
        stream.writeCString(self.name.encode('ascii'))
        stream.pack("ii", self.priority, len(self.rotKeys))
        self.serialize_keys(stream, self.rotKeys)
        stream.pack("i", len(self.locKeys))
        self.serialize_keys(stream, self.locKeys)

    def __repr__(self):
        return "<%s %s>" % (self.__class__.__name__, self)

    def __str__(self):
        return 'P%d %dR %dL: %s' % (
            self.priority, len(self.rotKeys), len(self.locKeys), self.name)


class JointConstraintSharedData(object):
    pass


class KeyframeMotion(object):
    def __init__(
            self,
            *,
            priority=3,
            duration = 0.0,
            emote = '',
            loopIn = 0.0,
            loopOut = None,
            loop = 1,
            easeIn = 0.8,
            easeOut = 0.8,
            handPosture = 1,
            ):
        self.version = 1
        self.subVersion = 0
        self.priority = priority
        self.duration = duration
        self.emote = emote
        self.loopIn = loopIn
        self.loopOut = duration if loopOut is None else loopOut
        self.loop = loop
        self.easeIn = easeIn
        self.easeOut = easeOut
        self.handPosture = handPosture
        self.joints = list()
        self.constraints = list()

    def deserialize(self, file):
        stream = BinaryStream(file)
        (self.version, self.subVersion, self.priority, self.duration) = stream.unpack("HHif")
        self.emote = stream.readCString().decode('ascii')
        (self.loopIn, self.loopOut, self.loop, self.easeIn, self.easeOut, self.handPosture, jointCount) = stream.unpack("ffiffii")
        self.joints = list()
        for jointNum in range(jointCount):
            joint = JointMotion()
            self.joints.append(joint)
            joint.deserialize(stream)
        (constraintCount,) = stream.unpack("i")
        self.constraints = list()
        for constraintNum in range(constraintCount):
            constraint = JointConstraintSharedData()
            self.constraints.append(constraint)
            (constraint.chainLength, constraint.type) = stream.unpack("BB")
            constraint.sourceVolume = stream.unpack("16s")[0].decode('ascii')
            constraint.sourceOffset = stream.unpack("3f")
            constraint.targetVolume = stream.unpack("16s")[0].decode('ascii')
            constraint.targetOffset = stream.unpack("3f")
            constraint.targetDir   = stream.unpack("3f")
            (constraint.easeInStart, constraint.easeInStop, constraint.easeOutStart, constraint.easeOutStop) = stream.unpack("4f")
        return self

    def serialize(self, file):
        stream = BinaryStream(file)
        stream.pack("HHif", self.version, self.subVersion, self.priority, self.duration)
        stream.writeCString(self.emote.encode('ascii'))
        stream.pack("ffiffii", self.loopIn, self.loopOut, self.loop, self.easeIn, self.easeOut, self.handPosture, len(self.joints))
        for joint in self.joints:
            joint.serialize(stream)
        stream.pack("i", len(self.constraints))
        for constraint in self.constraints:
            stream.pack("BB", constraint.chainLength, constraint.type)
            stream.pack("16s3f16s3f3f4f",
                constraint.sourceVolume.encode('ascii'), *constraint.sourceOffset,
                constraint.targetVolume.encode('ascii'), *constraint.targetOffset,
                *constraint.targetDir,
                constraint.easeInStart, constraint.easeInStop, constraint.easeOutStart, constraint.easeOutStop)

    def deserialize_filename(self, filename):
        print("reading " + filename)
        with open(filename, 'rb') as f:
            return self.deserialize(f)

    def serialize_filename(self, filename):
        print("writing " + filename)
        with open(filename, 'wb') as f:
            self.serialize(f)

    def dump(self):
        print("version: %d.%d" % (self.version, self.subVersion))
        print("priority: %d" % (self.priority,))
        print("duration: %f" % (self.duration,))
        print('emote: "%s"' % (self.emote,))
        print('loop: %d (%f - %f)' % (self.loop, self.loopIn, self.loopOut))
        print('ease: %f - %f' % (self.easeIn, self.easeOut))
        print('joints: %d' % (len(self.joints),))
        for joint in self.joints:
            print('\t%s' % joint)

        print('constraints: %d' % (len(self.constraints),))
        for constraint in self.constraints:
            print("\tchain: %d type: %d\n\t\t%s + %s ->\n\t\t%s + %s at %s\n\t\tease: %f, %f - %f, %f" %
                (constraint.chainLength, constraint.type,
                constraint.sourceVolume, constraint.sourceOffset,
                constraint.targetVolume, constraint.targetOffset,
                constraint.targetDir,
                constraint.easeInStart, constraint.easeInStop, constraint.easeOutStart, constraint.easeOutStop))

    def summary(self, filename=None):
        rotJointCount = 0
        locJointCount = 0
        for joint in self.joints:
            if (joint.rotKeys.size): rotJointCount += 1
            if (joint.locKeys.size): locJointCount += 1
        summary = 'P%d %dR %dL %dC' % (self.priority, rotJointCount, locJointCount, len(self.constraints))
        if filename:
            summary = '%s: %s' % (filename, summary)
        return summary

    def summarize(self, filename=None):
        print(self.summary(filename))

    def new_joint(self, name, priority=None, **kwargs):
        joint = JointMotion(name, self.priority if priority is None else priority, **kwargs)
        self.joints.append(joint)
        return joint

    def __getitem__(self, item):
        for joint in self.joints:
            if joint.name == item:
                return joint
        raise KeyError(item)


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


class SetAnimPriority(AnimTransform):
    def __init__(self, priority):
        self.priority = priority

    def __call__(self, anim):
        for joint in anim.joints:
            if joint.priority == anim.priority:
                joint.priority = self.priority
        anim.priority = self.priority


class SpeedAnimation(AnimTransform):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, anim):
        anim.duration *= self.factor
        anim.loopIn *= self.factor
        anim.loopOut *= self.factor


class OffsetJoint(AnimTransform):
    def __init__(self, *args):
        self.joint = 'mPelvis'
        offset = np.zeros(JointMotion.KEY_SIZE)
        if len(args) == 1:
            offset[3] = args[0]
        elif len(args) == 2:
            self.joint = args[0]
            offset[3] = args[1]
        elif len(args) == 3:
            offset[1:4] = args
        elif len(args) == 4:
            self.joint = args[0]
            offset[1:4] = args[1:4]
        else:
            raise ValueError("1-4 arguments required")
        self.offset = np.array(offset / JointMotion.LOC_MAX / 2 * JointMotion.U16MAX, JointMotion.U16)

    def __call__(self, anim):
        try:
            joint = anim[self.joint]
        except KeyError:
            joint = anim.new_joint(self.joint, locKeys=np.zeros[JointMotion.KEY_SIZE])
        joint._locKeys += self.offset


class TransformJointsMatching(AnimTransform):
    def __init__(self, *globs, jointTransform):
        self.dropGlobs = list()
        self.keepGlobs = list()
        for glob in globs:
            if glob.startswith('+'):
                self.keepGlobs.append(glob[1:])
            else:
                self.dropGlobs.append(glob);
        self.jointTransform = jointTransform

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
    joint.locKeys = []


def dropRotationKeyframes(anim, joint):
    joint.rotKeys = []


def dropPriority(anim, joint):
    joint.priority = anim.priority


def dropJoint(anim, joint):
    anim.joints.remove(joint)


class DropEmptyJoints(AnimTransform):
    def __call__(self, anim):
        anim.joints = [joint for joint in anim.joints if
                (joint.rotKeys.size or joint.locKeys.size)]


class AddConstraint(AnimTransform):
    def __init__(self, sourceVolume, targetVolume, chainLength):
        self.constraint = JointConstraintSharedData()
        self.constraint.chainLength = int(chainLength)
        self.constraint.type = 0 # 0: point, 1: plane

        self.constraint.sourceVolume = sourceVolume
        self.constraint.sourceOffset = (0, 0, 0)

        self.constraint.targetVolume = targetVolume
        self.constraint.targetOffset = (0, 0, 0)
        self.constraint.targetDir = (0, 0, 0)

        self.constraint.easeInStart = -1
        self.constraint.easeInStop = 0
        self.constraint.easeOutStart = 10
        self.constraint.easeOutStop = 10

    def __call__(self, anim):
        anim.constraints.append(self.constraint)


class DropConstraints(AnimTransform):
    def __call__(self, anim):
        anim.constraints = list()


class SetConstraintType(AnimTransform):
    def __init__(self, constraintType):
        self.constraintType = constraintType

    def __call__(self, anim):
        constraint = anim.constraints[-1]
        constraint.type = self.constraintType


class SetConstraintEase(AnimTransform):
    def __init__(self, easeInStart, easeInStop, easeOutStart,
            easeOutStop):
        self.easeInStart = easeInStart
        self.easeInStop = easeInStop
        self.easeOutStart = easeOutStart
        self.easeOutStop = easeOutStop

    def __call__(self, anim):
        constraint = anim.constraints[-1]
        constraint.easeInStart = self.easeInStart
        constraint.easeInStop = self.easeInStop
        constraint.easeOutStart = self.easeOutStart
        constraint.easeOutStop = self.easeOutStop


class SetConstraintSourceOffset(AnimTransform):
    def __init__(self, x, y, z):
        self.offset = (x, y, z)

    def __call__(self, anim):
        constraint = anim.constraints[-1]
        constraint.sourceOffset = self.offset


class SetConstraintTargetOffset(AnimTransform):
    def __init__(self, x, y, z):
        self.offset = (x, y, z)

    def __call__(self, anim):
        constraint = anim.constraints[-1]
        constraint.targetOffset = self.offset


class SetConstraintTargetDir(AnimTransform):
    def __init__(self, x, y, z):
        self.offset = (x, y, z)

    def __call__(self, anim):
        constraint = anim.constraints[-1]
        constraint.targetDir = self.offset


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
        #print('%r %r %r' % (namespace, values, self.kwargs))
        item = self.func(*values, **self.kwargs)
        items = copy.copy(_ensure_value(namespace, self.dest, []))
        items.append(item)
        setattr(namespace, self.dest, items)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description='Manipulate Secondlife .anim files',
            fromfile_prefix_chars='@')
    parser.add_argument('files', type=argparse.FileType('rb'), nargs='+',
                        help='anim files to dump or process')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--outputfiles', '-o', type=argparse.FileType('wb'),
            nargs='*')

    parser.add_argument('--scale', '--speed', '-s', action=AppendObjectAction,
            dest='actions', func=SpeedAnimation, nargs=1, type=float)
    parser.add_argument('--offset', '--adjust', action=AppendObjectAction,
                        dest='actions', func=OffsetJoint, nargs='+',
                        help="Adjust joint location [joint] [x y] z. joint is mPelvis if omitted. x, y are 0 if omitted")

    parser.add_argument('--pri', action=AppendObjectAction,
            dest='actions', func=SetAnimPriority, nargs=1,
            type=int)
    parser.add_argument('--ease-in', action=AppendObjectAction,
            dest='actions', func=SetAnimProperty, nargs=1,
            key='easeIn', type=float)
    parser.add_argument('--ease-out', action=AppendObjectAction,
            dest='actions', func=SetAnimProperty, nargs=1,
            key='easeOut', type=float)
    parser.add_argument('--loop', action=AppendObjectAction,
            dest='actions', func=SetAnimProperty, nargs=1,
            key='loop', type=int)
    parser.add_argument('--loop-start', action=AppendObjectAction,
            dest='actions', func=SetAnimProperty, nargs=1,
            key='loopIn', type=float)
    parser.add_argument('--loop-end', action=AppendObjectAction,
            dest='actions', func=SetAnimProperty, nargs=1,
            key='loopOut', type=float)

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
    parser.add_argument('--c-plane', action=AppendObjectAction,
            dest='actions', func=SetConstraintType, constraintType=1)
    parser.add_argument('--c-ease', action=AppendObjectAction,
            dest='actions', func=SetConstraintEase, nargs=4, type=float)
    parser.add_argument('--c-source-offset', action=AppendObjectAction,
            dest='actions', func=SetConstraintSourceOffset, nargs=3, type=float)
    parser.add_argument('--c-target-offset', action=AppendObjectAction,
            dest='actions', func=SetConstraintTargetOffset, nargs=3, type=float)
    parser.add_argument('--c-target-dir', action=AppendObjectAction,
            dest='actions', func=SetConstraintTargetDir, nargs=3, type=float)

    args = parser.parse_args()
    _ensure_value(args, 'actions', [])

    if (args.verbose >= 2):
        print(args)

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
            print("different number of input and output files")
            sys.exit();
        for inputFile,outputFile in zip(args.files, args.outputfiles):
            anim = KeyframeMotion()
            anim.deserialize(inputFile)
            for action in args.actions:
                action(anim)
            anim.summarize(outputFile.name)
            if (args.verbose > 0):
                anim.dump()
            anim.serialize(outputFile)