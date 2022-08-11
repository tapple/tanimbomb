#!/usr/bin/env python3

import argparse
import fnmatch
import struct
import copy
import sys
from pathlib import Path
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
        if rotKeysF is not None:
            self.rotKeysF = rotKeysF
        if locKeysF is not None:
            self.locKeysF = locKeysF

    @property
    def rotKeys(self):
        return self._rotKeys

    @rotKeys.setter
    def rotKeys(self, value):
        self._rotKeys = np.array(np.round(np.clip(value, 0, self.U16MAX)),
            self.U16).reshape((-1, self.KEY_SIZE))

    @property
    def locKeys(self):
        return self._locKeys

    @locKeys.setter
    def locKeys(self, value):
        self._locKeys = np.array(np.round(np.clip(value, 0, self.U16MAX)),
            self.U16).reshape((-1, self.KEY_SIZE))

    @property
    def rotKeysF(self):
        return self.keys_int_to_float(self.rotKeys)

    @rotKeysF.setter
    def rotKeysF(self, value):
        self.rotKeys = self.keys_float_to_int(value)

    def get_rotKeysF(self, dur=1.0, round_zero = True):
        return self.keys_int_to_float(self.rotKeys, dur=dur, round_zero=round_zero)

    @property
    def locKeysF(self):
        return self.keys_int_to_float(self.locKeys, self.LOC_MAX)

    @locKeysF.setter
    def locKeysF(self, value):
        self.locKeys = self.keys_float_to_int(value, self.LOC_MAX)

    def get_locKeysF(self, dur=1.0, round_zero=True):
        return self.keys_int_to_float(self.locKeys, self.LOC_MAX, dur=dur, round_zero=round_zero)

    @classmethod
    def keys_int_to_float(cls, keys, scale=1.0, dur=1.0, round_zero=True):
        m = array([dur, 2*scale, 2*scale, 2*scale]) / cls.U16MAX
        b = array([0, -scale, -scale, -scale])
        ans = keys * m + b
        if round_zero:
            for frame in ans:
                for i in range(1, 4):
                    if abs(frame[i]) < scale / cls.U16MAX:
                        frame[i] = 0
        return ans

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
    def deserialize(self, stream):
        (self.chainLength, self.type) = stream.unpack("BB")
        self.sourceVolume = stream.unpack("16s")[0].decode('ascii')
        self.sourceOffset = stream.unpack("3f")
        self.targetVolume = stream.unpack("16s")[0].decode('ascii')
        self.targetOffset = stream.unpack("3f")
        self.targetDir = stream.unpack("3f")
        (self.easeInStart, self.easeInStop, self.easeOutStart, self.easeOutStop) = stream.unpack("4f")
        return self

    def serialize(self, stream):
        stream.pack("BB", self.chainLength, self.type)
        stream.pack("16s3f16s3f3f4f",
            self.sourceVolume.encode('ascii'), *self.sourceOffset,
            self.targetVolume.encode('ascii'), *self.targetOffset,
            *self.targetDir,
            self.easeInStart, self.easeInStop, self.easeOutStart, self.easeOutStop)

    def dump(self):
        return "\tchain: %d type: %d\n\t\t%s + %s ->\n\t\t%s + %s at %s\n\t\tease: %f, %f - %f, %f" % (
            self.chainLength, self.type,
            self.sourceVolume, self.sourceOffset,
            self.targetVolume, self.targetOffset,
            self.targetDir,
            self.easeInStart, self.easeInStop, self.easeOutStart, self.easeOutStop,
        )


class KeyframeMotion(object):
    def __init__(
            self,
            # *, # Python 3.5
            priority=3,
            duration=0.0,
            emote='',
            loop_start=0.0,
            loop_end=-0.0,
            loop=1,
            easeIn=0.8,
            easeOut=0.8,
            handPosture=1,
            ):
        self.version = 1
        self.subVersion = 0
        self.priority = priority
        self.duration = duration
        self.emote = emote
        self.loop_start = loop_start
        self.loop_end = loop_end
        self.loop = loop
        self.easeIn = easeIn
        self.easeOut = easeOut
        self.handPosture = handPosture
        self.joints = list()
        self.constraints = list()

    @property
    def loop_start(self):
        return self._loop_start

    @loop_start.setter
    def loop_start(self, value):
        if np.copysign(1, value) < 0:  # negative, including -0
            self._loop_start = self.duration + value
        else:
            self._loop_start = value

    @property
    def loop_end(self):
        return self._loop_end

    @loop_end.setter
    def loop_end(self, value):
        if np.copysign(1, value) < 0:  # negative, including -0
            self._loop_end = self.duration + value
        else:
            self._loop_end = value

    def deserialize(self, file):
        stream = BinaryStream(file)
        (self.version, self.subVersion, self.priority, self.duration) = stream.unpack("HHif")
        self.emote = stream.readCString().decode('ascii')
        (self._loop_start, self._loop_end, self.loop, self.easeIn, self.easeOut, self.handPosture, jointCount) = stream.unpack("ffiffii")
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
            constraint.deserialize(stream)
        return self

    def serialize(self, file):
        stream = BinaryStream(file)
        stream.pack("HHif", self.version, self.subVersion, self.priority, self.duration)
        stream.writeCString(self.emote.encode('ascii'))
        stream.pack("ffiffii", self.loop_start, self.loop_end, self.loop, self.easeIn, self.easeOut, self.handPosture, len(self.joints))
        for joint in self.joints:
            joint.serialize(stream)
        stream.pack("i", len(self.constraints))
        for constraint in self.constraints:
            constraint.serialize(stream)

    def deserialize_filename(self, filename):
        # print("reading " + filename)
        with open(filename, 'rb') as f:
            return self.deserialize(f)

    def serialize_filename(self, filename):
        # print("writing " + filename)
        with open(filename, 'wb') as f:
            self.serialize(f)

    def dump(self, keys=False, sort=True):
        print("version: %d.%d" % (self.version, self.subVersion))
        print("priority: %d" % (self.priority,))
        print("duration: %f" % (self.duration,))
        print('emote: "%s"' % (self.emote,))
        print('loop: %d (%f - %f)' % (self.loop, self.loop_start, self.loop_end))
        print('ease: %f - %f' % (self.easeIn, self.easeOut))
        print('joints: %d' % (len(self.joints),))
        joints = self.joints
        if sort:
            joints.sort(key=lambda joint: joint.name)
        for joint in joints:
            print('\t%s' % joint)
            if keys:
                print('\t\tR:%s' % joint.rotKeys.tobytes().hex())
                print('\t\tL:%s' % joint.locKeys.tobytes().hex())

        print('constraints: %d' % (len(self.constraints),))
        for constraint in self.constraints:
            print(constraint.dump())

    def all_keyframe_times(self):
        if not self.joints:
            return np.array([], dtype=JointMotion.U16)
        locKeys = np.concatenate([joint.locKeys[:,0] for joint in self.joints])
        rotKeys = np.concatenate([joint.rotKeys[:,0] for joint in self.joints])
        return np.unique(np.concatenate((locKeys, rotKeys)))

    def calculate_frame_rate(self):
        diff_keys = np.diff(self.all_keyframe_times())
        if diff_keys.size == 0:
            return None
        frame_time = np.min(diff_keys)
        for tries in range(5):
            diff_frames = diff_keys / frame_time
            err = np.abs(diff_frames / np.rint(diff_frames) - 1)
            if np.max(err) < 0.05:
                return np.rint(JointMotion.U16MAX / self.duration / frame_time)
            frame_time *= np.max(diff_frames - np.floor(diff_frames))
        return None

    def summary(self, filename=None, markdown=False):
        rotJointCount = 0
        locJointCount = 0
        for joint in self.joints:
            if (joint.rotKeys.size): rotJointCount += 1
            if (joint.locKeys.size): locJointCount += 1
        format = '|%d|%2d|%2d|%2d|%3.1f|%3.1f|%7.4f|%s|%5.2f|%7.4f|%5.2f|' if markdown else 'P%d %2dR %2dL %2dC %3.1f-%3.1fEs %6.3fs %s (%5.2fin + %5.2f + %5.2fout)'
        summary = format % (
            self.priority, rotJointCount, locJointCount, len(self.constraints),
            self.easeIn, self.easeOut,
            self.duration, "  looped" if self.loop else "unlooped",
            self.loop_start, self.loop_end - self.loop_start, self.duration - self.loop_end,
        )

        frame_rate = self.calculate_frame_rate()
        if frame_rate:
            format = '%s%2d|%4d|' if markdown else '%s at %2dfps (%4d frames)'
            summary = format % (summary, frame_rate, self.duration * frame_rate)
        if filename:
            format = '|%s%s' if markdown else '%s: %s'
            summary = format % (filename, summary)
        return summary

    def summarize(self, filename=None, markdown=False):
        print(self.summary(filename, markdown))

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
        anim.loop_start *= self.factor
        anim.loop_end *= self.factor


class SetFrameRate(AnimTransform):
    def __init__(self, target_frame_rate):
        self.target_frame_rate = target_frame_rate

    def __call__(self, anim):
        frame_rate = anim.calculate_frame_rate()
        if frame_rate:
            factor = frame_rate / self.target_frame_rate
            anim.duration *= factor
            anim.loop_start *= factor
            anim.loop_end *= factor


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


class MirrorJoints(AnimTransform):
    ROTATED_JOINTS = {
        "CHEST": [0, -10, 0],
        "LEFT_PEC": [0, 4.29, 0],
        "RIGHT_PEC": [0, 4.29, 0],
        "L_UPPER_ARM": [-5, 0, 0],
        "R_UPPER_ARM": [5, 0, 0],
        "L_LOWER_ARM": [-3, 0, 0],
        "R_LOWER_ARM": [3, 0, 0],
        "L_HAND": [-3, 0, -10],
        "R_HAND": [3, 0, 10],
        "L_FOOT": [0, 10, 0],
        "R_FOOT": [0, 10, 0],
        "Chest": [0, 90, 90],
        "Spine": [0, -90, 90],
        "Skull": [0, 0, 90],
    }

    def __init__(self, *args):
        pass

    def __call__(self, anim: KeyframeMotion):
        loc_scale  = np.array([1,  1, -1,  1], JointMotion.U16)
        loc_offset = np.array([0,  0, -1,  0], JointMotion.U16)
        rot_scale  = np.array([1, -1,  1, -1], JointMotion.U16)
        rot_offset = np.array([0, -1,  0, -1], JointMotion.U16)
        for joint in anim.joints:
            assert joint.name not in self.ROTATED_JOINTS
            joint.name = self.mirror_joint_name(joint.name)
            joint.locKeys = joint.locKeys * loc_scale + loc_offset
            joint.rotKeys = joint.rotKeys * rot_scale + rot_offset

    @staticmethod
    def mirror_joint_name(name: str):
        if "Left" in name:
            return name.replace("Left", "Right", 1)
        elif "Right" in name:
            return name.replace("Right", "Left", 1)
        elif "LEFT" in name:
            return name.replace("LEFT", "RIGHT", 1)
        elif "RIGHT" in name:
            return name.replace("RIGHT", "LEFT", 1)
        elif name.startswith("L_"):
            return name.replace("L_", "R_", 1)
        elif name.startswith("R_"):
            return name.replace("R_", "L_", 1)
        elif name.startswith("L "):
            return name.replace("L ", "R ", 1)
        elif name.startswith("R "):
            return name.replace("R ", "L ", 1)
        else:
            return name


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


def output_filename(input_filename):
    if args.outputfile_pattern is None:
        return input_filename
    input_path = Path(input_filename)
    return f"{args.outputfile_pattern}{{suffix}}".format(input_path.stem, suffix=input_path.suffix)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description='Manipulate Secondlife .anim files',
            fromfile_prefix_chars='@')
    parser.add_argument('files', nargs='+', help='anim files to dump or process')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--no-sort', '-U', action='store_false', dest='sort',
                        help="when printing joints with -v, show in file order rather than abc order")
    parser.add_argument('--markdown', '--md', action='store_true', help="output in markdown table")
    parser.add_argument('--outputfile-pattern', '-o')

    parser.add_argument('--scale', '--speed', '-s', action=AppendObjectAction,
            dest='actions', func=SpeedAnimation, nargs=1, type=float)
    parser.add_argument('--frame-rate', '--fps', action=AppendObjectAction,
                        dest='actions', func=SetFrameRate, nargs=1, type=int)
    parser.add_argument('--offset', '--adjust', action=AppendObjectAction,
                        dest='actions', func=OffsetJoint, nargs='+',
                        help="Adjust joint location [joint] [x y] z. joint is mPelvis if omitted. x, y are 0 if omitted")
    parser.add_argument('--mirror', '--flip', action=AppendObjectAction,
                        dest='actions', func=MirrorJoints, nargs=0)

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
            key='loop_start', type=float)
    parser.add_argument('--loop-end', action=AppendObjectAction,
            dest='actions', func=SetAnimProperty, nargs=1,
            key='loop_end', type=float)

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

    if args.markdown:
        print('|Filename|Pri|Rots|Locs|Cons|E In|E Out|Dur|Loop|L In|L Dur|L Out|FPS|Frames|')
        print('|--------|--:|---:|---:|---:|---:|----:|--:|---:|---:|----:|----:|--:|-----:|')
    max_file_len = max(len(output_filename(file)) for file in args.files)
    format = f"%-{max_file_len}s"
    for filename in args.files:
        anim = KeyframeMotion()
        anim.deserialize_filename(filename)
        for action in args.actions:
            action(anim)
        anim.summarize(format % output_filename(filename), markdown=args.markdown)
        if args.verbose > 0:
            anim.dump(keys=args.verbose > 1, sort=args.sort)
        if args.outputfile_pattern:
            anim.serialize_filename(output_filename(filename))