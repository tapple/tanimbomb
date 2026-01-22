#!/usr/bin/env python3

import argparse
import fnmatch
import re
import struct
import copy
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from textwrap import wrap
import shutil
from collections.abc import Callable
import numpy as np

try:
    import quaternion
    HAS_QUATERNION = True
    qkey = np.dtype([("t", np.float32), ("q", np.quaternion)])
except ImportError:
    HAS_QUATERNION = False


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


def col_print(lines, term_width=None, indent=0, pad=2):
    """Print list of strings in multiple columns
    Original: https://gist.github.com/critiqjo/2ca84db26daaeb1715e1
    Adjusted: https://gist.github.com/Nachtalb/8a85c0793b4bea0a102b7414be5888d4
    """
    if not term_width:
        size = shutil.get_terminal_size((80, 20))
        term_width = size.columns

    n_lines = len(lines)
    if n_lines == 0:
        return ""

    col_width = max(len(line) for line in lines)
    n_cols = int((term_width + pad - indent) / (col_width + pad))
    n_cols = min(n_lines, max(1, n_cols))

    col_len = int(n_lines / n_cols) + (0 if n_lines % n_cols == 0 else 1)
    if (n_cols - 1) * col_len >= n_lines:
        n_cols -= 1

    cols = [lines[i * col_len: i * col_len + col_len] for i in range(n_cols)]

    rows = list(zip(*cols))
    rows_missed = zip(*[col[len(rows):] for col in cols[:-1]])
    rows.extend(rows_missed)

    return ("\n" + " " * indent).join(
        (" " * pad).join(
            line.ljust(col_width)
        for line in row)
    for row in rows)


class JointMotion(object):
    U16 = np.dtype('<H')
    U16MAX = 65535
    KEY_SIZE = 4
    LOC_MAX = 5

    def __init__(self, name='', priority=0, *,
            rotKeys=None, locKeys=None, rotKeysF=None, locKeysF=None, rotKeysQ=None):
        self.name = name
        self.priority = priority
        self.rotKeys = rotKeys or []
        self.locKeys = locKeys or []
        if rotKeysF is not None:
            self.rotKeysF = rotKeysF
        if locKeysF is not None:
            self.locKeysF = locKeysF
        if rotKeysQ is not None:
            self.rotKeysQ = rotKeysQ

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
    def rotKeysQ(self):
        import quaternion  # optional dependency; raise ImportError here if missing
        fkeys = self.rotKeysF
        xyz=fkeys[:, 1:]
        w = np.sqrt(1 - np.vecdot(xyz, xyz))[:, np.newaxis]
        q = quaternion.from_float_array(np.concat((w, xyz), axis=1))
        return np.rec.fromarrays((fkeys[:, 0], q), dtype=qkey)

    @rotKeysQ.setter
    def rotKeysQ(self, value):
        value = np.rec.array(value, dtype=qkey)
        a = quaternion.as_float_array(value.q)
        sign = ((a[:, 0] >= 0) * 2 - 1)[:, np.newaxis]
        self.rotKeysF = np.concat((value.t[:, np.newaxis], a[:, 1:] * sign), axis=1)

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

    def loc_range_squared(self):
        locKeysF = self.locKeysF
        if len(locKeysF) == 0:
            return 0
        return np.max(np.sum((locKeysF[:, 1:] - locKeysF[0, 1:]) ** 2, axis=1))

    def loc_range(self):
        return np.sqrt(self.loc_range_squared())

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
        return 'P%d %dR %dL (%.2fm): %s' % (
            self.priority, len(self.rotKeys), len(self.locKeys), self.loc_range(), self.name)

    def dump(self, dur=1.0, verbosity=0):
        print('  %s' % self)
        if verbosity == 1:
            print('    R:%s' % col_print(self.dump_keys_hex(self.rotKeys), indent=6, pad=1))
            print('    L:%s' % col_print(self.dump_keys_hex(self.locKeys), indent=6, pad=1))
        elif verbosity >= 2:
            print('    R:%s' % col_print(self.dump_keys_decimal(self.get_rotKeysF(dur=dur)), indent=6))
            print('    L:%s' % col_print(self.dump_keys_decimal(self.get_locKeysF(dur=dur)), indent=6))

    @staticmethod
    def dump_keys_hex(keys):
        return wrap(keys.tobytes().hex(), 16)

    @staticmethod
    def dump_keys_decimal(keys):
        return ["%7.4ft% .5fx% .5fy% .5fz" % tuple(frame) for frame in keys]


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
        return "  chain: %d type: %d\n    %s + %s ->\n    %s + %s at %s\n    ease: %f, %f - %f, %f" % (
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

    def dump(self, verbosity=0, sort=True):
        print("version: %d.%d" % (self.version, self.subVersion))
        print("priority: %d" % (self.priority,))
        print("duration: %f" % (self.duration,))
        print('emote: "%s"' % (self.emote,))
        print('loop: %d (%f - %f)' % (self.loop, self.loop_start, self.loop_end))
        print('ease: %f - %f' % (self.easeIn, self.easeOut))
        print('joints: %d' % (len(self.joints),))
        joints = self.joints
        if sort:
            joints = sorted(joints, key=lambda joint: joint.name)
        for joint in joints:
            joint.dump(self.duration, verbosity)

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
        if self.duration == 0.0:
            return None
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

    def loc_range_squared(self):
        return max((joint.loc_range_squared() for joint in self.joints), default=0)

    def loc_range(self):
        return np.sqrt(self.loc_range_squared())

    def summary(self, filename=None, markdown=False):
        rotJointCount = 0
        locJointCount = 0
        for joint in self.joints:
            if (joint.rotKeys.size): rotJointCount += 1
            if (joint.locKeys.size): locJointCount += 1
        format = '|%d|%2d|%3d|%4.2f|%3d|%3.1f|%3.1f|%7.4f|%s|%5.2f|%7.4f|%5.2f|' if markdown else 'P%d%3dR%3dL (%4.2fm)%2dC %3.1f-%3.1fEs %6.3fs %s (%4.2fin + %5.2f + %5.2fout)'
        summary = format % (
            self.priority, rotJointCount, locJointCount, self.loc_range(), len(self.constraints),
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
        """ create a new joint with priority derived from this animation """
        return JointMotion(name, self.priority if priority is None else priority, **kwargs)

    def get_joint_or_none(self, name):
        """ Return the joint named "item", or None if missing """
        try:
            return self[name]
        except KeyError:
            return None

    def get_joint_or_blank(self, name, priority=None, **kwargs):
        """ Return the joint named "item", or a blank one if missing """
        try:
            return self[name]
        except KeyError:
            return self.new_joint(name, priority, **kwargs)

    def ensure_joint(self, name, priority=None, **kwargs):
        """ Return the joint named "item", or add a blank one if missing """
        try:
            return self[name]
        except KeyError:
            joint = self.new_joint(name, priority, **kwargs)
            self.joints.append(joint)
            return joint

    def __getitem__(self, item):
        """ Return the joint named "item", or KeyError if missing """
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

class TransformBuilder(ABC):
    def __init__(self, *commands: str, initial_value:float=0):
        for command in commands:
            self.add_command(command)

    @abstractmethod
    def __bool__(self): ...

    @abstractmethod
    def add_command(self, command: str): ...

class XYZTransform(TransformBuilder):
    def __init__(self, *commands: str, initial_value:float=0):
        self.value = np.zeros(JointMotion.KEY_SIZE) + initial_value
        self.defined = np.zeros(JointMotion.KEY_SIZE, dtype=bool)
        for command in commands:
            self.add_command(command)

    def __repr__(self):
        return f"""{self.__class__.__name__}{tuple(
            str(v) + ' xyz'[i]
            for i, (v, d) in enumerate(zip(self.value, self.defined))
            if d
        )}"""

    def __bool__(self):
        return bool(self.defined.any())

    def add_command(self, command: str):
        letters = re.findall(r'[xyz]', command)
        if len(letters) == 0:
            self.defined[1:4] = True
            self.value[1:4] = float(command)
        elif len(letters) == 1:
            letter = letters[0]
            i = " xyz".index(letter)
            value = float(command.replace(letter, ""))
            self.defined[i] = True
            self.value[i] = value
        else:
            raise ValueError(f"Must contain zero or one of x, y, or z: {command}")


class XYZQuaternionTransform(TransformBuilder):
    def __init__(self, *commands: str, initial_value:float|np.quaternion=0):
        import quaternion  # optional dependency; raise ImportError here if missing
        if isinstance(initial_value, np.quaternion):
            self.value = initial_value
        else:
            self.value = quaternion.one
        for command in commands:
            self.add_command(command)

    def __repr__(self):
        return f"""{self.__class__.__name__}{tuple(
            f"{round(v, 4)}{'zyz'[i]}" 
            for i, v in enumerate(reversed(np.degrees(quaternion.as_euler_angles(self.value))))
            if abs(v) > 0.0001
        )}"""

    def __bool__(self):
        return self.value != quaternion.one

    def add_command(self, command: str):
        letters = re.findall(r'[xyz]', command)
        if len(letters) == 1:
            letter = letters[0]
            i = "xyz".index(letter)
            r = np.zeros(3)
            r[i] = np.radians(float(command.replace(letter, "")))
            self.value = quaternion.from_rotation_vector(r) * self.value
        else:
            raise ValueError(f"Must contain one of x, y, or z: {command}")


class XYZTransformJointsMatching(AnimTransform):
    def __init__(self, *globs: str, transform_func: Callable[[KeyframeMotion, JointMotion, XYZTransform], None], starting_globs: tuple[str], transform_factory=XYZTransform, initial_value:float=0):
        self.match_globs: dict[str, XYZTransform] = {}
        self.ignore_globs = list()
        current_glob = ""
        for glob in starting_globs + globs:
            if self.is_bone_glob(glob):
                if glob.startswith('+'):
                    self.ignore_globs.append(glob[1:])
                else:
                    self._discard_glob_if_zero(current_glob)
                    self.match_globs[glob] = transform_factory(initial_value=initial_value)
                    current_glob = glob
            elif current_glob:
                self.match_globs[current_glob].add_command(glob)
            else:
                raise ValueError(f"No joints specified for {transform_func}")
        self._discard_glob_if_zero(current_glob)
        self.transform_func = transform_func
        self.transform_factory = transform_factory

    @staticmethod
    def is_bone_glob(command: str):
        return len(re.findall(r'[a-wA-Z?*]', command)) > 0

    def _discard_glob_if_zero(self, current_glob):
        if current_glob in self.match_globs and not self.match_globs[current_glob]:
            del self.match_globs[current_glob]

    def __repr__(self):
        return "XYZTransformJointsMatching(%r, %r, %r)" % (self.match_globs,
                                                        self.ignore_globs, self.transform_func)

    def transform_for_joint(self, joint_name: str) -> XYZTransform:
        for match_glob, transform in self.match_globs.items():
            if (fnmatch.fnmatch(joint_name, match_glob)):
                for ignore_glob in self.ignore_globs:
                    if (fnmatch.fnmatch(joint_name, ignore_glob)):
                        return self.transform_factory()
                return transform
        return self.transform_factory()

    def __call__(self, anim):
        # iterate over a copy since joints may be removed
        for joint in anim.joints[:]:
            transform = self.transform_for_joint(joint.name)
            if transform:
                self.transform_func(anim, joint, transform)


def offset_joint(anim: KeyframeMotion, joint: JointMotion, transform: XYZTransform):
    joint.locKeysF += transform.value


def set_joint_loc(anim: KeyframeMotion, joint: JointMotion, transform: XYZTransform):
    offset = (transform.value - joint.locKeysF[0]) * transform.defined
    print(f"offsetting {joint.name} by {offset[1:4]}")
    joint.locKeysF += offset


def rotate_joint(anim: KeyframeMotion, joint: JointMotion, transform: XYZQuaternionTransform):
    qkeys = joint.rotKeysQ
    qkeys.q = transform.value * qkeys.q
    joint.rotKeysQ = qkeys


def scale_joint(anim: KeyframeMotion, joint: JointMotion, transform: XYZTransform):
    joint.locKeysF *= transform.value


class ExtendAnim(AnimTransform):
    def __init__(self, dur, prepend=False):
        self.dur = dur
        self.prepend = prepend

    def __call__(self, anim: KeyframeMotion):
        new_dur = anim.duration + self.dur
        scale_factor = anim.duration / new_dur
        scale  = np.array([scale_factor, 1, 1, 1])
        offset = np.array([0, 0, 0, 0])
        if self.prepend:
            offset[0] =1 - scale_factor
        for joint in anim.joints:
            joint.rotKeysF = joint.rotKeysF * scale + offset
            joint.locKeysF = joint.locKeysF * scale + offset

        if self.prepend:
            anim.loop_start += self.dur
            anim.loop_end   += self.dur
        anim.duration = new_dur


class AppendAnim(AnimTransform):
    DUR_MIN = 0.01

    def __init__(self, extra_anim_filename, prepend=False):
        self.extra_anim = KeyframeMotion()
        self.extra_anim.deserialize_filename(extra_anim_filename)
        self.prepend = prepend

    def __call__(self, anim: KeyframeMotion):
        if self.prepend:
            anim_1, anim_2 = self.extra_anim, anim
        else:
            anim_1, anim_2 = anim, self.extra_anim

        dur_1 = max(anim_1.duration, self.DUR_MIN)
        dur_2 = max(anim_2.duration, self.DUR_MIN)
        new_dur = dur_1 + dur_2
        scale_1  = np.array([dur_1/new_dur, 1, 1, 1])
        scale_2  = np.array([dur_2/new_dur, 1, 1, 1])
        offset_2 = np.array([dur_1/new_dur, 0, 0, 0])

        for joint in self.extra_anim.joints:
            anim.ensure_joint(joint.name)
        for joint in anim.joints:
            joint_1 = anim_1.get_joint_or_blank(joint.name)
            joint_2 = anim_2.get_joint_or_blank(joint.name)
            joint.rotKeysF = np.concatenate([
                joint_1.rotKeysF * scale_1,
                joint_2.rotKeysF * scale_2 + offset_2,
            ], )
            joint.locKeysF = np.concatenate([
                joint_1.locKeysF * scale_1,
                joint_2.locKeysF * scale_2 + offset_2,
            ])

        if self.prepend:
            anim.loop_start += dur_1
            anim.loop_end   += dur_1
        #     anim.easeIn = anim_1.easeIn
        # else:
        #     anim.easeOut = anim_2.easeOut
        anim.duration = new_dur


def interp_get(arr, t) -> tuple[np.ndarray, int]:
    i = np.searchsorted(arr[:, 0], t)
    if 1 == 0 or i == len(arr):
        return arr[i], i
    a = arr[i-1]
    b = arr[i]
    return a + (b - a) * (t - a[0]) / (b[0]- a[0]), i


def interp_insert(arr, t) -> tuple[np.ndarray, int]:
    if len(arr) == 0:
        return arr, 0
    v, i = interp_get(arr, t)
    return np.insert(arr, i, v, 0), i


class RightTruncateAnim(AnimTransform):
    def __init__(self, t):
        self.t = t

    def __call__(self, anim: KeyframeMotion):
        t_frac = self.t / anim.duration
        if t_frac > 1:
            return

        t_ufrac = t_frac * JointMotion.U16MAX
        t_scale = np.array([1/t_frac, 1, 1, 1])
        for joint in anim.joints:
            keys, i = interp_insert(joint.locKeys, t_ufrac)
            joint.locKeys = keys[:i] * t_scale
            keys, i = interp_insert(joint.rotKeys, t_ufrac)
            joint.rotKeys = keys[:i] * t_scale

        anim.duration = self.t


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
        loc_scale  = np.array([1,  1, -1,  1]).astype(JointMotion.U16)
        loc_offset = np.array([0,  0, -1,  0]).astype(JointMotion.U16)
        rot_scale  = np.array([1, -1,  1, -1]).astype(JointMotion.U16)
        rot_offset = np.array([0, -1,  0, -1]).astype(JointMotion.U16)
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


class SetJointPriority(AnimTransform):
    def __init__(self, jointGlob, priority):
        self.jointGlob = jointGlob
        self.priority = int(priority)

    def __call__(self, anim):
        for joint in anim.joints:
            if fnmatch.fnmatch(joint.name, self.jointGlob):
                joint.priority = self.priority


class TransformJointsMatching(AnimTransform):
    def __init__(self, *globs: str, transform_func: Callable[[KeyframeMotion, JointMotion], None], default_globs=()):
        self.match_globs = list()
        self.ignore_globs = list()
        for glob in globs or default_globs:
            if glob.startswith('+'):
                self.ignore_globs.append(glob[1:])
            else:
                self.match_globs.append(glob);
        self.transform_func = transform_func

    def __repr__(self):
        return "TransformJointsMatching(%r, %r, %r)" % (self.match_globs,
                self.ignore_globs, self.transform_func)

    def joint_matches(self, joint_name):
        for dropGlob in self.match_globs:
            if (fnmatch.fnmatch(joint_name, dropGlob)):
                for keepGlob in self.ignore_globs:
                    if (fnmatch.fnmatch(joint_name, keepGlob)):
                        return False
                return True
        return False

    def __call__(self, anim):
        # iterate over a copy since joints may be removed
        for joint in anim.joints[:]:
            if self.joint_matches(joint.name):
                self.transform_func(anim, joint)


def freeze_keyframes(anim: KeyframeMotion, joint: JointMotion):
    joint.locKeys = joint.locKeys[:1]
    joint.rotKeys = joint.rotKeys[:1]


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


class SortJoints(AnimTransform):
    def __call__(self, anim):
        anim.joints.sort(key=lambda joint: joint.name)


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


def main():
    def output_filename(input_filename):
        if args.outputfile_pattern is None:
            return input_filename

        input_path = Path(input_filename)
        output_path = Path(args.outputfile_pattern
                           .replace("%n", input_path.stem)
                           .replace("%p", str(input_path.parent.resolve()))
                           .replace("%%", "%")).with_suffix(input_path.suffix)

        return str(output_path)

    np.set_printoptions(precision=5, suppress=True, sign=' ', floatmode='fixed')

    parser = argparse.ArgumentParser(
            description='Manipulate Secondlife .anim files',
            fromfile_prefix_chars='@', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('files', nargs='+', help='anim files to dump or process')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--unordered', '-U', action='store_false', dest='ordered',
                        help="when printing joints with -v, show in file order rather than abc order")
    parser.add_argument('--markdown', '--md', action='store_true', help="output in markdown table")
    parser.add_argument('--outputfile-pattern', '-o',
                        help="""Output anim file path/name, with template substitution:
    %%n: input file name
    %%p: input file directory
    %%%%: literal '%%'
File extension will be appended automatically""")
    parser.add_argument('--time-scale', '--tscale', '--speed', '-s', action=AppendObjectAction,
            dest='actions', func=SpeedAnimation, nargs=1, type=float,
    help = """Adjust duration by the given factor eg:
    2.0 for half-speed/double duration, or
    0.5 for double speed/half duration""")
    parser.add_argument('--frame-rate', '--fps', action=AppendObjectAction,
                        dest='actions', func=SetFrameRate, nargs=1, type=int)
    parser.add_argument('--offset', '--adjust', action=AppendObjectAction,
            dest='actions', func=XYZTransformJointsMatching, nargs='+',
            transform_func=offset_joint, starting_globs=("mPelvis",),
    help="""Adjust joint location on all the given joint patterns (mPelvis by default). Examples:
    "--offset 0.5z": move mPelvis up 0.5
    "--offset mFaceNose* -0.2y 0.1x": move nose bones right 0.2m, forward 0.1m""")
    parser.add_argument('--loc', action=AppendObjectAction,
            dest='actions', func=XYZTransformJointsMatching, nargs='+',
            transform_func=set_joint_loc, starting_globs=("mPelvis",),
            help="Move joint location on all keyframes so the starting location is at the given coordinates. "
                 "Specify x y z to select which coordinates to set; missing axes will remain unchanged. "
                 """Joint is optional, and defaults to mPelvis. Example:
    "--loc mPelvis* 0x 0.25z": Move the animation so that it starts at 0 on x and 0.25 on z. Leave y alone""")
    if HAS_QUATERNION:
        parser.add_argument('--rotate', '--rot', action=AppendObjectAction,
            dest='actions', func=XYZTransformJointsMatching, nargs='+',
            transform_func=rotate_joint, transform_factory=XYZQuaternionTransform, starting_globs=("mPelvis",),
    help="""Rotate each given joint (or mPelvis by default) on the given axes the given joint patterns. Examples:
    "--rot 45z": rotate mPelvis 45° left around z
    "--rot mFaceNose* -20y mFaceJaw 30y": rotate nose bones 20° up; rotate jaw 30° down""")
    parser.add_argument('--scale', action=AppendObjectAction,
            dest='actions', func=XYZTransformJointsMatching, nargs='*',
            transform_func=scale_joint, starting_globs=("*",), initial_value=1,
            help="Scale location keys; eg 2.0 for double-size avatar, 0.5 for half-size avatar. Can specify joint patterns or x y z to control individual joints")
    parser.add_argument('--extend', action=AppendObjectAction,
                        dest='actions', func=ExtendAnim, nargs=1, type=float, prepend=False,
                        help="Hold the final pose for the given number of seconds in the outro of the animation")
    parser.add_argument('--delay', action=AppendObjectAction,
                        dest='actions', func=ExtendAnim, nargs=1, type=float, prepend=True,
                        help="Add the given number of seconds to the intro of the animation")
    parser.add_argument('--append', action=AppendObjectAction,
                        dest='actions', func=AppendAnim, nargs=1, prepend=False,
                        help="Add keyframes another anim file to the end of the ease out")
    parser.add_argument('--prepend', action=AppendObjectAction,
                        dest='actions', func=AppendAnim, nargs=1, prepend=True,
                        help="Add keyframes another anim file to the beginning of the ease in")
    parser.add_argument('--trim' '--rtrim', '--trunc', '--rtrunc', action=AppendObjectAction,
                        dest='actions', func=RightTruncateAnim, nargs=1, type=float,
                        help="Remove the end of the animation starting at the given time")
    parser.add_argument('--mirror', '--flip', action=AppendObjectAction,
                        dest='actions', func=MirrorJoints, nargs=0)
    parser.add_argument('--sort', action=AppendObjectAction,
                        dest='actions', func=SortJoints,
                        help="Sort the joints by name in the output files")
    parser.add_argument('--joint-pri', action=AppendObjectAction,
                        dest='actions', func=SetJointPriority, nargs=2)

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

    parser.add_argument('--freeze', '--static', action=AppendObjectAction,
            dest='actions', func=TransformJointsMatching, nargs='*',
            transform_func=freeze_keyframes, default_globs=("*",),
            help="Freeze the given joint patterns by removing all keyframes except the first (if no joints given, freeze all joints)")
    parser.add_argument('--drop-loc', action=AppendObjectAction,
            dest='actions', func=TransformJointsMatching, nargs='*',
            transform_func=dropLocationKeyframes, default_globs=("*", "+mPelvis"),
            help = "Drop location keyframes from the given joint patterns (if none specified, drop all except mPelvis)")
    parser.add_argument('--drop-rot', action=AppendObjectAction,
            dest='actions', func=TransformJointsMatching, nargs='+',
            transform_func=dropRotationKeyframes, help = "Drop rotation keyframes from the given joint patterns")
    parser.add_argument('--drop-pri', action=AppendObjectAction,
            dest='actions', func=TransformJointsMatching, nargs='*',
            transform_func=dropPriority, default_globs=("*",),
            help = "Drop per-joint priority from the given joint patterns (if none specified, drop from all joints)")
    parser.add_argument('--drop-joint', action=AppendObjectAction,
            dest='actions', func=TransformJointsMatching, nargs='+',
            transform_func=dropJoint, help = "Drop the given joint patterns")
    parser.add_argument('--drop-empty-joints', action=AppendObjectAction,
            dest='actions', func=DropEmptyJoints,
            help = "Drop all joints that have 0 rotation keyframes and 0 location keyframes")

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
        print('|Filename|Pri|Rots|Locs|Range|Cons|E In|E Out|Dur|Loop|L In|L Dur|L Out|FPS|Frames|')
        print('|--------|--:|---:|---:|----:|---:|---:|----:|--:|---:|---:|----:|----:|--:|-----:|')
    max_file_len = max(len(output_filename(file)) for file in args.files)
    format = f"%-{max_file_len}s"
    for filename in args.files:
        anim = KeyframeMotion()
        anim.deserialize_filename(filename)
        for action in args.actions:
            action(anim)
        anim.summarize(format % output_filename(filename), markdown=args.markdown)
        if args.verbose > 0:
            anim.dump(verbosity=args.verbose-1, sort=args.ordered)
        if args.outputfile_pattern:
            anim.serialize_filename(output_filename(filename))

if __name__ == '__main__':
    main()
