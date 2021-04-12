# To install, download this file, and in blender, do Preferences > Add-ons > Install Add-on from File

# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
    "name": "SecondLife Animation (ANIM) format",
    "author": "Tapple Gao",
    "version": (2021, 4, 10),
    "blender": (2, 74, 0),
    "location": "File > Import-Export",
    "description": "Import SecondLife anim files to Avastar rigs",
    "warning": "",
    "wiki_url": "http://wiki.blender.org/index.php/Extensions:2.6/Py/"
                "Scripts/Import-Export/BVH_Importer_Exporter",
    "support": 'COMMUNITY',
    "category": "Import-Export"}

import struct
import math
from pathlib import Path
import numpy as np
import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Vector, Quaternion, Matrix


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

    def create_fcurves(self, action, dur, armature):
        bone_rot = armature.data.bones[self.name].matrix_local.to_quaternion()
        bone_rot_inv = bone_rot.inverted()
        if self.rotKeys.size:
            data_path = ('pose.bones["%s"].rotation_quaternion' % self.name)
            print("data_path = %s" % data_path)
            fw = action.fcurves.new(data_path, 0, self.name)
            fx = action.fcurves.new(data_path, 1, self.name)
            fy = action.fcurves.new(data_path, 2, self.name)
            fz = action.fcurves.new(data_path, 3, self.name)
            for t, x, y, z in self.get_rotKeysF(dur):
                w2 = 1 - x*x - y*y - z*z
                w = math.sqrt(w2) if w2 > 0 else 0
                print("%ft %fw %fx %fy %fz" % (t, w, x, y, z))
                q = Quaternion((w, y, -x, z))
                q = bone_rot_inv * q * bone_rot
                fw.keyframe_points.insert(t, q.w)
                fx.keyframe_points.insert(t, q.x)
                fy.keyframe_points.insert(t, q.y)
                fz.keyframe_points.insert(t, q.z)
        if self.locKeys.size:
            data_path = 'pose.bones["%s"].location' % self.name
            print("data_path = %s" % data_path)
            fx = action.fcurves.new(data_path, 0, self.name)
            fy = action.fcurves.new(data_path, 1, self.name)
            fz = action.fcurves.new(data_path, 2, self.name)
            for t, x, y, z in self.get_locKeysF(dur):
                print("%ft %fx %fy %fz" % (t, x, y, z))
                v = Vector((y, -x, z))
                v.rotate(bone_rot_inv)
                fx.keyframe_points.insert(t, v.x)
                fy.keyframe_points.insert(t, v.y)
                fz.keyframe_points.insert(t, v.z)
        data_path = 'pose.bones["%s"]["priority"]' % self.name
        f = action.fcurves.new(data_path, 0, self.name)
        f.keyframe_points.insert(0, self.priority)

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
            loopIn=0.0,
            loopOut=None,
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
        self.loopIn = loopIn
        self.loopOut = duration if loopOut is None else loopOut
        self.loop = loop
        self.easeIn = easeIn
        self.easeOut = easeOut
        self.handPosture = handPosture
        self.joints = list()
        self.constraints = list()
        self.frameRate = 30

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
            constraint.deserialize(stream)
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
            constraint.serialize(stream)

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
            print(constraint.dump())
    
    def create_action(self, name, armature):
        print("create_action(%s)" % name)
        action = bpy.data.actions.new(name=name)
        armature.animation_data.action = action

        action.AnimProps.Priority = self.priority
        action.AnimProps.frame_start = 0
        action.AnimProps.frame_end = self.duration * self.frameRate + 0.5
        # action.AnimProps.??? = self.emote
        action.AnimProps.Loop_In = self.loopIn * self.frameRate + 0.5
        action.AnimProps.Loop_Out = self.loopOut * self.frameRate + 0.5
        action.AnimProps.Loop = self.loop
        action.AnimProps.Ease_In = self.easeIn
        action.AnimProps.Ease_Out = self.easeOut
        action.AnimProps.Hand_Posture = str(self.handPosture)
        action.AnimProps.fps = self.frameRate
        for joint in self.joints:
            joint.create_fcurves(action, self.duration * self.frameRate, armature)
        for cu in action.fcurves:
            for bez in cu.keyframe_points:
                bez.interpolation = 'LINEAR'
        return action

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


def active_armature():
    if type(bpy.context.active_object.data) == bpy.types.Armature:
        return bpy.context.active_object
    if bpy.context.active_object.find_armature():
        return bpy.context.active_object.find_armature()
    if bpy.data.armatures:
        return bpy.data.armatures[0]
    raise RuntimeError("Please select an armature before importing")


def load(filename, armature=None):
    filepath = Path(filename)
    if not armature:
        armature = active_armature()
    with open(filename, 'rb') as file:
        anim = KeyframeMotion()
        anim.deserialize(file)
        # anim.summarize(file.name)
        # anim.dump()
        print(anim.create_action(filepath.stem, armature))


class ImportANIM(bpy.types.Operator, ImportHelper):
    """Load a SecondLife Animation file"""
    bl_idname = "import_anim.anim"
    bl_label = "Import SecondLife Anim"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".anim"
    filter_glob = StringProperty(default="*.anim", options={'HIDDEN'})

    def execute(self, context):
        print(self.as_keywords())
        load(self.filepath)
        return {'FINISHED'}


def menu_func_import(self, context):
    self.layout.operator(ImportANIM.bl_idname, text="SL Animation (.anim)")


def register():
    bpy.utils.register_module(__name__)
    bpy.types.INFO_MT_file_import.append(menu_func_import)


def unregister():
    bpy.utils.unregister_module(__name__)
    bpy.types.INFO_MT_file_import.remove(menu_func_import)


if __name__ == "__main__":
    # register()
    # load('Z:/fridge/blender-offline/quad/bc/Teeglepet/ripped anims/face_stripped_horse_anims/TH_roll1.anim')
#    load('Z:/fridge/blender-offline/quad/bc/Teeglepet/ripped anims/Joint Testing HUD/classic/body1 mPelvis-x.anim')
#    load('Z:/fridge/blender-offline/quad/bc/Teeglepet/ripped anims/Joint Testing HUD/classic/body1 mPelvis-y.anim')
#    load('Z:/fridge/blender-offline/quad/bc/Teeglepet/ripped anims/Joint Testing HUD/classic/body1 mPelvis-z.anim')
#    load('Z:/fridge/blender-offline/quad/bc/Teeglepet/ripped anims/Joint Testing HUD/classic/legL1 mHipLeft-x.anim')
#    load('Z:/fridge/blender-offline/quad/bc/Teeglepet/ripped anims/Joint Testing HUD/classic/legL1 mHipLeft-y.anim')
#    load('Z:/fridge/blender-offline/quad/bc/Teeglepet/ripped anims/Joint Testing HUD/classic/legL1 mHipLeft-z.anim')

    load('C:/Users/TAPPL/cabbage/tanimbomb/scripts/mPelvis_rot_x.anim')
    load('C:/Users/TAPPL/cabbage/tanimbomb/scripts/mPelvis_rot_y.anim')
    load('C:/Users/TAPPL/cabbage/tanimbomb/scripts/mPelvis_rot_z.anim')
    load('C:/Users/TAPPL/cabbage/tanimbomb/scripts/mHipLeft_rot_x.anim')
    load('C:/Users/TAPPL/cabbage/tanimbomb/scripts/mHipLeft_rot_y.anim')
    load('C:/Users/TAPPL/cabbage/tanimbomb/scripts/mHipLeft_rot_z.anim')
    load('C:/Users/TAPPL/cabbage/tanimbomb/scripts/mHipRight_rot_x.anim')
    load('C:/Users/TAPPL/cabbage/tanimbomb/scripts/mHipRight_rot_y.anim')
    load('C:/Users/TAPPL/cabbage/tanimbomb/scripts/mHipRight_rot_z.anim')
