from math import sin, cos, pi
import animDump
import numpy as np

JOINTS_NAMES = [
    "mPelvis",
    # "mSpine1", "mSpine2", "mTorso", "mSpine3", "mSpine4", "mChest", "mNeck", "mHead", "mSkull", "mEyeLeft", "mEyeRight",
    # "mCollarLeft", "mShoulderLeft", "mElbowLeft", "mWristLeft",
    # "mCollarRight", "mShoulderRight", "mElbowRight", "mWristRight",
    # "mWingsRoot", "mGroin", "mHindLimbsRoot",
    # "mWing1Left", "mWing2Left", "mWing3Left", "mWing4Left", "mWing4FanLeft",
    # "mWing1Right", "mWing2Right", "mWing3Right", "mWing4Right", "mWing4FanRight",
    "mHipLeft",
    # "mKneeLeft", "mAnkleLeft", "mFootLeft", "mToeLeft",
    "mHipRight",
    # "mKneeRight", "mAnkleRight", "mFootRight", "mToeRight",
    # "mTail1", "mTail2", "mTail3", "mTail4", "mTail5", "mTail6",
    # "mHindLimb1Left", "mHindLimb2Left", "mHindLimb3Left", "mHindLimb4Left",
    # "mHindLimb1Right", "mHindLimb2Right", "mHindLimb3Right", "mHindLimb4Right",
    #
    # "mHandThumb1Left", "mHandThumb2Left", "mHandThumb3Left",
    # "mHandIndex1Left", "mHandIndex2Left", "mHandIndex3Left",
    # "mHandMiddle1Left", "mHandMiddle2Left", "mHandMiddle3Left",
    # "mHandRing1Left", "mHandRing2Left", "mHandRing3Left",
    # "mHandPinky1Left", "mHandPinky2Left", "mHandPinky3Left",
    # "mHandThumb1Right", "mHandThumb2Right", "mHandThumb3Right",
    # "mHandIndex1Right", "mHandIndex2Right", "mHandIndex3Right",
    # "mHandMiddle1Right", "mHandMiddle2Right", "mHandMiddle3Right",
    # "mHandRing1Right", "mHandRing2Right", "mHandRing3Right",
    # "mHandPinky1Right", "mHandPinky2Right", "mHandPinky3Right",
    #
    # "mFaceRoot",
    # "mFaceEyeAltLeft", "mFaceEyeAltRight",
    # "mFaceForeheadLeft", "mFaceForeheadCenter", "mFaceForeheadRight",
    # "mFaceEyebrowOuterLeft", "mFaceEyebrowCenterLeft", "mFaceEyebrowInnerLeft",
    # "mFaceEyebrowOuterRight", "mFaceEyebrowCenterRight", "mFaceEyebrowInnerRight",
    # "mFaceEyeLidUpperLeft", "mFaceEyeLidLowerLeft",
    # "mFaceEyeLidUpperRight", "mFaceEyeLidLowerRight",
    # "mFaceEyecornerInnerLeft", "mFaceEyecornerInnerRight",
    # "mFaceEar1Left", "mFaceEar2Left", "mFaceEar1Right", "mFaceEar2Right",
    # "mFaceNoseBase", "mFaceNoseBridge", "mFaceNoseLeft", "mFaceNoseCenter", "mFaceNoseRight",
    # "mFaceCheekLowerLeft", "mFaceCheekUpperLeft", "mFaceCheekLowerRight", "mFaceCheekUpperRight",
    # "mFaceChin", "mFaceJaw", "mFaceJawShaper",
    # "mFaceTeethLower", "mFaceTeethUpper", "mFaceTongueBase", "mFaceTongueTip",
    # "mFaceLipLowerLeft", "mFaceLipLowerCenter", "mFaceLipLowerRight",
    # "mFaceLipUpperLeft", "mFaceLipUpperCenter", "mFaceLipUpperRight",
    # "mFaceLipCornerLeft", "mFaceLipCornerRight",
    #
    # "PELVIS", "BUTT", "BELLY", "LEFT_HANDLE", "RIGHT_HANDLE", "LOWER_BACK", "CHEST", "LEFT_PEC", "RIGHT_PEC", "UPPER_BACK", "NECK", "HEAD",
    # "L_CLAVICLE", "L_UPPER_ARM", "L_LOWER_ARM", "L_HAND",
    # "R_CLAVICLE", "R_UPPER_ARM", "R_LOWER_ARM", "R_HAND",
    # "L_UPPER_LEG", "L_LOWER_LEG", "L_FOOT",
    # "R_UPPER_LEG", "R_LOWER_LEG", "R_FOOT",
    #
    # "Avatar Center",
    # "Pelvis", "Stomach", "Left Pec", "Right Pec", "Chest", "Spine", "Neck",
    # "Mouth", "Chin", "Nose", "Skull", "Left Ear", "Right Ear", "Left Eyeball", "Right Eyeball",
    # "Alt Left Ear", "Alt Right Ear", "Alt Left Eye", "Alt Right Eye", "Jaw", "Tongue",
    # "Tail Base", "Tail Tip",
    # "Left Wing", "Right Wing",
    # "Groin",
    # "Left Hind Foot", "Right Hind Foot",
    # "Left Shoulder", "L Upper Arm", "L Forearm", "Left Hand", "Left Ring Finger",
    # "Right Shoulder", "R Upper Arm", "R Forearm", "Right Hand", "Right Ring Finger",
    # "Left Hip", "L Upper Leg", "L Lower Leg", "Left Foot",
    # "Right Hip", "R Upper Leg", "R Lower Leg", "Right Foot",
    # "Center 2", "Top Right", "Top", "Top Left", "Center", "Bottom Left", "Bottom", "Bottom Right",
]


def make_rot_anim(joint_name, axis, dur=2.0, frames=12):
    """ make a looping rotation of joint about axis """
    keys = np.zeros((frames+1, 4))
    for i in range(frames+1):
        t = i / frames
        x = sin(pi*t)  # pi not tau because quaternion double cover
        w = cos(pi*t)
        if w < 0:
            x = -x
        keys[i][0] = t
        keys[i][axis+1] = x
    anim = animDump.KeyframeMotion(priority=6, easeIn=0.0, easeOut=0.0, duration=dur)
    anim.new_joint(joint_name, rotKeysF=keys)
    axis_name = ['x', 'y', 'z'][axis]
    anim.serialize_filename('%s_rot_%s.anim' % (joint_name, axis_name))


for joint_name in JOINTS_NAMES:
    for axis in range(3):
        make_rot_anim(joint_name, axis)
