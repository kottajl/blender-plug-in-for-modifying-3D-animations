class Bone:
    def __init__(self, name, parent, head_local=None, matrix=None, matrix_local=None):
        self.name = name
        self.parent = parent
        self.head_local = head_local
        self.matrix = matrix
        self.matrix_local = matrix_local


class ValueIterableDict(dict):
    def __iter__(self):
        return iter(self.values())

class Armature:
    def __init__(self, bones):
        self.bones = ValueIterableDict(dict([(bone.name, bone) for bone in bones]))


class Pose:
    def __init__(self, bones):
        self.bones = ValueIterableDict(dict([(bone.name, bone) for bone in bones]))


class Object:
    def __init__(self, armature, pose):
        self.data = armature
        self.pose = pose


class Scene:
    def __init__(self, frame_start, frame_end):
        self.frame_start = frame_start
        self.frame_end = frame_end
