import src.addon_functions as add_f
from src.test.types import Bone, Object, Armature, Pose
from mathutils import Matrix, Vector


def test_should_return_skeleton():
    bone1 = Bone("bone1", None)
    bone2 = Bone("bone2", bone1)
    bone3 = Bone("bone3", bone1)
    bone4 = Bone("bone4", bone2)
    assert sorted(add_f.get_object_skeleton(Object(Armature([bone1, bone2, bone3, bone4]), None))) == \
           sorted([("bone1", None), ("bone3", "bone1"), ("bone2", "bone1"), ("bone4", "bone2")])


def test_get_anim_data():
    a_bone1 = Bone("bone1", None, Vector([0, 0, 0]), Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
    a_bone2 = Bone("bone2", a_bone1, Vector([0, 0, 0]), Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
    a_bone3 = Bone("bone3", a_bone1, Vector([0, 0, 0]), Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
    a_bone4 = Bone("bone4", a_bone2, Vector([0, 0, 0]), Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
    p_bone1 = Bone("bone1", None, Vector([0, 0, 0]), Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
    p_bone2 = Bone("bone2", p_bone1, Vector([0, 0, 0]), Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
    p_bone3 = Bone("bone3", p_bone1, Vector([0, 0, 0]), Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
    p_bone4 = Bone("bone4", p_bone2, Vector([0, 0, 0]), Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
    assert add_f.get_anim_data(Object(Armature([a_bone1, a_bone2, a_bone3, a_bone4]), Pose([p_bone1, p_bone2, p_bone3, p_bone4])))
