from src.utils import convert_array_3x3matrix_to_euler_zyx, copy_object, has_missing_keyframes_between, \
    export_dict_to_file
from mathutils import Matrix
import numpy as np
import bpy
import os


def test_convert_array_3x3matrix_to_euler_zyx():
    euler = convert_array_3x3matrix_to_euler_zyx(np.array([[
        Matrix([[10, 20, 30], [40, 50, 60], [70, 80, 90]]),
        Matrix([[-10, -20, -30], [-40, -50, -60], [-70, -80, -90]])
    ]]))

    np.testing.assert_allclose(euler, np.array(
        [
            [
                [11.83, 30.74, 168.17],
                [146.88, -66.61, -108.67]
            ]
        ]
    ), atol=1e-2, rtol=1e-2)


def test_copy_object():
    object1 = bpy.data.objects.new("object1", bpy.data.meshes.new('mesh1'))
    bpy.context.collection.objects.link(object1)
    object1.select_set(True)
    object2 = copy_object(object1, bpy.context)
    assert object1 not in bpy.context.selected_objects
    assert object2.name == "object1.001"
    assert object2.animation_data is not None
    assert object2 in bpy.context.selected_objects


def test_has_missing_keyframes_between_when_no_keyframe_missing():
    object1 = bpy.data.objects.new("object1", bpy.data.meshes.new('mesh1'))
    object1.animation_data_create()
    action = bpy.data.actions.new(name="action1")
    object1.animation_data.action = action

    for axis_i in range(3):
        curve = action.fcurves.new(data_path="test", index=axis_i)
        keyframe_points = curve.keyframe_points
        keyframe_points.add(5)

        for frame_i in range(1, 5):
            keyframe_points[frame_i].co = (frame_i, 0)

    assert not has_missing_keyframes_between(object1, (0, 5))


def test_has_missing_keyframes_between_when_keyframe_missing():
    object1 = bpy.data.objects.new("object1", bpy.data.meshes.new('mesh1'))
    object1.animation_data_create()
    action = bpy.data.actions.new(name="action1")
    object1.animation_data.action = action

    for axis_i in range(3):
        curve = action.fcurves.new(data_path="test", index=axis_i)
        keyframe_points = curve.keyframe_points
        keyframe_points.add(5)

        for frame_i in range(1, 5):
            if frame_i != 2:
                keyframe_points[frame_i].co = (frame_i, 0)

    assert has_missing_keyframes_between(object1, (0, 5))


def test_has_missing_keyframes_between_when_keyframe_missing_not_in_range():
    object1 = bpy.data.objects.new("object1", bpy.data.meshes.new('mesh1'))
    object1.animation_data_create()
    action = bpy.data.actions.new(name="action1")
    object1.animation_data.action = action

    for axis_i in range(3):
        curve = action.fcurves.new(data_path="test", index=axis_i)
        keyframe_points = curve.keyframe_points
        keyframe_points.add(5)

        for frame_i in range(1, 5):
            if frame_i != 1:
                keyframe_points[frame_i].co = (frame_i, 0)

    assert not has_missing_keyframes_between(object1, (1, 5))


def test_has_missing_keyframes_between_when_no_animation_data():
    object1 = bpy.data.objects.new("object1", bpy.data.meshes.new('mesh1'))
    assert not has_missing_keyframes_between(object1, (0, 5))


def test_has_missing_keyframes_between_when_no_action():
    object1 = bpy.data.objects.new("object1", bpy.data.meshes.new('mesh1'))
    object1.animation_data_create()
    assert not has_missing_keyframes_between(object1, (0, 5))


def test_export_dict_to_file_txt():
    export_dict_to_file({"test1": "aaa", "test2": "bbb"}, "testfile", "TXT")
    assert os.path.isfile("testfile.txt")
    with open("testfile.txt", "r") as f:
        lines = [line.strip() for line in f]
    assert len(lines) == 2
    assert lines[0] == "test1: aaa"
    assert lines[1] == "test2: bbb"
    os.remove("testfile.txt")


def test_export_dict_to_file_json():
    export_dict_to_file({"test1": "aaa", "test2": "bbb"}, "testfile", "JSON")
    assert os.path.isfile("testfile.json")
    with open("testfile.json", "r") as f:
        lines = [line.strip() for line in f]
    assert len(lines) == 4
    assert lines[0] == "{"
    assert lines[1] == '"test1": "aaa",'
    assert lines[2] == '"test2": "bbb"'
    assert lines[3] == "}"
    os.remove("testfile.json")


def test_export_dict_to_file_csv():
    export_dict_to_file({"test1": "aaa", "test2": "bbb"}, "testfile", "CSV")
    assert os.path.isfile("testfile.csv")
    with open("testfile.csv", "r") as f:
        lines = [line.strip() for line in f]
    assert len(lines) == 3
    assert lines[0] == "Name,Value"
    assert lines[1] == "test1,aaa"
    assert lines[2] == "test2,bbb"
    os.remove("testfile.csv")
