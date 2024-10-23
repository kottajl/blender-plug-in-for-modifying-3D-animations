# --- TODO improve

# import subprocess
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch'])



# --- Imports 

import sys
import os
import bpy
import json
import pathlib
import torch
import importlib

directory_path = pathlib.Path(bpy.context.space_data.text.filepath).parent.resolve()
modules_path = str(directory_path.parent.resolve())
sys.path.append(modules_path)

import src.metrics as metrics

from interface.general_interface import GeneralInterface

from src.addon_functions import apply_transforms, get_anim_data
from src.utils import copy_object, convert_array_3x3matrix_to_euler_zyx

from bpy.utils import register_class, unregister_class
from bpy.types import PropertyGroup
from bpy.props import IntProperty, PointerProperty, BoolProperty, EnumProperty
import bpy_extras

sys.path.remove(modules_path)



# --- Main function

def generate_anim(
    start_frame: int, 
    end_frame: int, 
    post_processing: bool, 
    interface: GeneralInterface,
    device: str,
    create_new: bool,
    obj_to_report: object
) -> set[str]:
    
    '''
    Function doing all stuff including using of model.
    '''

    context = bpy.context
    scene = context.scene  
    scene_start_frame = scene.frame_start
    scene_end_frame = scene.frame_end
    obj = context.object

    # check simple things
    if len(context.selected_objects) != 1 or obj is None or obj.type != 'ARMATURE':
        obj_to_report.report({'ERROR'}, "Something went wrong.")
        return {"CANCELLED"} 

    if start_frame < scene_start_frame or end_frame > scene_end_frame or start_frame >= end_frame: 
        obj_to_report.report({'ERROR'}, "Selected frames range is invalid.")
        return {"CANCELLED"}
    
    # check if frames are correct in model - function from interface
    if interface.check_frames_range(start_frame, end_frame, scene_start_frame, scene_end_frame) == False: 
        obj_to_report.report({'ERROR'}, "Model needs more frames before or after selected range.")
        return {"CANCELLED"} 
          
    # load anim data
    anim = get_anim_data(obj)
    
    # infer results from ai model - function from interface
    inferred_pos, inferred_rot = interface.infer_anim(anim, start_frame, end_frame, post_processing, device)
    
    # copy object
    if create_new: new_obj = copy_object(obj, context)
    else: new_obj = obj

    # convert original rotation to ZYX Euler angles
    original_rot = convert_array_3x3matrix_to_euler_zyx(anim["rotations"])

    # apply new transforms
    apply_transforms(
        new_obj,
        true_original_pos=anim["positions"], 
        true_inferred_pos=inferred_pos, 
        true_original_rot=original_rot, 
        true_inferred_rot=inferred_rot, 
        offset=start_frame 
    )
    
    # calculate metrics
    print(metrics.calculate_metrics(obj, new_obj, start_frame, end_frame))

    return {"FINISHED"}

# end function generate_anim



# --- Blender window stuff 

bl_info = {
    "name" : "Addon",
    "author" : "Pluginowcy",
    "version" : (1, 0),
    "blender" : (4, 0, 0),
    "location" : "View3D > N",
    "description" : "",
    "warning" : "",
    "doc_url" : "",
    "category" : "",
} 

selected_model : GeneralInterface = None
selected_device : str = None

def select_ai_model(index):
    model_path = pathlib.Path(get_ai_models()[index][2])
    model = importlib.import_module(str(model_path.name))
    global selected_model
    selected_model = model.ModelInterface()

# end function select_ai_model

def select_ai_model_dropdown(self, context):
    select_ai_model(int(self.model))

# end function select_ai_model_dropdown

def get_ai_models_dropdown(self, context):
    return get_ai_models()

# end function get_ai_models_dropdown

def get_ai_models():
    models = []
    i = 0
    for model in json.load(open(directory_path / "model_paths.json"))["model_paths"]:
        models.append((str(i), model["name"], model["path"]))
        model_path = pathlib.Path(model["path"])
        if not os.path.isabs(model_path):
            model_path = pathlib.Path(modules_path) / model_path
        sys.path.append(str(model_path.parent.resolve()))
        i += 1
    return models

# end function get_ai_models

def get_device_list(self, context):
    devices = [("0", "cpu", "cpu")] 
    device_i = 1

    # Cuda for NVIDIA GPU
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            devices.append((str(device_i), f"cuda:{i}", f"cuda:{i}"))
            device_i += 1

    # MPS for ARM GPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        devices.append((str(device_i), "mps", "mps"))
        device_i += 1

    return devices

# end function get_device_list

def select_device(self, context):
    global selected_device
    selected_device = get_device_list(self, context)[int(self.device)][2]

# end function select_device
                                                        
class GenerationProperties(PropertyGroup):
    start_frame : IntProperty(name = "start frame", default = 0, min = 0)
    end_frame : IntProperty(name = "end frame", default = 0, min = 0)
    model: EnumProperty(
        name="",
        description="Select model for generating frames",
        items=get_ai_models_dropdown,
        update=select_ai_model_dropdown
    )
    device: EnumProperty(
        name="",
        description="Select device to compute on",
        items=get_device_list,
        update=select_device
    )
    post_processing : BoolProperty(name = "", default = False)
    create_new : BoolProperty(name = "", default = True)
    
# end class GenerationProperties
     
class GenerationButtonOperator(bpy.types.Operator):
    bl_idname = "plugin.generation_button"
    bl_label = "plugin"

    def execute(self, context):
        mt = bpy.context.scene.my_tool
        return generate_anim(mt.start_frame, mt.end_frame, mt.post_processing, selected_model, selected_device, mt.create_new, self)
         
# end class GenerateButtonOperator

class AddModelButtonOperator(bpy.types.Operator, bpy_extras.io_utils.ImportHelper):
    bl_idname = "plugin.add_model_button"
    bl_label = "plugin"

    def execute(self, context):
        with open(directory_path / "model_paths.json", "r") as file:
            models = json.load(file)
        path = pathlib.Path(self.filepath)
        models["model_paths"].append({"name": str(path.stem), "path": str(path.parent.resolve() / path.stem)})
        with open(directory_path / "model_paths.json", "w") as file:
            file.write(json.dumps(models, indent=4))
        return {"FINISHED"}
         
# end class GenerateButtonOperator
       
class GenerationPanel(bpy.types.Panel):
    bl_label = "Addon"
    bl_idname = "plugin.generation_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Addon"
    bl_context = ""   

    def draw(self, context):
        layout = self.layout
        scene = context.scene  
        mytool = scene.my_tool

        row_0 = layout.row()
        row_0.alignment = 'CENTER'
        row_0.label(text="Model options")
 
        layout.label(text="  Selected model")
        layout.prop(mytool, "model")
        
        layout.label(text="  Computing device")
        layout.prop(mytool, "device")
             
        split_0 = layout.split(factor=0.77) 
        split_0.label(text="  Post processing")
        split_0.prop(mytool, "post_processing")

        layout.separator()
        layout.separator()
        
        row_1 = layout.row()
        row_1.alignment = 'CENTER'
        row_1.label(text="Addon options")
        
        layout.prop(mytool, "start_frame")
        layout.prop(mytool, "end_frame")   

        split_1 = layout.split(factor=0.77) 
        split_1.label(text="  Create new object")
        split_1.prop(mytool, "create_new")
        
        layout.separator()
        
        row_2 = layout.row()
        row_2.alignment = 'CENTER'
        row_2.label(text="Actions")
                  
        layout.operator(GenerationButtonOperator.bl_idname, text="Generate frames")
        layout.operator(AddModelButtonOperator.bl_idname, text="Add model from directory")
        
# end class GeneratePanel

def get_selected_keyframes():
    selected_frames = []
    action = bpy.context.object.animation_data.action  # Active action
    
    if action:
        for fcurve in action.fcurves:
            for keyframe in fcurve.keyframe_points:
                if keyframe.select_control_point:  
                    selected_frames.append(int(keyframe.co.x))

    return sorted(set(selected_frames))  

# end function get_selected_keyframes

class dope_sheet_generation_button(bpy.types.Operator):
    bl_idname = "dope.dope_sheet_generation_button"
    bl_label = "Generate frames"
    
    def execute(self, context):
        selected_frames = get_selected_keyframes()
        if len(selected_frames) != 2 or selected_frames[0] + 1 == selected_frames[1]:
            self.report({'ERROR'}, "Wrong frames range selected.")
            return {"CANCELLED"} 
        else:
            mt = bpy.context.scene.my_tool
            mt.start_frame, mt.end_frame = selected_frames[0], selected_frames[1]
            return generate_anim(mt.start_frame, mt.end_frame, mt.post_processing, selected_model, selected_device, mt.create_new, self)

# end class dope_sheet_generation_button  

class dope_sheet_options_button(bpy.types.Operator):
    bl_idname = "dope.dope_sheet_options_button"
    bl_label = "Generation options"
    bl_description = "Opens model generation options"
    
    def execute(self, context):
        return context.window_manager.invoke_popup(self)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool
                    
        layout.separator()
        
        split_0 = layout.split(factor=0.37)  
        split_0.label(text="Add model")
        split_0.operator(AddModelButtonOperator.bl_idname, text="Choose directory                          ")
 
        split_1 = layout.split(factor=0.37) 
        split_1.label(text="Selected model")
        split_1.prop(mytool, "model")
        
        split_2 = layout.split(factor=0.37) 
        split_2.label(text="Computing device")
        split_2.prop(mytool, "device")
        
        split_3 = layout.split(factor=0.37) 
        split_3.label(text="Post processing")
        split_3.prop(mytool, "post_processing")

        split_4 = layout.split(factor=0.37) 
        split_4.label(text="Create new object")
        split_4.prop(mytool, "create_new")
        
        layout.separator()
       
# end class dope_sheet_options_button

def draw_buttons_in_dope_sheet(self, context):
    layout = self.layout
    layout.separator()  
    layout.operator(dope_sheet_generation_button.bl_idname, text="Generate frames")
    layout.operator(dope_sheet_options_button.bl_idname, text="Generation options")

# end function draw_buttons_in_dope_sheet

generation_window_classes = [AddModelButtonOperator, GenerationButtonOperator, GenerationPanel, GenerationProperties]

def register():
    for x in generation_window_classes: register_class(x)  
    bpy.types.Scene.my_tool = PointerProperty(type=GenerationProperties)
    
    bpy.utils.register_class(dope_sheet_generation_button)
    bpy.utils.register_class(dope_sheet_options_button)
    bpy.types.DOPESHEET_MT_context_menu.append(draw_buttons_in_dope_sheet)
    
# end function register

def unregister():
    for x in generation_window_classes: unregister_class(x)
    del bpy.types.Scene.my_tool

    bpy.utils.unregister_class(dope_sheet_generation_button)
    bpy.utils.unregister_class(dope_sheet_options_button)
    bpy.types.DOPESHEET_MT_context_menu.remove(draw_buttons_in_dope_sheet)
    
# end function unregister
    
if __name__ == "__main__":   
    register()    
    select_ai_model(0)
    selected_device = "cpu"   
    
# end main
