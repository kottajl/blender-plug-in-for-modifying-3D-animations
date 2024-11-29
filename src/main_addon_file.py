# --- Instaling libraries for addon in Blender python instance if not installed

import subprocess
import sys
import importlib
import os
import bpy
import pathlib
import json
import platform 

directory_path = pathlib.Path(bpy.context.space_data.text.filepath).parent.resolve()
modules_path = str(directory_path.parent.resolve())
sys.path.append(modules_path)

os_name = platform.system()
req_path = str(directory_path) + str(os.sep) + "addon_requirements.txt"

def handle_pytorch3d_installation():
    try:
        import pytorch3d
    except ModuleNotFoundError:
        print("Started installing pytorch3d.")
        if os_name == "Windows":
            pytorch3d_whl_path = str(directory_path).removesuffix("src") + "lib" +  str(os.sep) + "pytorch3d-0.7.8-cp310-cp310-win_amd64.whl"
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', pytorch3d_whl_path])
            except Exception as e:
                print("\033[33m" + str(e) + "\033[0m")
                return False
            else:
                print("Successfully installed pytorch3d.")
                return True
            
        elif os_name == "Darwin": # Mac
            print("To install pytorch3d on MAC you have to install it from source.")
            print("https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#building--installing-from-source")
            print("Install it in Blender Python instance using python -m [commands].")
            return False
    
        else: # Linux
            try:
                import torch
                version_str = "".join([
                    "py38_cu",
                    torch.version.cuda.replace(".",""),
                    "_pyt1110"
                ])
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', "iopath"])
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', "pytorch3d", "-f", f"https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html"])
            except Exception as e:
                print("\033[33m" + str(e) + "\033[0m")
                return False
            else:
                print("Successfully installed pytorch3d.")
                return True
            
    else:
        print("Pytorch3d already installed.")
        return True

def handle_torch_installation():
    try:
        import torch
        import torchaudio
        import torchvision
    except ModuleNotFoundError:
        pip_parts = [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio']
        if os_name == "Windows":
            pip_parts.append('--index-url')
            cuda_path = os.getenv('CUDA_PATH')
            if cuda_path: 
                cuda_version = cuda_path[-4] + cuda_path[-3] + cuda_path[-1]
                if cuda_version in ["124", "121"]: url = "https://download.pytorch.org/whl/cu" + cuda_version
                else: url = "https://download.pytorch.org/whl/cu118"
                pip_parts.append(url)
            else:
                pip_parts.append("https://download.pytorch.org/whl/cu118")

        elif os_name == "Linux":
            cuda_present = False
            try:
                result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    cuda_present = True
                    cuda_version_line = result.stdout.splitlines()[0]
                    cuda_version = cuda_version_line.split(",")[1].split()[1]  
                    cuda_version = str(cuda_version)
                    cuda_version = cuda_version[0] + cuda_version[1] + cuda_version[3]
                    if version in ["118", "121"]: 
                        pip_parts.append('--index-url')
                        url = "https://download.pytorch.org/whl/cu" + cuda_version
                        pip_parts.append(url)
            except FileNotFoundError:
                pass

            if not cuda_present:
                try:
                    result = subprocess.run(['rocminfo'], capture_output=True, text=True)
                    if result.returncode == 0:
                        pip_parts.append('--index-url')
                        pip_parts.append("https://download.pytorch.org/whl/rocm6.2")
                except FileNotFoundError:
                    pass
        
        try:
            print("Started installing torch, torchvision and torchaudio.")
            subprocess.check_call(pip_parts)
        except Exception as e:
            print("\033[33m" + str(e) + "\033[0m")
            return False
        else:
            print("Successfully installed torch, torchvision and torchaudio.")
            return True
                
    else:
        print("Torch, torchvision and torchaudio already installed.")
        return True

installing = False
with open(req_path, 'r', encoding='utf-8') as file:
    for line in file:
        x = line.strip()
        if not x or x.startswith('#'): continue

        pip_parts = [sys.executable, '-m', 'pip', 'install'] + [
            part.replace("==", ">=") if "==" in part else part for part in x.split()
        ]

        lib_name = pip_parts[4]
        if '>=' in lib_name: lib_name = lib_name.split('>=')[0]

        try:
            importlib.import_module(lib_name)
        except ModuleNotFoundError: 
            if not installing: 
                installing = True
                bpy.context.window_manager.popup_menu(
                    lambda self, context: self.layout.label(text="Started installing libraries."), 
                    title="Info", 
                    icon='INFO'
                )
                print("--- Updating PIP")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
                print("--- Installing libraries")

            if lib_name in ["torch", "torchvision", "torchaudio"]: handle_torch_installation()
            elif lib_name == "pytorch3d": handle_pytorch3d_installation()     
            else: subprocess.check_call(pip_parts)
                
if installing:
    print("--- Done")
    bpy.context.window_manager.popup_menu(
        lambda self, context: self.layout.label(text="Done installing libraries."), 
        title="Info", 
        icon='INFO'
   )



# --- Addon imports 

import torch

import src.metrics as metrics

from interface.general_interface import GeneralInterface

from src.addon_functions import apply_transforms, get_anim_data
from src.utils import copy_object, convert_array_3x3matrix_to_euler_zyx

from bpy.utils import register_class, unregister_class
from bpy.types import PropertyGroup
from bpy.props import IntProperty, PointerProperty, BoolProperty, EnumProperty, StringProperty, FloatProperty
import bpy_extras

sys.path.remove(modules_path)



# --- Main function

def generate_anim(
    start_frame: int, 
    end_frame: int, 
    interface: GeneralInterface,
    create_new: bool,
    obj_to_report: object,
    calculate_metrics: bool,
    **kwargs: dict
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
    if obj is None or obj.type != 'ARMATURE':
        obj_to_report.report({'ERROR'}, "Selected object must be armature.")
        return {"CANCELLED"} 
    
    if len(context.selected_objects) != 1: 
        obj_to_report.report({'ERROR'}, "Only 1 object can be selected.")
        return {"CANCELLED"}  

    if start_frame < scene_start_frame or end_frame > scene_end_frame or start_frame + 1 >= end_frame: 
        obj_to_report.report({'ERROR'}, "Selected frames range is invalid.")
        return {"CANCELLED"}
    
    # check if frames are correct in model - function from interface
    b, s = interface.check_frames_range(start_frame, end_frame, scene_start_frame, scene_end_frame)
    if b == False: 
        obj_to_report.report({'ERROR'}, f"Selected frames range is invalid in model, {s}")
        return {"CANCELLED"} 
    
    # load anim data
    try: 
        anim = get_anim_data(obj)
    except Exception as e: 
        obj_to_report.report({'ERROR'}, f"Error with loading animation data, {e}.")
        return {"CANCELLED"} 

    # infer results from ai model - function from interface
    try:
        print("--- Start of model infer anim logs")
        inferred_pos, inferred_rot = interface.infer_anim(anim, start_frame, end_frame, **kwargs)
        print("--- End of model infer anim logs")
    except Exception as e: 
        obj_to_report.report({'ERROR'}, f"Error with using model, {e}.")
        return {"CANCELLED"} 
    
    # copy object
    try:
        if create_new: new_obj = copy_object(obj, context)
        else: new_obj = obj
    except Exception as e: 
        obj_to_report.report({'ERROR'}, f"Error while copying object, {e}.")
        return {"CANCELLED"} 

    # convert original rotation to ZYX Euler angles
    try:
        original_rot = convert_array_3x3matrix_to_euler_zyx(anim["rotations"])
    except Exception as e: 
        obj_to_report.report({'ERROR'}, f"Error with animation data, {e}.")
        return {"CANCELLED"} 

    # apply new transforms
    try:
        apply_transforms(
        new_obj,
        true_original_pos=anim["positions"], 
        true_inferred_pos=inferred_pos, 
        true_original_rot=original_rot, 
        true_inferred_rot=inferred_rot, 
        offset=start_frame 
        )
    except Exception as e: 
        obj_to_report.report({'ERROR'}, f"Error while applying new animation to object, {e}.")
        return {"CANCELLED"} 
    
    # calculate metrics
    if calculate_metrics:
        print(metrics.calculate_metrics(
            orginal_obj=obj, 
            generated_obj=new_obj, 
            start_frame=start_frame, 
            end_frame=end_frame
        ))

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
our_model_num = 2
selected_model_index = 0
current_kwargs_attributes = []
device_arg_name = ""

def select_ai_model(index):  
    try:
        model_path = pathlib.Path(get_ai_models()[index][2])
        model = importlib.import_module(str(model_path.name))
        global selected_model, selected_model_index
        selected_model = model.ModelInterface()
        selected_model_index = index
        bpy.context.scene.my_tool.update_properties(bpy.context)
    except Exception as e:
        select_ai_model(0)
        raise(e)

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
    devices = [("0", "CPU", "cpu")] 
    device_i = 1

    # Cuda for NVIDIA GPU
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            devices.append((str(device_i), f"GPU {i}", f"cuda:{i}"))
            device_i += 1

    # MPS for ARM GPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        devices.append((str(device_i), "MPS", "mps"))
        device_i += 1

    return devices

# end function get_device_list

def select_device(self, context):
    global selected_device
    selected_device = get_device_list(self, context)[int(getattr(self, device_arg_name))][2]

# end function select_device
                                                        
class GenerationProperties(PropertyGroup): 
    start_frame : IntProperty(name = "Start frame", default = 0, min = 0)
    end_frame : IntProperty(name = "End frame", default = 0, min = 0)
    model: EnumProperty(
        name="AI model",
        description="Select model for generating frames",
        items=get_ai_models_dropdown,
        update=select_ai_model_dropdown,
        default=0
    )
    create_new : BoolProperty(name = "  Create new object", default = True)
    calculate_metrics: BoolProperty(name = "  Calculate metrics", default = True)

    def update_properties(self, context): 
        global current_kwargs_attributes
        for x in current_kwargs_attributes: delattr(GenerationProperties, x) 
        current_kwargs_attributes = []
    
        if selected_model is not None:
            kwargs_list = selected_model.get_infer_anim_kwargs()
            device_added = False
            for var_type, var_name, var_desc in kwargs_list:
                if var_type == torch.device and not device_added:
                    setattr(GenerationProperties, var_name, EnumProperty(
                        name= var_name,
                        description=var_desc,
                        items=get_device_list,
                        update=select_device,
                        default=0
                    ))
                    device_added = True
                    global device_arg_name
                    device_arg_name = var_name
                    current_kwargs_attributes.append(var_name)
                elif var_type == bool:
                    setattr(GenerationProperties, var_name, BoolProperty(
                        name= "  " + var_name,
                        description=var_desc,
                        default=False
                    ))
                    current_kwargs_attributes.append(var_name)
                elif var_type == int:
                    setattr(GenerationProperties, var_name, IntProperty(
                        name= var_name,
                        description=var_desc,
                        default=0
                    ))
                    current_kwargs_attributes.append(var_name)
                elif var_type == float:
                    setattr(GenerationProperties, var_name, FloatProperty(
                        name= var_name,
                        description=var_desc,
                        default=0.0
                    ))
                    current_kwargs_attributes.append(var_name)
                elif var_type == str:
                    setattr(GenerationProperties, var_name, StringProperty( 
                        name= var_name,
                        description=var_desc,
                        default=""
                    ))
                    current_kwargs_attributes.append(var_name)

# end class GenerationProperties
     
class GenerationButtonOperator(bpy.types.Operator):
    bl_idname = "plugin.generation_button"
    bl_label = "plugin"

    def execute(self, context):
        mt = bpy.context.scene.my_tool
        kwargs = dict()
        if selected_model is not None:
            kwargs_list = selected_model.get_infer_anim_kwargs()
            for t, n, d in kwargs_list:
                if t == torch.device: kwargs[n] = selected_device
                else: kwargs[n] = getattr(mt, n) 

        return generate_anim(mt.start_frame, mt.end_frame, selected_model, mt.create_new, self, mt.calculate_metrics, **kwargs)
         
# end class GenerateButtonOperator

class AddModelButtonOperator(bpy.types.Operator, bpy_extras.io_utils.ImportHelper):
    bl_idname = "plugin.add_model_button"
    bl_label = "plugin"

    filename_ext = ".py"
    filter_glob: bpy.props.StringProperty(default="*.py", options={'HIDDEN'})

    def execute(self, context):
        with open(directory_path / "model_paths.json", "r") as file:
            models = json.load(file)
        try:
            path = pathlib.Path(self.filepath)
            models["model_paths"].append({"name": str(path.stem), "path": str(path.parent.resolve() / path.stem)})
        except Exception as e:
            self.report({'ERROR'}, f"Someting went wrong while adding model, {e}.")
            return {"CANCELLED"} 
        
        with open(directory_path / "model_paths.json", "w") as file:
            file.write(json.dumps(models, indent=4))
        try:   
            select_ai_model((len(get_ai_models_dropdown(self, context)) - 1))
        except Exception as e:
            with open(directory_path / "model_paths.json", "r") as file:
                models = json.load(file)
            del models["model_paths"][len(models["model_paths"]) - 1]
            with open(directory_path / "model_paths.json", "w") as file:
                json.dump(models, file, indent=4)
            self.report({'ERROR'}, f"Someting went wrong while adding model, {e}.")
            return {"CANCELLED"} 
        return {"FINISHED"}
         
# end class AddModelButtonOperator

class InstallLibrariesOperator(bpy.types.Operator, bpy_extras.io_utils.ImportHelper):
    bl_idname = "plugin.install_libraries_button"
    bl_label = "Install"

    filename_ext = ".txt"
    filter_glob: bpy.props.StringProperty(default="*.txt", options={'HIDDEN'})

    def execute(self, context):
        print("--- Updating PIP")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])

        print("--- Installing libraries")
        not_installed = []

        with open(self.filepath, 'r', encoding='utf-8') as file:
            for line in file:
                x = line.strip()
                if not x or x.startswith('#'): continue

                pip_parts = [sys.executable, '-m', 'pip', 'install'] + [
                    part.replace("==", ">=") if "==" in part else part for part in x.split()
                ]

                lib_name = pip_parts[4]
                if '>=' in lib_name: lib_name = lib_name.split('>=')[0]

                if lib_name in ["torch", "torchvision", "torchaudio"]: 
                    if handle_torch_installation() == False: not_installed.append(x)
                    continue
                elif lib_name == "pytorch3d": 
                    if handle_pytorch3d_installation() == False: not_installed.append(x)
                    continue

                try:
                    subprocess.check_call(pip_parts)
                except Exception as e:
                    print("\033[33m" + str(e) + "\033[0m")
                    not_installed.append(x)

        if len(not_installed) > 0:
            print("--- Not installed libraries")
            for x in not_installed: print(x)
            print("--- End")
        else:
            print("--- Successfully installed all libraries")

        return {"FINISHED"}
    
# end class InstallLibrariesOperator

class DeleteModelButtonOperator(bpy.types.Operator):
    bl_idname = "plugin.delete_model_button"
    bl_label = "plugin"

    def execute(self, context):
        if selected_model_index < our_model_num:
            self.report({'ERROR'}, "Can't delete premade models.")
            return {"CANCELLED"} 
        else:
            with open(directory_path / "model_paths.json", "r") as file:
                models = json.load(file)
            del models["model_paths"][selected_model_index]
            with open(directory_path / "model_paths.json", "w") as file:
                json.dump(models, file, indent=4)
            select_ai_model(0)
        return {"FINISHED"}
         
# end class DeleteModelButtonOperator
       
class PLUGIN_PT_GenerationPanel(bpy.types.Panel):
    bl_label = "Addon"
    bl_idname = "PLUGIN_PT_generation_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Addon"
    bl_context = ""   

    def draw(self, context):
        layout = self.layout
        scene = context.scene  
        mytool = scene.my_tool

        layout.prop(mytool, "model")
        for x in current_kwargs_attributes: layout.prop(mytool, x)
        
        layout.separator()
        layout.separator()
        layout.separator()

        layout.operator(AddModelButtonOperator.bl_idname, text="Add model from directory")
        layout.operator(DeleteModelButtonOperator.bl_idname, text="Delete selected model")
        layout.operator(InstallLibrariesOperator.bl_idname, text="Install libraries from TXT") 

        layout.separator()
        layout.separator()
        layout.separator()
              
        layout.prop(mytool, "start_frame")
        layout.prop(mytool, "end_frame")   
        layout.prop(mytool, "create_new")
        layout.prop(mytool, "calculate_metrics")

        layout.operator(GenerationButtonOperator.bl_idname, text="Generate frames")
        
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
            kwargs = dict()
            if selected_model is not None:
                kwargs_list = selected_model.get_infer_anim_kwargs()
                for t, n, d in kwargs_list:
                    if t == torch.device: kwargs[n] = selected_device
                    else: kwargs[n] = getattr(mt, n) 

            return generate_anim(mt.start_frame, mt.end_frame, selected_model, mt.create_new, self, mt.calculate_metrics, **kwargs)

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

        layout.prop(mytool, "model")
        for x in current_kwargs_attributes: layout.prop(mytool, x)

        layout.separator()

        layout.prop(mytool, "create_new")
        layout.prop(mytool, "calculate_metrics")

        layout.separator()
    
        layout.operator(AddModelButtonOperator.bl_idname, text="Add model from directory")
        layout.operator(DeleteModelButtonOperator.bl_idname, text="Delete selected model")
        layout.operator(InstallLibrariesOperator.bl_idname, text="Install libraries from TXT")   
        
        layout.separator()
       
# end class dope_sheet_options_button

def draw_buttons_in_dope_sheet(self, context):
    layout = self.layout
    layout.separator()  
    layout.operator(dope_sheet_generation_button.bl_idname, text="Generate frames")
    layout.operator(dope_sheet_options_button.bl_idname, text="Generation options")

# end function draw_buttons_in_dope_sheet

generation_window_classes = [InstallLibrariesOperator, AddModelButtonOperator, GenerationButtonOperator, PLUGIN_PT_GenerationPanel, GenerationProperties, DeleteModelButtonOperator]

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
