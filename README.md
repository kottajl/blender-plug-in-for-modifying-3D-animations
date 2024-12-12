# Blender software plug-in for modifying 3D character models’ animations

Project Structure
- /interface -> contains the main interface that models must implement
- /src -> contains source files, including main_addon_file.py, which must be loaded as a script
- /models -> contains pre-prepared models that can be used. Each model has its own folder, with a file named model_name.py located directly within the folder, which implements our interface 
- /sample_bvh_files -> sample motion capture files
- /lib -> contains binary files required for the plugin
