#!/bin/bash
blenderPath=$(<blender_executable_path)
cd ..
"$blenderPath" --command "extension build"
cd scripts