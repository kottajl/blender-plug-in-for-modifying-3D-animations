#!/bin/bash
blenderPath=$(<blender_executable_path)
cd ../src/test
pytest --blender-executable "$blenderPath"
cd ../../scripts