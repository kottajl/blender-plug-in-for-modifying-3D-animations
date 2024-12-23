#!/bin/bash

if [[ -n "$AI_ADDON_BLENDER_EXEC" ]]; then
    cd .. || exit 1
    "$AI_ADDON_BLENDER_EXEC" --command extension build
    echo "Packaging done."
    cd scripts || exit 1
else
    echo "Environment variable AI_ADDON_BLENDER_EXEC is not set."
fi
