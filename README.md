# Blender software plug-in for modifying 3D character models’ animations

## Installation

It is advised to open Blender from **terminal**, at least during installation to see progress (Blender appears not responding). To do this simply put path to your Blender excecutable in console and press **ENTER**.

### Blender version 4.2.0 and above

To install the plug-in:

1. Download **ZIP** file from this repository.

2. Open **Blender**.

3. Go to **Edit** (in top left corner) **> Preferences**.

4. On the list from left side, select **Add-ons**.

5. Press the arrow in the top right corner and select **Install from Disk**.

6. Select downloaded **ZIP** file.

7. Confirm the selection with the **Install from Disk** button. 

### Older Blender versions

Using addon on these versions is not recommended. Currently there is no option to install addon from **ZIP** file. Only option is to run code for single use:

1. Download whole repository.

2. Open **Blender**.

3. Go to **Scripting** tab (in top section)

4. Click **Open** button in the middle.

5. Select **main_addon_file.py** from **src** folder in repository.

6. Confirm the selection with the **Open Text** button. 

7. Press **Run Script** (with arrow icon) button.

### While installing

The plug-in may take some time to download the relevant dependencies and files from external repositories. This proces may take a few or more minutes depending on network speed. Progress can be checked from the **terminal**.

## How to use

### Opening the tool

To start using the tool:

1. Go to the **Layout** tab (on the top bar).

2. Display the panels with additional options to the right of the view (using the **N** key).

3. In the bar that appears, select the **AI Animation Bridge** tab. You should see a panel with various options and a **Generate frames** button at the bottom.

It is also possible to use the plugin via the context menu on the timeline. To do this:

1. Make sure that you are in the **Layout** tab and that you can see the expanded timeline view at the bottom of the window. 
    - To expand the timeline view, hover the cursor over the space between the scene and the top edge of the timeline bar. Then press the **left mouse button** and move up the screen as desired.

2. Place the cursor inside the timeline view and open the context menu (using the **right mouse button**). You should see additional items such as **Generate frames**, **Generation options**, etc.

### Generating animation

First generation can take a little longer (max 1 minute). To generate frames:

1. While in the **Layout** tab, select the desired 3D object, you can import sample animation file from our tool.

2. In the top left corner, change the mode from Object mode to **Pose mode**.

3. Select all bones of the skeleton (most conveniently using the **left mouse button** on the scene and "drawing" square on whole skeleton).

4. Expand the left side of timeline in the bottom with hovering on the edge of screen. You should see exactly 3 lines of frames (for **Summary** and 2 for object you have selected, expand **Summary** if needed and close lines for single bones)

5. Open the tool options
    - If you are using the context menu, select **Generation options**.

6. Select the AI model from the list and then configure the model and generation options.

7. Select the start and end frames of the animation to be generated

    - If you are using the context menu:

    1. Deselect all frames by clicking on timeline but not on frames.

    2. Select 2 frames holding down the **Shift** key, on the timeline. It will automatically select points on all 3 lines of frames.

8. Press the **Generate frames** button.

## Project structure

- **/interface** -> contains the main interface that models must implement

- **/lib** -> contains binary files required for the plugin

- **/models** -> contains pre-prepared AI models that can be used, each AI model implements our interface 

- **/sample_bvh_files** -> sample motion capture files

- **/scripts** -> contains scripts including those that pack plug-in and test code

- **/src** -> contains source files, including main_addon_file.py

- **/blender_manifest.toml** -> file required for packing addon, contains information about it

## Running tests

To start testing code:

1. Install the plugin and run it in **Blender** at least once to install necessary libraries.

2. Install **pytest** and **pytest-blender** libraries in your local Python installation (not that of Blender).

3. Set environment variable **AI_ADDON_BLENDER_EXEC** as path to your Blender executable file.

4. Change directory to **scripts** and use relevant file.

## Packing addon

To pack plug-in, Blender version 4.2.0 or above is required. To make **ZIP** file:

1. Set environment variable **AI_ADDON_BLENDER_EXEC** as path to your Blender executable file.

2. Change directory to **scripts** and use relevant file.
