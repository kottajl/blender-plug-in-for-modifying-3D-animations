# Plugin dla programu Blender do modyfikacji animacji modeli postaci 3D

Foldery
- /src -> zawiera pliki zródłowe, w nim jest plik main_addon_file.py, który należy wczytać jako skrypt
- /interface -> zawiera główny interfejs który muszą zaimplentować modele 
- /models -> będzie zawierał modele których można uzyć, w osobnych folderach o ich nazwach, bezpośrednio w danym folderze powinien być plik nazwa-modelu.py który będzie implementował nasz interfejs
- /sample_bvh_files -> przykładowe pliki motion capture

Instrukcja
- /models/motion_inbetweening/datasets/lafan1, pliki z https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip odpakowanego
- /models/motion_inbetweening/experiments, pliki z https://github.com/victorqin/motion_inbetweening/releases/download/v1.0.0/pre-treained.zip odpakowanego (2 foldery)     
