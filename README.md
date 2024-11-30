# Plugin dla programu Blender do modyfikacji animacji modeli postaci 3D

Struktura projektu
- /interface -> zawiera główny interfejs który muszą zaimplentować modele 
- /src -> zawiera pliki zródłowe, w nim jest plik main_addon_file.py, który należy wczytać jako skrypt
- /models -> będzie zawierał modele których można uzyć, w osobnych folderach o ich nazwach, bezpośrednio w danym folderze powinien być plik nazwa_modelu.py który będzie implementował nasz interfejs
- /sample_bvh_files -> przykładowe pliki motion capture
- /lib -> zawiera pliki binarne, potrzebne dla wtyczki
