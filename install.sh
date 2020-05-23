pip install pycddlib

# setup tex fonts
sudo apt install msttcorefonts -qq
python3 -c "import matplotlib.font_manager; del matplotlib.font_manager.weight_dict['roman']; matplotlib.font_manager._rebuild()"
