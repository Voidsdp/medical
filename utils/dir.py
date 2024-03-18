import os
from print import color_str

def check_dir(folder_path,create_path=False,alert=False):
    if not os.path.exists(folder_path):
       if create_path:
          os.makedirs(folder_path)
        
       assert not alert, color_str('No such file or directory: {}, you should build it first.'.format(folder_path),'red')
