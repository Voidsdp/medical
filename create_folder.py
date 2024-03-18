import os

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f" '{folder_path}' create successfully")
    else:
        print(f"'{folder_path}' is exist")


