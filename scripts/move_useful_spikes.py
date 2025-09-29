import os
import shutil

def copy_files_with_index(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for filename in os.listdir(src_folder):
        src_path = os.path.join(src_folder, filename)
        if os.path.isfile(src_path):
            name, ext = os.path.splitext(filename)
            if "(" in name:
                name = name[:-3]
            
            index = 0
            while True:
                if index >= 10:
                    print(f"Warning: More than 10 duplicates for file {filename}. Stopping further copies.")
                    break
                new_filename = f"{name}_{index}{ext}"
                dst_path = os.path.join(dst_folder, new_filename)
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    break
                index += 1

            if index < 10:
                print(f"File {filename} has now index: {index}")

if __name__ == "__main__":
    src_folder = "/home/lkolmar/Documents/metavision/recordings/tmp/"
    dst_folder = "/home/lkolmar/Documents/metavision/recordings/processed/"
    copy_files_with_index(src_folder, dst_folder)