import shutil
import os

def move_folder(source_folder, destination_folder):
    """
    Moves a folder (with all subfolders and files) to another directory.

    :param source_folder: Path to the folder you want to move
    :param destination_folder: Path to the target directory where it should be moved
    """
    try:
        # Ensure the source exists
        if not os.path.exists(source_folder):
            print(f"Source folder does not exist: {source_folder}")
            return
        
        # Ensure destination exists (create if not)
        os.makedirs(destination_folder, exist_ok=True)

        # Construct the final destination path
        folder_name = os.path.basename(source_folder.rstrip(os.sep))
        final_destination = os.path.join(destination_folder, folder_name)

        # Move the folder
        shutil.move(source_folder, final_destination)
        print(f"Moved '{source_folder}' → '{final_destination}'")

    except Exception as e:
        print(f"Error moving folder: {e}")

if __name__=='__main__':
    src = "/path/to/folder"
    dst = "/path/to/folder_dest"
    move_folder(src, dst)
