import os
import argparse

def parse_args():
    '''' Function enabling command-line arguments'''
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Rename Result Files")

    # Define arguments
    parser.add_argument('-expdirs', '--exp_dirs', nargs='+', type=str, required=False, help="Dirs to rename subdirs")  # Unchanged

    return parser.parse_args()

def remove_spaces_from_folders():
    args = parse_args()
    for EA_dir in args.exp_dirs:
        # Walk through the directory, including subdirectories
        for root, dirs, _ in os.walk(EA_dir, topdown=False):
            for dir_name in dirs:
                # Create full path to the folder
                old_dir_path = os.path.join(root, dir_name)
                # Remove spaces from the folder name
                new_dir_name = dir_name.replace(" ", "")
                new_dir_path = os.path.join(root, new_dir_name)
                # Rename the folder if necessary
                if old_dir_path != new_dir_path:
                    os.rename(old_dir_path, new_dir_path)
                    print(f'Renamed: {old_dir_path} -> {new_dir_path}\n')

if __name__ == '__main__':
    remove_spaces_from_folders()