import os
import time

# Directory where the files are stored
directory_path = "/home/rehan/Projects/Pytorch_Image_Classification/processed_images"

# Get the current time
current_time = time.time()

# Time limit (1 hour in seconds)
time_limit = 60 * 60  # 1 hour = 60 minutes * 60 seconds

# Loop through all files in the directory
for file_name in os.listdir(directory_path):
    file_path = os.path.join(directory_path, file_name)

    # Check if it's a file (not a directory)
    if os.path.isfile(file_path):
        # Get the last modified time of the file
        file_mod_time = os.path.getmtime(file_path)

        # Check if the file is older than the time limit (1 hour)
        if current_time - file_mod_time > time_limit:
            try:
                os.remove(file_path)  # Delete the file
                print(f"Deleted: {file_name}")
            except Exception as e:
                print(f"Error deleting {file_name}: {e}")
