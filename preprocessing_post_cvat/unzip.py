import subprocess

unzip_file = "box_datasets/box_dataset_rm_veh.zip"
dest_folder = "box_datasets/box_dataset_rm_veh/val/"

# Unzip a file with pv progress (Python way)
subprocess.run(f'pv {unzip_file} | unzip -d {dest_folder}', shell=True)