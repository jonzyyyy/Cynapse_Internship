import os
import re


def check_and_delete_files(base_directory):
    img_rm_count = 0
    txt_rm_count = 0
    labels_dir = os.path.join(base_directory, "labels")
    images_dir = os.path.join(base_directory, "images")
    
    if not os.path.isdir(labels_dir) or not os.path.isdir(images_dir):
        print("Labels or images directory not found.")
        return
    
    for file_name in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, file_name)
        image_path = os.path.join(images_dir, os.path.splitext(file_name)[0] + ".jpg")  # Assuming images are .jpg
        
        try:
            with open(label_path, 'r', encoding='utf-8') as file:
                for line in file:
                    tokens = line.strip().split()
                    if len(tokens) != 5 or not tokens[0].isdigit():  # Ensure first value is an integer, followed by 4 float values
                        print(f"File '{label_path}' contains an invalid line format. Deleting...")
                        os.remove(label_path)
                        txt_rm_count += 1
                        if os.path.exists(image_path):
                            os.remove(image_path)
                            img_rm_count += 1
                            print(f"Deleted corresponding image file: {image_path}")
                        break
                    
                    float_values = tokens[1:]
                    if not all(re.match(r'[-+]?[0-9]*\.?[0-9]+', value) for value in float_values):
                        print(f"File '{label_path}' contains an invalid floating-point format. Deleting...")
                        os.remove(label_path)
                        txt_rm_count += 1
                        if os.path.exists(image_path):
                            img_rm_count += 1
                            os.remove(image_path)
                            print(f"Deleted corresponding image file: {image_path}")
                        break
        except FileNotFoundError:
            print(f"File '{label_path}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
    print(f"Finished checking and deleting files in '{base_directory}'. Removed {txt_rm_count} .txt files and {img_rm_count} .jpg files.")

# Example usage
base_directory = "scooter_datasets/mosiac_scooter_datasets/raw_dataset/train/"  # Replace with your actual base directory
check_and_delete_files(base_directory)
