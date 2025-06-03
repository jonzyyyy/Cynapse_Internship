import os

"""
Processes a folder of txt files and modify the first column (class id's) based on the given mapping in "modifications"

Parameters:
    -----------
    input_folder : Path to the folder containing the original text files to be modified 
    output_folder : Path to the folder where modified files will be saved 
    modifications : Specify to replace the original class id value to the desired class id value 

Functionality: 
    ------------
    1. Process:
        - Reads the input file line by line.
        - Checks if the first column of each line matches any target values in modifications.
        - If a match is found, it replaces the target value with new_value and updates the line in the file.
        - Logs any changes by line number, original value, and new value.
        - Saves the modified lines to the output file.
"""

input_folder = "/path/to/orginal/txt/folder" # Replace with original txt folder
output_folder = "/path/to/output/folder" # Replace with the folder you would like to create 

modifications = [
    (0, 1),  # Replace '0' in the first column with '1'
    (1, 2),  # Replace '1' in the first column with '2'
]

def modify_file(input_file, output_file, modifications):
    with open(input_file, "r") as file:
        lines = file.readlines()

    changes = [] 

    for i, line in enumerate(lines):
        line_parts = line.split()
        
        # Check if the first column matches any target value in modifications
        for target_value, new_value in modifications:
            if line_parts[0] == str(target_value):  # Match the first column
                # Log 
                changes.append((i + 1, line_parts[0], new_value))  
    
                # Apply 
                line_parts[0] = str(new_value)  # Replace with new value
                lines[i] = " ".join(line_parts) + "\n"  # Update the line in lines list
                break  

    with open(output_file, "w") as file:
        file.writelines(lines)

    if changes:
        print(f"Modifications in file '{input_file}':")
        for line_num, old_value, new_value in changes:
            print(f" - Line {line_num}: Changed first column from {old_value} to {new_value}")
    else:
        print(f"No modifications made in file '{input_file}'.")

def process_folder(input_folder, output_folder, modifications):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            modify_file(input_file, output_file, modifications)


process_folder(input_folder, output_folder, modifications)
    

