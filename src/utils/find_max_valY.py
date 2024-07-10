import os
import re

def find_max_y_values_per_file(directory):
    y_pattern = re.compile(r'\(Y_(\d{1,2}) ([\d\.\-]+)\)')

    for root, _, files in os.walk(directory):
        files = sorted(files)
        for file in files:
            if file.endswith(".out"): 
                file_path = os.path.join(root, file)
                max_y_value = float('-inf')
                max_y_index = -1
                with open(file_path, 'r') as f:
                    content = f.read()
                    matches = y_pattern.findall(content)
                    for match in matches:
                        y_index, y_value = int(match[0]), float(match[1])
                        if 0 <= y_index <= 42 and y_value > max_y_value:
                            max_y_value = y_value
                            max_y_index = y_index
                if max_y_index != -1:
                    print(f"File: {file}")
                    print(f"Maximum Y value: {max_y_value} at Y_{max_y_index}")

directory_path = "output/verification/counterexamples/new_neuralsat"
find_max_y_values_per_file(directory_path)
