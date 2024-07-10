import re
import subprocess
import os

def modify_file(file_path):

    pattern = r'^\s*\(?\(?([XY])_(\d+)\s+(-?[\d\.]+(?:e-?\d+)?)\)?\)?\s*$'
    
    matched_lines = 0
    total_lines = 0
    modified_lines = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        total_lines += 1
        match = re.match(pattern, line)
        if match:
            prefix, number, value = match.groups()
            number = int(number)
            modified_line = f"(assert (= {prefix}_{number} {value}))\n"
            modified_lines.append(modified_line)
            matched_lines += 1
        else:
            modified_lines.append(line)

    with open(file_path, 'w') as file:
        file.writelines(modified_lines)

    print(f"Total lines processed: {total_lines}")
    print(f"Total matched lines: {matched_lines}")
    print("File has been modified and saved.")



def merge_files(file_location1, file_location2, file_output):
    with open(file_location1, 'r') as file1, open(file_location2, 'r') as file2:
        content1 = file1.readlines()
        content2 = file2.readlines()
        if content2 and content2[0].strip().startswith("sat"):
            content2 = content2[1:]
    merged_content = content1 + content2

    with open(file_output, 'w') as outfile:
        outfile.writelines(merged_content)
        outfile.write("\n(check-sat)")
    outfile.close()
    print(f"Files have been merged and saved to {file_output}.")


def verify_smt2_files(source_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    smt2_files = [f for f in os.listdir(source_directory) if f.endswith(".smt2")]
    smt2_files.sort()

    for filename in smt2_files:
        source_file_path = os.path.join(source_directory, filename)
        print(f"Verifying {source_file_path}...")

        output_file_path = os.path.join(output_directory, f"{filename}.sat")
        command = f"z3 {source_file_path} > {output_file_path}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.stderr:
            print(f"Error processing {filename}: {result.stderr}")
        else:
            print(f"Result saved to {output_file_path}")

        
        
def unsat_modify_X(file_path):
    pattern = r'^\(assert \(= [X]_'
    modified_lines = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if re.match(pattern, line):
            modified_lines.append(line)
            modified_lines.append("(push)\n")
            modified_lines.append("(check-sat)\n")
            modified_lines.append("(pop)\n")
        else:
            modified_lines.append(line)
        

    with open(file_path, 'w') as file:
        file.writelines(modified_lines)
    
    print(f"File {file_path} has been modified with incremental checks.")
    


import re

def convert_scientific_notation(directory):
    pattern = re.compile(r'\(assert \(= Y_(\d+) (-?\d+\.\d+e-?\d+)\)\)')
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            matched_lines = 0
            total_lines = 0
            modified_lines = []

            with open(file_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                total_lines += 1
                match = pattern.search(line)
                if match:
                    matched_lines += 1
                    y_index, sci_value = match.groups()
                    value = float(sci_value)
                    if "e" in sci_value or "E" in sci_value:
                        exponent = int(sci_value.split('e')[-1])
                        decimal_value = format(value, f'.{abs(exponent)+1}f').rstrip('0').rstrip('.')
                    else:
                        decimal_value = sci_value
                    modified_line = f"(assert (= Y_{y_index} {decimal_value}))\n"
                    modified_lines.append(modified_line)
                else:
                    modified_lines.append(line)

            with open(file_path, 'w') as f:
                f.writelines(modified_lines)

            print(f"Processed file: {file_path}")
            print(f"Total lines processed: {total_lines}")
            print(f"Total matched lines: {matched_lines}")
            print("File has been modified and saved to expand scientific notation for Y variables.")



if __name__ == "__main__":
    # 1) put together the vnnlib file and counterexample file in the new smt2 file
    #merge_files('properties/vnnlib/model_64_idx_11985_eps_15.00000.vnnlib', 'output/verification/counterexamples/neuralsat/3_64_64_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_1024_BN_Dense_43_ep_30_model_64_idx_11985_eps_15.00000.counterexample', 'output/verification/testing_counterexample_quality/neuralsat/smt2/model_64_idx_11985_eps_15.00000.smt2')

    # 2) modify withh assert for X and Y values because counterexample file is missing that
    #modify_file('output/verification/testing_counterexample_quality/neuralsat/smt2/model_64_idx_11985_eps_15.00000.smt2')
    # go in folder and create in sat_results the output given by z3
    #verify_smt2_files('output/verification/testing_counterexample_quality/neuralsat/smt2', folder_paths)
    
    # 3) if tool is alphabeta then it converts the scientific notation e
    folder_paths=[
        'output/verification/testing_counterexample_quality/new_alphabeta/smt2',
        'output/verification/testing_counterexample_quality/alpha-beta-crown/smt2',
        'output/verification/testing_counterexample_quality/pyrat/smt2' ]
    
    path = 'output/verification/testing_counterexample_quality/pyrat/smt2'
    
    # for path in folder_paths:
    #     if path in folder_paths:
    #         convert_scientific_notation(path)
        
    # 4) the z3 call
    verify_smt2_files(path, 'output/verification/testing_counterexample_quality/pyrat/sat_results')

    
    # 5) it's showing us where the first X or Y's value is giving unsatisfiable
    #unsat_modify_X('output/verification/testing_counterexample_quality/new_neuralsat/smt2/model_30_idx_7040_eps_1.00000.smt2')
    #convert_scientific_notation('output/verification/testing_counterexample_quality/new_alphabeta/smt2/model_30_idx_7040_eps_1.00000.smt2')