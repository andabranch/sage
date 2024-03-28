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

def convert_scientific_notation(file_path):
    pattern = re.compile(r'\(assert \(= Y_(\d+) (-?\d+\.\d+e-?\d+)\)\)')
    matched_lines = 0
    total_lines = 0
    modified_lines = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

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

    with open(file_path, 'w') as file:
        file.writelines(modified_lines)

    print(f"Total lines processed: {total_lines}")
    print(f"Total matched lines: {matched_lines}")
    print("File has been modified and saved to expand scientific notation for Y variables.")



if __name__ == "__main__":
    #pt merge la vnnlib+counterexample si output file smt2
    #merge_files('vnnlib/all-vnnlib/model_64_idx_11985_eps_15.00000.vnnlib', 'vnnlib/alphabeta/3_64_64_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_1024_BN_Dense_43_ep_30_model_64_idx_11985_eps_15.00000.counterexample', 'vnnlib/png/alphabeta/new/model_64_idx_11985_eps_15.00000.smt2')

    #modificat cu adaugare de assert pt valorile X si Y
    #modify_file('vnnlib/png/alphabeta/new/model_64_idx_11985_eps_15.00000.smt2')
    verify_smt2_files('vnnlib/png/alphabeta/new', 'vnnlib/png/alphabeta/new/sat_results')

    
    #unsat_modify_X('vnnlib/png/alphabeta/new/model_64_idx_11985_eps_15.00000.smt2')
    #convert_scientific_notation('vnnlib/png/alphabeta/new/model_64_idx_11985_eps_1.00000.smt2')