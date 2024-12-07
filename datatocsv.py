import csv


# Documentation comment
# This script converts multiple FASTA files into CSV format, replacing 'NA' values with 0.0
# It also appends a header line with indices (1, 2, 3, ..., N) for the length of the sequences
# To use: Specify the input FASTA files in the 'files' list. The output CSV files will be saved 
# in the same directory with the same name but with a .csv extension.
#
# Example:
# 1. Update the 'files' list with the correct input filenames.
# 2. Run the script to generate the CSV files.

# This script converts a FASTA-like format file into a CSV-like format.
# The input file should contain sequences in the following format:
# >header_line
# sequence_line
#
# For each sequence pair (header and sequence), the script writes a new line in the following format:
# header sequence 0
# where '0' is a placeholder number. This number can be adjusted as needed.
#
# The script reads the input file, processes it line by line, and outputs the result into a new file.
# The output file will have the format:
# >header sequence 0
#
# Usage:
# - Replace 'input_file.txt' with the name of your input file.
# - The script assumes sequences are always paired with headers, i.e., a header line is followed by a sequence line.
# - The output file is written to 'output_file.csv' by default, but this can be customized.

matches = '../randomized_output.fasta'  # replace with your actual file name
non_matches = '../filtered_non_matches.fasta'
matches_filename = 'seq.csv'  # desired output file name
non_matches_filename = 'non_matches.csv'  # desired output file name

with open(matches, 'r') as infile, open(matches_filename, 'w') as outfile:
    lines = infile.readlines()
    for i in range(0, len(lines), 2):  # assuming sequences are always paired with headers
        name = lines[i].strip()  # header line (e.g., >chr11:38337069-38337084_shuf)
        sequence = lines[i + 1].strip()  # sequence line (e.g., TCAGGAGATAGAGACC)
        if name[-1] == '-' or name[-1] == '+':
            outfile.write(f"{name} {sequence} 1\n")  # change '0' to any other number if needed
        
        else:
            outfile.write(f"{name} {sequence} 0\n")  # change '0' to any other number if needed


# with open(non_matches, 'r') as infile, open(non_matches_filename, 'w') as outfile:
#     lines = infile.readlines()
#     for i in range(0, len(lines), 2):  # assuming sequences are always paired with headers
#         name = lines[i].strip()  # header line (e.g., >chr11:38337069-38337084_shuf)
#         sequence = lines[i + 1].strip()  # sequence line (e.g., TCAGGAGATAGAGACC)
#         outfile.write(f"{name} {sequence} 0\n")  # change '0' to any other number if needed


#------

matches = '../randomized_output.fasta'  # replace with your actual file name
# non_matches = '../non_matches.fasta'
# matches_filename = 'matches.csv'  # desired output file name
# non_matches_filename = 'non_matches.csv'  # desired output file name

# List of input file names
files = [
    f'{matches}.EP',
    f'{matches}.Roll',
    f'{matches}.HelT',
    f'{matches}.MGW',
    f'{matches}.ProT'
]

# Function to convert fasta to CSV
def fasta_to_csv(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
        
    # Prepare to write to a CSV file
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header with sequence indices
        csv_writer.writerow([str(i) for i in range(1, 17)])  # Indices for each sequence (1, 2, ..., 16)
        
        # Process each line in the fasta file
        for line in lines:
            # Remove leading/trailing spaces and newline characters
            line = line.strip()
            
            # Skip the header lines (those starting with '>')
            if line.startswith(">"):
                continue
            
            # Replace "NA" with "0.0" and split by commas
            row = line.split(',')
            row = [0.0 if value == "NA" else float(value) for value in row]
            
            # Write the processed row to the CSV
            csv_writer.writerow(row)

# Loop through all the files and process them
for file in files:
    # Define the output CSV file name (same as the input file but with .csv extension)
    output_file = file.replace('.fasta', '.csv')
    fasta_to_csv(file, output_file)

    print(f"Processed {file} into {output_file}")
