def filter_short_sequences_fasta(input_file, output_file, min_length=16):
    """
    Filters FASTA entries with sequences shorter than a given minimum length.
    
    Args:
        input_file (str): Path to the input FASTA file.
        output_file (str): Path to save the filtered FASTA file.
        min_length (int): Minimum sequence length to keep. Default is 15.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        header = None
        sequence = []
        
        for line in infile:
            line = line.strip()
            if line.startswith('>'):  # Header line
                if header and len(''.join(sequence)) >= min_length:
                    # Write the previous entry if it's valid
                    outfile.write(f"{header}\n{''.join(sequence)}\n")
                # Start a new entry
                header = line
                sequence = []
            else:  # Sequence line
                sequence.append(line)
        
        # Write the last entry if it's valid
        if header and len(''.join(sequence)) >= min_length:
            outfile.write(f"{header}\n{''.join(sequence)}\n")

# Main
input_file = '../all.fasta'  # Replace with your input FASTA file
output_file = 'all_filtered.fasta'  # Replace with your desired output file

filter_short_sequences_fasta(input_file, output_file)

print(f"Filtered FASTA file saved to {output_file}")
