import random

def randomize_fasta(input_file, output_file):
    """
    Randomizes the order of sequences in a FASTA file while preserving sequence-header integrity.
    
    Args:
        input_file (str): Path to the input FASTA file.
        output_file (str): Path to save the randomized FASTA file.
    """
    # Read and pair headers with their sequences
    fasta_entries = []
    with open(input_file, 'r') as infile:
        header = None
        sequence = []
        counter = 0
        for line in infile:
            counter += 1
            if counter < 200000:
                continue
            line = line.strip()
            if line.startswith('>'):  # Header line
                if header:  # Save the previous entry
                    fasta_entries.append((header, ''.join(sequence)))
                header = line  # Start a new entry
                sequence = []
            else:  # Sequence line
                sequence.append(line)
        
        # Save the last entry
        if header:
            fasta_entries.append((header, ''.join(sequence)))
    
    # Shuffle the entries
    random.shuffle(fasta_entries)
    
    # Write the randomized entries back to a file
    with open(output_file, 'w') as outfile:
        for header, sequence in fasta_entries:
            outfile.write(f"{header}\n{sequence}\n")

# Main
input_file = '../all.fasta'  # Replace with your input FASTA file
output_file = 'randomized_output.fasta'  # Replace with

randomize_fasta(input_file, output_file)

print(f"Randomized FASTA file saved to {output_file}")