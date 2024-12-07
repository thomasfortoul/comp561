def remove_last_n_lines(input_file, output_file, n):
    """
    Removes the last n lines from a file and writes the updated content to an output file.
    
    :param input_file: Path to the input file.
    :param output_file: Path to the output file where the result will be saved.
    :param n: Number of lines to remove from the end.
    """
    try:
        # Read all lines from the input file
        with open(input_file, 'r') as file:
            lines = file.readlines()
        
        # Write back all but the last n lines
        with open(output_file, 'w') as file:
            file.writelines(lines[:-n])  # Exclude the last n lines
        print(f"Successfully removed the last {n} lines.")
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example Usage
if __name__ == "__main__":
    input_file = "data/Sequence/Test_seq.csv"
    output_file = "data/Sequence/Test_seq.csv"
    n = 15000
    
    remove_last_n_lines(input_file, output_file, n)
