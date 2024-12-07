def split_data(input_file, train_file, test_file, num_train_lines):
    # Read all lines from the input file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Split the lines into training and testing sets
    train_lines = lines[:num_train_lines]
    test_lines = lines[num_train_lines:num_train_lines+num_test_lines]

    # Write the training set to the train file
    with open(train_file, 'w') as file:
        file.writelines(train_lines)

    # Write the testing set to the test file
    with open(test_file, 'w') as file:
        file.writelines(test_lines)

# Define the number of training lines and testing lines
num_train_lines = 70000
num_test_lines = 25000

# Split output.txt into Train_seq.txt and Test_seq.txt
split_data('seq.csv', 'Train_seq.csv', 'Test_seq.csv', num_train_lines)

# List of DNA file types
matches = '../randomized_output.csv'
files = [
    f'{matches}.EP',
    f'{matches}.Roll',
    f'{matches}.HelT',
    f'{matches}.MGW',
    f'{matches}.ProT'
]

# Split each of the DNA files into Train and Test files
for i in range(len(files)):
    extensions = ["EP", "Roll", "HelT", "MGW", "ProT"]

    train_file = f'Train_{extensions[i]}.csv'
    test_file = f'Test_{extensions[i]}.csv'

    split_data(files[i], train_file, test_file, num_train_lines)
