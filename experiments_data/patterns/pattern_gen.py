import os
import random

file_size = 1000
fixed_letter = 'a'
letter_pattern_3 = 'abc'

pattern_folder = 'experiments_data/patterns'
random_file_path = 'experiments_data/patterns/Pattern3R'
fixed_file_path = 'experiments_data/patterns/Pattern3F'
letter_pattern_3_file_path = 'experiments_data/patterns/Pattern3P3'

#create a file with random letters
with open(random_file_path, 'w') as f:
    for i in range(file_size):
        f.write(chr(random.randint(0, 255)))
        
with open(fixed_file_path, 'w') as f:
    for i in range(file_size):
        f.write(fixed_letter)
        
with open(letter_pattern_3_file_path, 'w') as f:
    for i in range(file_size):
        f.write(letter_pattern_3[i % 3])