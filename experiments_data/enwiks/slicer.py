input_file = "experiments_data/enwiks/enwik4"  
output_file = "experiments_data/enwiks/enwik3" 



with open(input_file, "rb") as f:
    content = f.read()

new_length = len(content) // 10

first_part = content[:new_length]

with open(output_file, "wb") as f:
    f.write(first_part)