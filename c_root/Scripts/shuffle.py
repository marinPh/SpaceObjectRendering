import numpy as np

# Create an array with numbers from 0 to 99
numbers = np.arange(100)

# Shuffle the array
np.random.shuffle(numbers)

# Generate the file content
shuffled_numbers = "\n".join(f"{n:04d}.png" for n in numbers)

# Save to a text file
with open("shuffled_numbers.txt", "w") as file:
    file.write(shuffled_numbers)