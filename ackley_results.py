import pandas as pd
import os
import matplotlib.pyplot as plt

# Directory containing the Ackley files
directory_path = 'Ackley'

# List all files in the specified directory that match the Ackley txt files
files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.startswith('Ackley') and f.endswith('.txt')]

# Read all files into a single DataFrame
df_list = []
for file in files:
    data = pd.read_csv(file, delim_whitespace=True)
    df_list.append(data)

# Combine all dataframes into a single DataFrame
combined_df = pd.concat(df_list, ignore_index=True)
print(combined_df)

# Calculate average values
average_score = combined_df['Score'].mean()
average_x = combined_df['X'].mean()
average_y = combined_df['Y'].mean()

# Display averages
average_values = {
    "Average Score": average_score,
    "Average X": average_x,
    "Average Y": average_y
}
print(average_values)

# Calculate min and max for X, Y, and Score
min_x = combined_df['X'].min()
max_x = combined_df['X'].max()
min_y = combined_df['Y'].min()
max_y = combined_df['Y'].max()
min_score = combined_df['Score'].min()
max_score = combined_df['Score'].max()

# Display min and max values
min_max_values = {
    "Min X": min_x,
    "Max X": max_x,
    "Min Y": min_y,
    "Max Y": max_y,
    "Min Score": min_score,
    "Max Score": max_score
}

print(min_max_values)


# Plotting distributions of X and Y
plt.figure(figsize=(12, 6))

# Distribution plot for X
plt.subplot(1, 2, 1)
plt.hist(combined_df['X'], bins=20, color='blue', alpha=0.7)
plt.title('Distribution of X')
plt.xlabel('X')
plt.ylabel('Frequency')

# Distribution plot for Y
plt.subplot(1, 2, 2)
plt.hist(combined_df['Y'], bins=20, color='green', alpha=0.7)
plt.title('Distribution of Y')
plt.xlabel('Y')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()