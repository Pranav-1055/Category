import pandas as pd

# Load your dataset
df = pd.read_csv('/content/balanced_sample.csv')

# Define a function to count words in a string
def word_count(text):
    return len(str(text).split())

# Filter rows where word count in 'text' is >= 50
df_filtered = df[df['text'].apply(word_count) >= 50]

# Optional: save filtered dataframe back to a CSV
df_filtered.to_csv('/content/balanced_sample_filtered.csv', index=False)

print(f"Filtered dataset size: {len(df_filtered)} rows")
