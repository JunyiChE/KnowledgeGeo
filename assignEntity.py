import pandas as pd
import re
from tqdm import tqdm

# Load the CSV files with Latin encoding
words_to_entities_df = pd.read_csv('words_to_entities.csv', encoding='latin1')
training_list_df = pd.read_csv('test_list.csv', encoding='latin1').head(100)  # Process only the first 100 rows

# Ensure all entities are treated as strings and convert to lowercase
words_to_entities_df['Entity'] = words_to_entities_df['Entity'].fillna('').astype(str).str.lower()

# Function to filter and process tweets
def process_tweets(tweet):
    original_tweets = tweet.split('|||')
    filtered_tweets = []

    # Entity recognition data
    entities = []
    tags = []
    wiki_links = []

    # Filtering tweets
    for t in tqdm(original_tweets, desc="Processing Tweets"):
        try:
            # Check if tweet mentions a user or contains indicative words
            if '@USER_' in t or any(re.search(r'\b' + re.escape(entity) + r'\b', t.lower()) for entity in words_to_entities_df['Entity']):
                # Check for entities in the tweet
                for _, row in tqdm(words_to_entities_df.iterrows(), desc="Checking Entities", total=words_to_entities_df.shape[0]):
                    if re.search(r'\b' + re.escape(row['Entity']) + r'\b', t.lower()):
                        entities.append(row['Entity'])
                        tags.append(row['Tag'])
                        wiki_links.append(row['Wikipedia_Link'])
                filtered_tweets.append(t)
        except Exception as e:
            print(f"An error occurred with tweet: {t}. Error: {e}")
            continue

    filtered_number = len(original_tweets) - len(filtered_tweets)
    return '|||'.join(filtered_tweets), entities, tags, wiki_links, filtered_number

# Apply the function to each row in the training list with tqdm
training_list_df[['Tweet', 'Entities', 'Tags', 'Wiki_Links', 'Filtered_Number']] = list(tqdm(training_list_df.apply(
    lambda row: pd.Series(process_tweets(row['Tweet'])), axis=1), desc="Applying to DataFrame"))

# Display the first few rows of the modified DataFrame
print(training_list_df.head())
