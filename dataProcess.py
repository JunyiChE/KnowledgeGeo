import pandas as pd
import ast
# def extract_entities(data):
#       #   specific_row = data.iloc[14]
#       #   original_word_list = specific_row['word_list']
#       #   original_tag_list = specific_row['tag_list']
#       #   original_wiki_list = specific_row['wiki_list']
#       #
#       #   # Displaying the original information
#       #   original_information = {
#       #       "Word List": original_word_list,
#       #       "Tag List": original_tag_list,
#       #       "Wiki List": original_wiki_list
#       # }
#     #
#         # Extracting entities based on the provided logic
#         extracted_data = []
#         # for i in range(len(original_word_list)):
#             # words = original_word_list[i]
#             # tags = original_tag_list[i] if i < len(original_tag_list) else []
#             # wiki = original_wiki_list[i] if i < len(original_wiki_list) else 0
#             #
#             # if words and wiki != 0:  # Check if words is not empty and wiki is not 0
#             #     entity = ' '.join(words)  # Join words in the group
#             #     tag_str = ' '.join(tags) if tags else 'None'  # Join tags or label as 'None' if empty
#             #     extracted_data.append((entity, tag_str, wiki))
#         #     if original_word_list[i]:  # Check if the word list at index i is not empty
#         #         words = ''.join(original_word_list[i])  # Join words in the group
#         #         tags = ''.join(original_tag_list[i]) if i < len(original_tag_list) and original_tag_list[i] else 'None'
#         #         wiki = original_wiki_list[i] if i < len(original_wiki_list) and original_wiki_list[i] != 0 else 'None'
#         #         extracted_data.append((words, tags, wiki))
#         # extracted_df = pd.DataFrame(extracted_data, columns=['Entity', 'Tag', 'Wikipedia_Link'])
#
#         print(f'original_information: {original_information}')
#         print(f'extracted_df\n{extracted_df}')
    # extracted_data = []
    #
    # for _, row in data.iterrows():
    #     for i in range(len(row['word_list'])):
    #         words = row['word_list'][i]
    #         tags = row['tag_list'][i] if i < len(row['tag_list']) else []
    #         wiki = row['wiki_list'][i] if i < len(row['wiki_list']) else 0
    #
    #         if words and wiki != 0:  # Check if words is not empty and wiki is not 0
    #             entity = ' '.join(words)  # Join words in the group
    #             tag_str = ' '.join(tags) if tags else 'None'  # Join tags or label as 'None' if empty
    #             extracted_data.append((entity, tag_str, wiki))
    #
    # return pd.DataFrame(extracted_data, columns=['Entity', 'Tag', 'Wikipedia_Link'])
def extract_entities(data):
    # specific_row = data.iloc[48]
    # original_word_list = eval(specific_row['word_list'])
    # original_tag_list = eval(specific_row['tag_list'])
    # original_wiki_list = eval(specific_row['wiki_list'])
    #
    # # Displaying the original information
    # original_information = {
    #     "Word List": original_word_list,
    #     "Tag List": original_tag_list,
    #     "Wiki List": original_wiki_list
    # }
    #
    # # Process and aggregate the word, tag, and wiki information
    # # aggregated_words = [''.join(word_group) for word_group in original_word_list if word_group]
    # # aggregated_tags = [''.join(tag_group) for tag_group in original_tag_list if tag_group]
    # # filtered_wiki_links = [wiki_link for wiki_link in original_wiki_list if isinstance(wiki_link, str)]
    # #
    # # # Combine the aggregated data into a DataFrame
    # # extracted_data = list(zip(aggregated_words, aggregated_tags, filtered_wiki_links))
    # # extracted_df = pd.DataFrame(extracted_data, columns=['Entity', 'Tag', 'Wikipedia_Link'])
    # extracted_data = []
    # for i, word_group in enumerate(original_word_list):
    #     if word_group:  # Check if the word group is not empty
    #         word = ' '.join(word_group)
    #         tag = ' '.join(original_tag_list[i]) if i < len(original_tag_list) and original_tag_list[i] else 'None'
    #         wiki_link = original_wiki_list[i] if i < len(original_wiki_list) and isinstance(original_wiki_list[i],
    #                                                                                         str) else 'None'
    #         extracted_data.append((word, tag, wiki_link))
    # extracted_df = pd.DataFrame(extracted_data, columns=['Entity', 'Tag', 'Wikipedia_Link'])
    extracted_data = []

    for _, row in data.iterrows():
        # Safely evaluate strings as lists for each row
        original_word_list = ast.literal_eval(row['word_list'])
        original_tag_list = ast.literal_eval(row['tag_list'])
        original_wiki_list = ast.literal_eval(row['wiki_list'])

        for i, word_group in enumerate(original_word_list):
            if word_group:  # Check if the word group is not empty
                word = ' '.join(word_group)
                tag = ' '.join(original_tag_list[i]) if i < len(original_tag_list) and original_tag_list[i] else 'None'
                wiki_link = original_wiki_list[i] if i < len(original_wiki_list) and original_wiki_list[
                    i] != 0 else 'None'
                if not (word.lower() == 'lol' and wiki_link == 'None'):
                 extracted_data.append((word, tag, wiki_link))
    # extracted_df = pd.DataFrame(extracted_data, columns=['Entity', 'Tag', 'Wikipedia_Link'])
    # filtered_df = extracted_df[
    #     ~((extracted_df['Entity'].str.lower() == 'lol') & (extracted_df['Wikipedia_Link'] == 'None'))]
    # return filtered_df
    return pd.DataFrame(extracted_data, columns=['Entity', 'Tag', 'Wikipedia_Link'])

    # print(f'original_information: {original_information}')
    # print(f'extracted_df:\n{extracted_df}')

    # return extracted_df




# Example usage
# Load your data
# data = pd.read_csv('Tweeki_data_0.csv')
# List of CSV file names to be read, except 'Tweeki_data_2.csv'
csv_files = [f'Tweeki_data_{i}.csv' for i in range(10) if i != 2]

# Read and concatenate all CSV files into a single DataFrame
data = pd.concat([pd.read_csv(file) for file in csv_files])

# Extract the entities
extracted_df =\
extract_entities(data)

# Save the extracted data to a new CSV file
extracted_df.to_csv('words_to_entities.csv', index=False)
