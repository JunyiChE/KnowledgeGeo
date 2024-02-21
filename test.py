import spacy

# # Load the English model
# nlp = spacy.load("en_core_web_sm")
#
# # Sample text
# text = "The Amazon Rainforest is a vast tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America."
#
# # Process the text
# doc = nlp(text)
#
# # Iterate through each token in the document
# for token in doc:
#     # Check if the token is a named entity
#     # if token.ent_type_ == "GPE":
#         print(f"Word: '{token.text}', Geographical Entity: '{token.ent_type_}'")

import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Sample text
text = "Apple Inc. is planning to open a new store in San Francisco, United States,America."
text=" Welcome to Ferguson where Americans started waking up to the militarization of their police force.”"

# Process the text
doc = nlp(text)

# Print all named entities and their labels
# for ent in doc.ents:
#     print(ent.text, ent.type_)
for token in doc:
    # Check if the token is a named entity
    # if token.ent_type_ == "GPE":
        print(f"Word: '{token.text}', Geographical Entity: '{token.ent_type_}'")

