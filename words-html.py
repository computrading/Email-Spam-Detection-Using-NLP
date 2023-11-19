import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from collections import Counter

# Assuming you have a CSV file with 'text' and 'label' columns, where 'label' is 'spam' or 'ham'
# Replace 'your_dataset.csv' with your actual dataset filename

df = pd.read_csv('dataset/spamails.csv')

# Filter spam messages
spam_messages = df[df['CATEGORY'] == 1]['MESSAGE']

# Function to extract text from HTML
def extract_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()

# Tokenize and process HTML content
spam_words = []
for message in spam_messages:
    text = extract_text_from_html(message)
    tokens = word_tokenize(text.lower())
    spam_words.extend(tokens)

# Remove stopwords, including HTML tags
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in spam_words if word.isalnum() and word not in stop_words]

# Count the occurrences of each word
word_counts = Counter(filtered_words)

# Get the most common words
most_common_words = word_counts.most_common(50)  # Change 10 to your desired number

print("Most common words in spam messages:")
for word, count in most_common_words:
    print(f"{word}: {count}")
