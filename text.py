import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.stem import WordNetLemmatizer, PorterStemmer
# Task 1: Language Identification (Basic Identification using characters)
def language_identification(text):
    # Assume language based on common English characters/words
    if any(char.isalpha() for char in text):
        return "English (Assumed)"
    return "Unknown Language"

# Task 2: Tokenization
def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens


# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

# Task 3: Part-of-Speech (POS) Tagging
def pos_tagging(tokens):
    pos_tags = pos_tag(tokens)
    return pos_tags

# Task 4: Chunking (Named Entity Recognition)
def chunking(pos_tags):
    chunks = ne_chunk(pos_tags)
    return chunks

# Task 5: Removing Stop Words
def remove_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

# Task 6: Syntax Parsing (simulated using POS tags)
def syntax_parsing(pos_tags):
    return [(word, tag) for word, tag in pos_tags]

# Task 7: Lemmatization
def lemmatization(tokens):
    lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_words

# Task 8: Stemming
def stemming(tokens):
    stemmed_words = [ps.stem(word) for word in tokens]
    return stemmed_words
# Sample text
text = "John, who is an engineer, went to the market to buy fresh vegetables."
# Process the text
tokens = tokenize_text(text)
pos_tags = pos_tagging(tokens)

# Print results for each task
print("Language Identification:", language_identification(text))
print("Tokens:", tokens)
print("POS Tags:", pos_tags)
print("Chunks (NER):", chunking(pos_tags))
print("Filtered Tokens (Stop Words Removed):", remove_stopwords(tokens))
print("Syntax Parsing (POS):", syntax_parsing(pos_tags))
print("Lemmatized Words:", lemmatization(tokens))
print("Stemmed Words:", stemming(tokens))
