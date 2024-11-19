import nltk
from nltk.corpus import stopwords,wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.stem import WordNetLemmatizer,PorterStemmer

def lan_ide(text):
    if any(char.isalpha()for char in text):
        return 'en'
    return "Unknown Language";

def tokenize(text):
    tokens=word_tokenize(text)
    return tokens

#download
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')

nltk.download('maxent_ne_chunker')

def pos_tagging(tokens):
    pos_tags=pos_tag(tokens)
    return pos_tags

def chunking (pos_tags):
    chunked=ne_chunk(pos_tags)
    return chunked

def remove_stopword(tokens):
    stop_words=set(stopwords.words('english'))
    filtered_tokens=[word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

def syntax_parsing(pos_Tags):
    return[(word,tag)for word,tag in pos_Tags]

def lemmatization(tokens):
    lemmatizer=WordNetLemmatizer()
    lemmatized_tokens=[lemmatizer.lemmatize(word) for word in tokens]
    return lemmatized_tokens
def stemming(tokens):
    stemmer=PorterStemmer()
    stemmed_tokens=[stemmer.stem(word) for word in tokens]
    return stemmed_tokens
# Sample text
text = "John, who is an engineer, went to the market to buy fresh vegetables."
tokens=tokenize(text)
pos_tags=pos_tag(tokens)
# Print results for each task
print("Language Identification:", lan_ide(text))
print("Tokens:", tokens)
print("Filtered Tokens (Stop Words Removed):", remove_stopword(tokens))
print("Lemmatized Words:", lemmatization(tokens))
print("Stemmed Words:", stemming(tokens))
#POS_TAG USED
print("POS Tags:", pos_tags)
print("Chunks (NER):", chunking(pos_tags))
print("Syntax Parsing (POS):", syntax_parsing(pos_tags))
