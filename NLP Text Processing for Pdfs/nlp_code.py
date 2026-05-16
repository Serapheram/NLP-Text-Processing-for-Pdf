import PyPDF2
import re
import os
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.io as pio
import webbrowser

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, 'The Red and the Black.pdf')

# Q1(a) It will open the pdf and pull out all the text from every page
with open(pdf_path, 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    total_pages = len(reader.pages)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text()

# just grabbing a small sample to show in the report
sample_text = raw_text[:600]

# Q1(b) It will clean up the text step by step
# first make everything lowercase
text = raw_text.lower()

# remove all numbers using regex
text = re.sub(r'\d+', '', text)

# remove special symbols, keep only letters and spaces
text = re.sub(r'[^a-z\s]', '', text)

# collapse multiple spaces into one
text = re.sub(r'\s+', ' ', text).strip()

# split the text into individual words
tokens = word_tokenize(text)

# remove stop words like "the", "is", "and" etc
stop_words = set(stopwords.words('english'))
stop_count = sum(1 for w in tokens if w in stop_words)
valid_words = [w for w in tokens if w not in stop_words]

# stem each word down to its root form
ps = PorterStemmer()
stemmed = [ps.stem(w) for w in valid_words]

# lemmatize each word to its dictionary form
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w) for w in valid_words]

# Q1(c) It will perform one-hot encoding on a sample of 50 words
sample_words = valid_words[:50]
unique_words = list(set(sample_words))
ohe_matrix = [[1 if word == uw else 0 for uw in unique_words] for word in sample_words]
ohe_df = pd.DataFrame(ohe_matrix, columns=unique_words)

# Q1(d) It will calculate TF-IDF on the full cleaned text, picking top 20 words
sentences = text.split('.')
sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
vectorizer = TfidfVectorizer(max_features=20)
tfidf_matrix = vectorizer.fit_transform(sentences)
feature_names = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# get the average tfidf score per word for the plot
mean_scores = tfidf_df.mean().reset_index()
mean_scores.columns = ['Word', 'TF-IDF Score']

# Q1(d) It will build the scatter plot with plotly
fig = px.scatter(
    mean_scores,
    x='Word',
    y='TF-IDF Score',
    title='TF-IDF Scores of Top 20 Words',
    labels={'Word': 'Words', 'TF-IDF Score': 'TF-IDF Score'},
    color='TF-IDF Score',
    size='TF-IDF Score'
)
fig.update_layout(
    plot_bgcolor='#0f0f0f',
    paper_bgcolor='#0f0f0f',
    font_color='#e0e0e0',
    title_font_size=18
)
plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

# converting dataframes to html tables for the report
ohe_html = ohe_df.to_html(classes='data-table', border=0, max_cols=10)
tfidf_html = tfidf_df.head(10).to_html(classes='data-table', border=0)

# turning the word lists into html chips for display
stemmed_chips = ''.join(f'<span class="word-chip">{w}</span>' for w in stemmed[:15])
lemma_chips = ''.join(f'<span class="word-chip">{w}</span>' for w in lemmatized[:15])
feature_chips = ''.join(f'<span class="word-chip">{w}</span>' for w in feature_names)

# all the data that gets injected into the html template
data = {
    'total_pages': total_pages,
    'raw_text_len': f'{len(raw_text):,}',
    'sample_text': sample_text,
    'total_tokens': f'{len(tokens):,}',
    'stop_count': f'{stop_count:,}',
    'valid_words': f'{len(valid_words):,}',
    'stemmed_chips': stemmed_chips,
    'lemma_chips': lemma_chips,
    'ohe_html': ohe_html,
    'feature_chips': feature_chips,
    'tfidf_html': tfidf_html,
    'plot_html': plot_html,
}

# read the template file and swap in all the real data
template_path = os.path.join(BASE_DIR, 'template.html')
with open(template_path, 'r', encoding='utf-8') as f:
    template = f.read()

for key, value in data.items():
    template = template.replace(f'{{{{{key}}}}}', str(value))

# save the final report and open it in the browser
output_path = os.path.join(BASE_DIR, 'nlp_report.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(template)

print("Done! Opening report in your browser...")
webbrowser.open(f'file:///{output_path}')