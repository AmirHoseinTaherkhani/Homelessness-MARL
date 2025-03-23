import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

nltk.download('punkt_tab')
# Load data
df = pd.read_json("negotiation_logs.json", lines=True)

# NLP preprocessing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return " ".join(tokens)

df['processed_proposal'] = df['proposal'].apply(preprocess_text)

# Create word clouds
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
agents = ['law_enforcement', 'shelter_services', 'city_government', 'residents']
colors = ['Blues', 'Greens', 'Purples', 'Oranges']
for ax, agent, cmap in zip(axes.flatten(), agents, colors):
    text = " ".join(df[df['agent'] == agent]['processed_proposal'])
    wordcloud = WordCloud(width=400, height=300, background_color='white', colormap=cmap).generate(text)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(agent.replace('_', ' ').title(),  fontsize=18)
    ax.axis('off')
plt.suptitle("Voices of Negotiation: What Each Agent Cares About Most in AI-Driven Policy Talks", fontsize=22)
plt.savefig("vis/word_cloud.png", dpi = 500)
plt.show()