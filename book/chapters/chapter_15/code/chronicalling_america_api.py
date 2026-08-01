import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

# Step 1: Data Collection from Chronicling America API
BASE_URL = "https://chroniclingamerica.loc.gov/newspapers.json"
response = requests.get(BASE_URL)
data = response.json()

# Extract relevant fields
newspapers = []
for paper in data['newspapers']:
    newspapers.append({
        'title': paper.get('title', ''),
        'state': paper.get('state', ''),
        'lccn': paper.get('lccn', ''),
        'start_year': int(paper.get('start_year', '0')) if paper.get('start_year') else None,
        'end_year': int(paper.get('end_year', '0')) if paper.get('end_year') else None,
        'language': paper.get('language', '')
    })

# Convert to DataFrame
df = pd.DataFrame(newspapers)

# Step 2: Data Cleaning and Preprocessing
# Remove entries with missing years
df = df.dropna(subset=['start_year', 'end_year'])
df['duration'] = df['end_year'] - df['start_year']

# Step 3: Representativeness Analysis
# Count newspapers per state
df_state_counts = df['state'].value_counts().reset_index()
df_state_counts.columns = ['state', 'count']

# Step 4: Bias Measurement using Jensen-Shannon Divergence
# Simulated historical distribution (assumed from historical sources)
historical_distribution = df_state_counts.set_index('state')['count'] / df_state_counts['count'].sum()

# Normalize digitized distribution
digitized_distribution = df_state_counts.set_index('state')['count'] / df_state_counts['count'].sum()

# Jensen-Shannon Divergence
jsd_score = jensenshannon(historical_distribution, digitized_distribution)
print(f'Jensen-Shannon Divergence Score: {jsd_score}')

# Step 5: Visualization
plt.figure(figsize=(12,6))
plt.bar(df_state_counts['state'], df_state_counts['count'], color='steelblue')
plt.xticks(rotation=90)
plt.title("Distribution of Digitized Newspapers by State")
plt.xlabel("State")
plt.ylabel("Number of Newspapers")
plt.tight_layout()
plt.savefig("newspaper_distribution.png")
plt.show()

# Trend Analysis Over Time
df_yearly = df.groupby('start_year').size().reset_index(name='count')
plt.figure(figsize=(12,6))
plt.plot(df_yearly['start_year'], df_yearly['count'], marker='o', linestyle='-', color='steelblue')
plt.title("Number of Newspapers Over Time")
plt.xlabel("Year")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("newspaper_trends.png")
plt.show()





import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import re

# Step 1: Fetch US Senate Speeches from Congressional Record API
API_KEY = "YOUR_PROPUBLICA_API_KEY"
BASE_URL = "https://api.propublica.org/congress/v1/statements/latest.json"
headers = {"X-API-Key": API_KEY}
response = requests.get(BASE_URL, headers=headers)
data = response.json()

# Extract relevant fields
speeches = []
for statement in data['results']:
    speeches.append({
        'senator': statement.get('name', ''),
        'party': statement.get('party', ''),
        'state': statement.get('state', ''),
        'date': statement.get('date', ''),
        'text': statement.get('text', '')
    })

# Convert to DataFrame
df = pd.DataFrame(speeches)

# Step 2: Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

df['clean_text'] = df['text'].apply(preprocess_text)

# Step 3: Compute TF-IDF for N-grams
vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words='english')  # Use bigrams and trigrams
X = vectorizer.fit_transform(df['clean_text'])

tfidf_df = pd.DataFrame(X.toarray(), index=df['senator'], columns=vectorizer.get_feature_names_out())

# Step 4: Compute Cosine Similarity Within Parties
similarity_matrix = cosine_similarity(X)
similarity_df = pd.DataFrame(similarity_matrix, index=df['senator'], columns=df['senator'])

# Step 5: Visualizing Similarity Within Parties
plt.figure(figsize=(12, 8))
im = plt.imshow(similarity_df.values, cmap='coolwarm', aspect='auto')
plt.colorbar(im)
plt.xticks(range(len(similarity_df.columns)), similarity_df.columns, rotation=90)
plt.yticks(range(len(similarity_df.index)), similarity_df.index)
plt.title("Senator Rhetoric Similarity Based on TF-IDF N-grams")
plt.tight_layout()
plt.savefig("senate_rhetoric_similarity.png")
plt.show()

# Step 6: Aggregate by Party
df_party_avg = df.groupby('party')['clean_text'].apply(lambda x: ' '.join(x)).reset_index()
party_tfidf = vectorizer.fit_transform(df_party_avg['clean_text'])
party_similarity = cosine_similarity(party_tfidf)
party_sim_df = pd.DataFrame(party_similarity, index=df_party_avg['party'], columns=df_party_avg['party'])

# Plot Party Similarity
plt.figure(figsize=(8, 6))
im = plt.imshow(party_sim_df.values, cmap='coolwarm', aspect='auto')
plt.colorbar(im)
# Add annotations
for i in range(len(party_sim_df.index)):
    for j in range(len(party_sim_df.columns)):
        plt.text(j, i, f'{party_sim_df.iloc[i, j]:.2f}', 
                ha='center', va='center', color='white' if party_sim_df.iloc[i, j] < 0.5 else 'black')
plt.xticks(range(len(party_sim_df.columns)), party_sim_df.columns)
plt.yticks(range(len(party_sim_df.index)), party_sim_df.index)
plt.title("Intra-Party Rhetoric Similarity")
plt.tight_layout()
plt.savefig("party_rhetoric_similarity.png")
plt.show()

