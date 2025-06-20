# app/sentiment_logic.py

from googleapiclient.discovery import build
import pandas as pd
from langdetect import detect, LangDetectException
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from textblob import TextBlob
import seaborn as sns
import os

# Téléchargement NLTK (à ne faire qu’une fois)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Clé API YouTube (⚠️ à sécuriser en .env pour production)
api_key = "AIzaSyBZQ-kNIZ0EpN0ApWFufLg6lBAqH6iTgh0"
youtube = build("youtube", "v3", developerKey=api_key)
lemmatizer = WordNetLemmatizer()

def get_video_comments(video_id):
    comments = []
    results = youtube.commentThreads().list(
        part="snippet", videoId=video_id, textFormat="plainText", maxResults=100
    ).execute()

    while results:
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        if 'nextPageToken' in results:
            results = youtube.commentThreads().list(
                part="snippet", videoId=video_id,
                textFormat="plainText", pageToken=results['nextPageToken'], maxResults=100
            ).execute()
        else:
            break
    return comments

def filter_comments_in_english(comments):
    english_comments = []
    for comment in comments:
        try:
            if len(comment.strip()) > 3 and detect(comment) == 'en':
                english_comments.append(comment)
        except LangDetectException:
            continue
    return english_comments

def preprocess_comment(comment):
    comment = comment.lower()
    comment = re.sub(r'http\S+|www\S+', '', comment)
    comment = re.sub(r'@\w+|#\w+', '', comment)
    comment = re.sub(r'[^a-z\s]', '', comment)
    words = nltk.word_tokenize(comment)
    words = [word for word in words if word not in stopwords.words('english')]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def analyse_video(video_id):
    comments = get_video_comments(video_id)
    english_comments = filter_comments_in_english(comments)
    cleaned_comments = [preprocess_comment(c) for c in english_comments]

    if len(cleaned_comments) == 0:
        return "Aucun commentaire anglais n'a été trouvé sur cette vidéo."

    df = pd.DataFrame({"video_id": video_id, "comment": cleaned_comments})

    # Sauvegarde CSV dans /data/
    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "english_comments_cleaned.csv")
    df.to_csv(csv_path, index=False)

    # Nuage de mots (optionnel à afficher)
    all_words = ' '.join(df['comment']).split()
    word_freq = Counter(all_words)
    top_words = word_freq.most_common(10)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['comment']))
    os.makedirs("images", exist_ok=True)
    wordcloud_path = os.path.join("images", f"wordcloud_{video_id}.png")
    wordcloud.to_file(wordcloud_path)

    # Labellisation
    df['label'] = df['comment'].apply(lambda x: 'positif' if TextBlob(x).sentiment.polarity > 0 else ('négatif' if TextBlob(x).sentiment.polarity < 0 else 'neutre'))

    # ML - Classification
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['comment'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)

    polarities = [TextBlob(c).sentiment.polarity for c in df['comment']]
    if polarities:
        score = sum(polarities) / len(polarities)
    else:
        score = 0

    nb_positif = (df['label'] == 'positif').sum()
    nb_negatif = (df['label'] == 'négatif').sum()
    nb_neutre = (df['label'] == 'neutre').sum()
    longueur_moyenne = df['comment'].apply(len).mean()
    commentaire_le_plus_long = df['comment'].iloc[df['comment'].apply(len).idxmax()]
    commentaire_le_plus_court = df['comment'].iloc[df['comment'].apply(len).idxmin()]
    mots_uniques = len(set(' '.join(df['comment']).split()))
    premier_commentaire = df['comment'].iloc[0] if not df.empty else ""
    dernier_commentaire = df['comment'].iloc[-1] if not df.empty else ""

    resume = (
    f"✅ Analyse terminée pour la vidéo {video_id}\n"
    f"🔢 Nombre de commentaires : {len(cleaned_comments)}\n"
    f"📊 Accuracy : {accuracy:.2f}\n"
    f"🧠 F1-score : {f1:.2f}\n"
    f"🔤 Top mots : {top_words}"
    f"👍 Nombre de commentaires positifs : {nb_positif}\n"
    f"👎 Nombre de commentaires négatifs : {nb_negatif}\n"
    f"😐 Nombre de commentaires neutres : {nb_neutre}\n"
    f"📝 Longueur moyenne des commentaires : {longueur_moyenne:.1f} caractères\n"
    f"🔠 Nombre de mots uniques : {mots_uniques}\n"
    f"🏆 Commentaire le plus long : {commentaire_le_plus_long}\n"
    f"⚡ Commentaire le plus court : {commentaire_le_plus_court}\n"
    f"⏩ Premier commentaire : {premier_commentaire}\n"
    f"⏪ Dernier commentaire : {dernier_commentaire}\n"
    )
    return resume, score