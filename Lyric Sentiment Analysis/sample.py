import pandas as pd
import numpy as np
import spacy
import langid
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans

#create Spacy and Vader natural language processing objects
nlp = spacy.load('en_core_web_sm')
analyzer = SentimentIntensityAnalyzer()

#create data frames to store all necessary source data and drop unwanted columns
df = pd.read_csv('songs_dataset.csv')
df.drop(['Album', 'Featuring','Tags','Producers','Writers'], axis = 1, inplace = True)
df_scores = pd.read_csv('Ratings_Warriner_et_al.csv', index_col="Word")
df_lyrics = pd.read_csv('lyrics_features.csv')

#function to perfrom data cleansing on the genre attribute
def clean_genre(df):
    l = len(df)
    for i in range(l):
        genre = df["Genre"].iloc[i]
        genre = genre.replace('[','')
        genre = genre.replace(']','')
        genre = genre.replace("'",'')
        genre = genre.split(', ')

        if len(genre) == 1:
            df["Genre"].iloc[i] = genre[0]
        else:
            comb_genre = ""
            count = 0
            c = 0
            for g in genre:
                try:
                    c = df.Genre.value_counts()["['" + g + "']"]
                except:
                    next
                if c > count:
                    comb_genre = g
                    count = c
                df["Genre"].iloc[i] = comb_genre
    return df

#function to perform data cleansing on the song lyrics
def clean_lyrics(song):
    song = song.lower()
    song = song.replace('verse', '')
    song = song.replace('hook', '')
    song = song.replace('chorus', '')
    song = song.replace('intro', '')
    song = re.sub(r'\[.*?\]','',song)
    song = nlp(song)
    song = [s.lemma_ for s in song if s.is_alpha and not s.is_stop]
    return (song)

#function to separate the song into individual words, also assigns valence and arousal scores to each word
def separate_words(song):
        dict = {}
        for s in song:
            if s in dict.keys():
                dict[s]['Count']+= 1
            if s not in dict.keys() and s in df_scores.index :
                nested_dict = {}
                nested_dict['Count'] = 1
                nested_dict['Valence'] = df_scores.loc[s]["V.Mean.Sum"]
                nested_dict['Arousal'] = df_scores.loc[s]["A.Mean.Sum"]
                dict[s] = nested_dict

        return dict

#function to include the top 10 occuring words for each song
def top_10_words(words):
    return dict(sorted(words.items(), key=lambda item: item[1]['Count'])[-10:])

#function to calculate valence and arousal scores for the entire song
def calc_features(df):
    l = len(df)
    for i in range(l):
        total_word_count = 0
        total_valence = 0.00
        total_arousal = 0.00
        for k,v in df["Words_10"].iloc[i].items():
            total_word_count += v['Count']
            total_valence += (v['Valence'] * v['Count'])
            total_arousal += (v['Arousal'] * v['Count'])
        df["Valence"].iloc[i] = total_valence/total_word_count
        df["Arousal"].iloc[i] = total_arousal/total_word_count
    return(df)

#function to extract the decade out of the song year
def get_decade(date):
    return(str(date)[0:3]+'0')

#function to assign sentiment based on simple valence and arousal threshold logic
def assign_sentiment(row):
    if row['Valence'] >= 5.00 and row['Arousal'] >=5.00:
        sent = 'Happy'
    elif row['Valence'] >= 5.00 and row['Arousal'] < 5.00:
        sent = 'Relaxed'
    elif row['Valence'] < 5.00 and row['Arousal'] >= 5.00:
        sent = 'Angry'
    else:
        sent = 'Sad'
    return(sent)

#function to calculate vader sentiment score
def vader_score(song):
    return analyzer.polarity_scores(song)

#random sampling of 5000 records with random_state assigned to re-create results
df_comb = df.sample(n = 5000, random_state=1).copy()

#use langid library to assign a language to each assign, filter to include only songs in English
df_comb['LyricsTemp'] = df_comb['Lyrics'].apply(langid.classify)
df_comb['LyricsLang'], df_comb['LyricsLangConf'] = zip(*df_comb.LyricsTemp)
df_en = df_comb[df_comb.LyricsLang == 'en'].copy().reset_index()

#assign Vader sentiment score and remove unwanted columns
df_en["Vader_Score"] = df_en['Lyrics'].apply(vader_score)
df_en.drop(['LyricsTemp', 'LyricsLang','LyricsLangConf', 'index'], axis = 1, inplace = True)

#apply functions to clean the genre attribute ans song lyrics
df_clean_genre = clean_genre(df_en).copy()
df_clean_genre["Lyrics"] = df_clean_genre["Lyrics"].apply(clean_lyrics)

#separate songs into individual words, select top 10 occuring words and assign valence and arousal scores based on the top 10 words
df_clean_genre["Words"] = df_clean_genre["Lyrics"].apply(separate_words)
df_clean_genre["Words_10"] = df_clean_genre["Words"].apply(top_10_words)
df_clean_genre["Valence"] = 0.00
df_clean_genre["Arousal"] = 0.00
df_features = calc_features(df_clean_genre).copy()

#create a decade attirbute using song year, remove unwanted columns
df_features['Decade'] = df_features['Date'].apply(get_decade)
df_features.drop(['Date','Lyrics','Words','Words_10'], inplace = True,axis = 1)

#assign sentiment using basic thresholds, include addtional song attributes from lyrics features data set
df_features['Sentiment'] = df_features.apply(assign_sentiment, axis = 1)
df_lyrics_filtered = df_lyrics[['Singer', 'Song','0','1','2','4','5','6','7','11']].copy()
df_merged = df_features.merge(df_lyrics_filtered, on=['Singer','Song'])
df_merged.rename(columns={'0':'Total_Word_Count', '1':'Unique_Word_Count','2':'Fraction_Unique_Words', '4':'Numer_Unique_Word_Lengths',
                          '5':'Min_Word_Length', '6':'Max_Word_Length','7':'Average_Word_Length', '11':'Sum_All_Word_Lengths'}, inplace = True)

#Clustering to assign sentiment labels
valence = df_merged['Valence'].to_numpy()
arousal = df_merged['Arousal'].to_numpy()

#Stacking the numpy arrays as multidimentional arrays to be used in KMeans as "X".
X = np.column_stack((valence, arousal))

#Choose cluster num. I'll start with 4.
n = 4

#X = 2-dim array of just valence and arousal
kmeans = KMeans(n_clusters=n)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

#Put Centroids into dataframe with current labels as indices, sort by valence.
df_centroids = pd.DataFrame(kmeans.cluster_centers_, columns=["valence", "arousal"])
df_centroids.sort_values('valence', ascending = False, inplace=True)

#label dictionary
labels = {}

df_centroids['sum'] = np.square(df_centroids['valence'] + df_centroids['arousal'])

val_mean = df_centroids['valence'].mean()
aro_mean = df_centroids['arousal'].mean()

happy_relaxed = df_centroids[df_centroids['valence'] > val_mean]
angry_sad = df_centroids[df_centroids['valence'] < val_mean]

happy = happy_relaxed[happy_relaxed['arousal'] == happy_relaxed['arousal'].max()].index.values.astype(int)[0]
relaxed = happy_relaxed[happy_relaxed['arousal'] == happy_relaxed['arousal'].min()].index.values.astype(int)[0]
angry = angry_sad[angry_sad['valence'] == angry_sad['valence'].max()].index.values.astype(int)[0]
sad = angry_sad[angry_sad['valence'] == angry_sad['valence'].min()].index.values.astype(int)[0]

labels[str(happy)] = 'happy'
labels[str(relaxed)] = 'relaxed'
labels[str(sad)] = 'sad'
labels[str(angry)] = 'angry'

#update full song data to include column "labels" with sentiment strings
df_merged = df_merged.assign(X=lambda x: y_kmeans)
df_merged['Cluster_Label'] = df_merged['X'].apply(lambda x: labels[str(x)])
df_merged.drop(['X'], inplace = True,axis = 1)


df_merged.to_csv('processed_lyrics_sample.csv', index = True)
