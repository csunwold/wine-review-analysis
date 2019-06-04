import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import MiniBatchKMeans

import nltk
nltk.download('punkt')

reviews = pd.read_csv("winemag-data-130k-v2.csv")

scaler = MinMaxScaler()

reviews['points'] = scaler.fit_transform(reviews['points'].values.reshape(-1, 1))

# Optimize for several different dimensions
dimensions = ['winery', 'variety', 'province', 'region_1']
# Scoring could be done one of several ways
scoring = ['points', 'price', 'points per dollar']

# Output the highest scoring wines for each dimension
for dimension in dimensions:
    for score in scoring:
        grouped = reviews.groupby(dimension).agg({
            'points': 'mean',
            'title': 'count',
            'price': 'mean'
        })
        grouped['points per dollar'] = grouped['points'] / grouped['price']
        # Remove groups that don't have enough reviews for consideration
        sorted = grouped[grouped['title'] > 10].sort_values(score, ascending=False)
        top = sorted.head()
        top.to_csv('highest_' + score + '_' + dimension + '.csv')

# Create a new stop word list custom for this data set.
# Remove all words that are not descriptive of the wine itself.
ignored_words = [
 'a',
 'for',
 ',',
 '.',
 'with',
 'and',
 'it',
 'has',
 'more',
 'than',
 'by',
 'the',
 'of',
 'are',
 'have',
 'but',
 'this',
 '\'s',
 'is',
 'as',
 'in',
 'like',
 'having',
 'giving',
 'makes',
 'line',
 'taste',
 'aftertaste',
 'sip',
 'notice',
 'held',
 'showing',
 'case',
 'used',
 'notes',
 'smell',
 '25',
 '50',
 '100',
 'maybe',
 'briefly',
 '12',
 '42',
 '43,'
 '90',
 '91',
 '92',
 '93',
 '94',
 '95',
 '96',
 '97',
 '98',
 '99',
 '100',
 'points',
 'point',
 'mainly',
 'search',
 'yes',
 'actual',
 'wait',
 '80',
 'near',
 '48',
 'way',
 '75',
 '25',
 '25th',
 'did',
 'like',
 'tastes',
 'quality',
 'allows',
 '100',
 'makes',
 '140',
 '220',
 'sb',
 'v90',
 '20',
 'meant',
 '50',
 '125',
 '87',
 '13',
 '25',
 'used',
 'only,'
 'situated',
 'most',
 'task',
 'absolute',
 'happens',
 'lot',
 'adequately',
 'issues',
 'parts',
 'disparate',
 'characteristically',
 'elegantly',
 'lend',
 'soon',
 'enjoy',
 'need',
 'begins',
 'aggressive',
 '47',
 '53',
 '1800s',
 '70',
 '14',
 'deals',
 'doesn',
 'suggests',
 'offset',
 'create',
 'totally',
 'benefit',
 'ending',
 'bring',
 'level',
 '000',
 'development',
 'tightly',
 'coupled',
 'integrated',
 'quite',
 'closed',
 '500',
 'end',
 'help',
 'drink',
 'follows',
 '89',
 '90',
 'great',
 'ba',
 'extremely',
 '07',
 'feel',
 '10',
 'initial',
 'come',
 'wine',
 'accompany',
 'providing',
 'lightly',
 'slightly',
 'core',
 'weight',
 '21',
 'finish',
 'pronounced',
 'produced',
 'turning',
 'display',
 'define',
 'sample',
 'years',
 'contributing',
 'eventually',
 'isn',
 'offers',
 '10',
 'term',
 'settle'
 '40',
 'smells',
 'coming',
 'applied',
 'deftly',
 'precise',
 'sample',
 'stands',
 'crowd',
 'disconnected',
 'dramatically',
 '1978',
 'zero',
 'likelihood',
 'zone',
 'turns',
 'lends',
 'prevailing',
 'note',
 'shouldn',
 'far',
 '17',
 'sources',
 'limits',
 '2011',
 '52',
 '19',
 '24',
 'single',
 '60',
 'start',
 'somewhat',
 '30',
 'private',
 'largest',
 '250',
 'pinch',
 'aim',
 'll',
 'lots',
 '19',
 'loosen',
 'qualify',
 'comparison',
 'especially',
 '1999',
 '2016',
 'added',
 'appreciate',
 '88',
 '2017',
 'producers',
 '2009',
 '88'
]
ignored_words_set = text.ENGLISH_STOP_WORDS.union(ignored_words)
svd = TruncatedSVD()
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

vectorizer = TfidfVectorizer(max_df=0.5,
                                 min_df=2, stop_words=ignored_words_set,
                                 use_idf=True)

km = MiniBatchKMeans(n_clusters=1000, init='k-means++', n_init=1,
                         init_size=5000, batch_size=5000)

descriptions = []
for index, row in reviews.iterrows():
 descriptions.append(row['description'])

matrix = vectorizer.fit_transform(descriptions)
km.fit(matrix)

clusters = km.labels_.tolist()

reviews['category'] = clusters

reviews.to_csv('categorized.csv')

# Print out top terms per cluster
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(1000):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()