import pymc as pm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from bertopic import BERTopic
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data_final = pd.read_csv('manifesto.csv')



data_env = data_final[data_final.cmp_code == '501']
data_env['year'] = data_env.edate.dt.year
data_env['half_decade'] = data_env.year//5*5
data_env['decade'] = data_env.year//10*10

conds = np.r_[np.NINF, np.linspace(10, 90, 9), 95, 98, np.inf]
choice = ['ECO', 'LEF', 'SOC', 'LIB', 'CHR', 'CON', 'NAT', 'AGR', 'ETH', 'SIP', 'DIV', 'MI']

data_env['partyfam'] = pd.cut(data_env.parfam, bins=conds, labels=choice)

data_env_filt = data_env[data_env.year < 2020]
plt_dt = (data_env_filt
 .groupby(['half_decade', 'partyfam'])
 .agg({'cmp_code': 'count'})
 .reset_index()
)

sns.boxplot(data=plt_dt, x='partyfam', y='cmp_code')

plt_dt = plt_dt[plt_dt.cmp_code > 10]

sns.lineplot(data=plt_dt, x='half_decade', y='cmp_code', hue='partyfam')


dynLDA_df = (data_env_filt
             .groupby(['half_decade', 'partyname'])
             .agg({'text': lambda x: ' '.join(x)})
             )

vectorizer_model = CountVectorizer(stop_words='english')

topic_model = BERTopic(verbose=True, vectorizer_model=vectorizer_model)

topics, probs = topic_model.fit_transform(dynLDA_df.text.to_list())

topics_over_time = topic_model.topics_over_time(dynLDA_df.text.to_list(), 
                                                dynLDA_df.reset_index().half_decade.to_list(),
                                                datetime_format='%Y')

topic_model.get_topic_info()


import plotly.io as pio
pio.renderers.default = "vscode"

fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20)

dir(fig)

import ipykernel
fig.write_json('topics_over_time.json')


def cmp_code_tranform(cmp_col):
    n_topic = (cmp_col
    .str.extract(r'(^[1-9])')
    .nunique()
    )
    return n_topic


sLDA_data = (data_final
.groupby('manifesto_id')
.agg({'text': lambda x: ' '.join(x), 
      'pervote': 'mean',
      'cmp_code': lambda x: cmp_code_tranform(x),
      'parfam': 'mean'
      })
)

docs = sLDA_data.text
vectorizer = TfidfVectorizer(min_df=5)
embeddings = vectorizer.fit_transform(docs)

topic_model = BERTopic(stop_words='english')




topic_model = BERTopic(verbose=True)
topics, probs = topic_model.fit_transform(data_final.text)


topic_model.topics_per_class(data_final.text, classes=data_final.parfam.to_list())
topic_model.get_topic_info()

topic_model.get_topic(0)



X_train, X_test, y_train, y_test = train_test_split(data_final.text, 
                                                    data_final.pervote,
                                                    random_state=42,
                                                    test_size=0.25)

tfidf = TfidfVectorizer(stop_words='english')

train_tfidf = tfidf.fit_transform(X_train)

tfidf_dict = tfidf.get_feature_names()

num_tops = 7
num_words = train_tfidf.shape[1]
num_docs = train_tfidf.shape[0]

data = train_tfidf.toarray()

Wd = [len(doc) for doc in data]
alpha = np.ones(num_tops)
beta = np.ones(num_words)

theta = pm.Container([pm.CompleteDirichlet()])

pm.CompletedDirichlet()