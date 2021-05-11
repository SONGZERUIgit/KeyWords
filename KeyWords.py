#!/usr/bin/env python
# coding: utf-8

# In[3]:


from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import pandas as pd
import os
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
import gensim
import spacy
import nltk
from nltk.corpus import stopwords
from spacy import displacy


# In[4]:


def load_data(path = r"D:\NLP\dataset"):
    document_files = os.listdir(path)
    doc = [[""]] * len(document_files)
    i = 0
    for document in document_files:
        with open(path + "/" + document, "r") as f:
            doc[i] = f.read()
        i += 1
    return doc


# In[7]:


def deal_data(doc):
    news_df = pd.DataFrame({'document': doc})
    # removing everything except alphabets`
    news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z#]", " ")
    # removing short words
    news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
    # make all text lowercase
    news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    # tokenization
    tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
    # remove stop-words
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
    detokenized_doc = []
    for i in range(len(news_df)):
        t = ' '.join(tokenized_doc[i])
        detokenized_doc.append(t)
    tokenizer = RegexpTokenizer(r'\w+')
    # create English stop words list
    en_stop = get_stop_words('en')
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    # compile sample documents into a list
    # list for tokenized documents in loop
    texts = []
    for i in detokenized_doc:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        # add tokens to list
        texts.append(stemmed_tokens)
    return texts


# In[8]:


class KeyWord():
    def __init__(self,method,num_keywords):
        self.method=method
        self.num_keywords=num_keywords
    def LSA(self,texts,num_keywords):
        detokenized_doc = []
        for i in range(len(texts)):
            t = ' '.join(texts[i])
            detokenized_doc.append(t)
        news_df=pd.DataFrame()
        news_df['clean_doc'] = detokenized_doc
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, smooth_idf=True)
        X = vectorizer.fit_transform(news_df['clean_doc'])
        from sklearn.decomposition import TruncatedSVD
        # SVD represent documents and terms in vectors
        svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)
        svd_model.fit(X)
        terms = vectorizer.get_feature_names()
        for i, comp in enumerate(svd_model.components_):
            terms_comp = zip(terms, comp)
            sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:num_keywords]
            print("text " + str(i) + "'s keyword: ")
            for t in sorted_terms:
                print(t[0])
                print(" ")
    def LDA(self,texts,num_keywords):
        dictionary = corpora.Dictionary(texts)

        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in texts]
        # generate LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=30)
        print(ldamodel.print_topics(num_topics=5, num_words=num_keywords))
    def spacy1(self,data):
        a=len(data)
        nlp = spacy.load("en_core_web_sm")
        for i in range(a):
            result=self.extract_keywords(nlp,data[i])
            print("text"+str(i)+"'s keyword:")
            print(result)
    def extract_keywords(self,nlp, sequence, special_tags: list = None):
        result = []
        # edit this list of POS tags according to your needs.
        pos_tag = ['PROPN', 'NOUN', 'ADJ']
        doc = nlp(sequence.lower())
        if special_tags:
            tags = [tag.lower() for tag in special_tags]
            for token in doc:
                if token.text in tags:
                    result.append(token.text)
        for chunk in doc.noun_chunks:
            final_chunk = ""
            for token in chunk:
                if (token.pos_ in pos_tag):
                    final_chunk = final_chunk + token.text + " "
            if final_chunk:
                result.append(final_chunk.strip())
        for token in doc:
            if (token.text in nlp.Defaults.stop_words or token.text in punctuation):
                continue
            if (token.pos_ in pos_tag):
                result.append(token.text)
        return list(set(result))
    def spacy2(self,doc):
        for ent in doc.ents:
            print(ent.text, ent.label_)
        displacy.render(doc, style='ent', jupyter=True)
    def fit(self):
        if self.method=='LSA':
            doc=load_data()
            data=deal_data(doc)
            self.LSA(data,num_keywords=self.num_keywords)
        if self.method=='LDA':
            doc = load_data()
            data = deal_data(doc)
            self.LDA(data,num_keywords=self.num_keywords)
        if self.method=='spacy1':
            data=load_data()
            self.spacy1(data)
        if self.method=='spacy2':
            nlp = spacy.load("en_core_web_sm")
            data=load_data()
            a = len(data)
            for i in range(a):
                doc = nlp(data[i].lower())
                print("text" + str(i) + "'s keyword:")
                self.spacy2(doc)
        


# In[9]:


if __name__ == '__main__':
    model=KeyWord(method='LSA',num_keywords=20)
    model.fit()


# In[10]:


model=KeyWord(method='LDA',num_keywords=20)
model.fit()


# In[11]:


model=KeyWord(method='spacy1',num_keywords=20)
model.fit()


# In[12]:


model=KeyWord(method='spacy2',num_keywords=20)
model.fit()


# In[ ]:




