#!/usr/bin/env python
# coding: utf-8

# In[1]:


cd fastText_multilingual-master


# In[2]:


import numpy as np
#import FastVector


# In[3]:


import scipy


# In[4]:


import pybind11


# In[5]:


class FastVector:
    """
    Minimal wrapper for fastvector embeddings.
    ```
    Usage:
        $ model = FastVector(vector_file='/path/to/wiki.en.vec')
        $ 'apple' in model
        > TRUE
        $ model['apple'].shape
        > (300,)
    ```
    """

    def __init__(self, vector_file='', transform=None):
        """Read in word vectors in fasttext format"""
        self.word2id = {}

        # Captures word order, for export() and translate methods
        self.id2word = []

        print('reading word vectors from %s' % vector_file)
        with open(vector_file, 'r') as f:
            (self.n_words, self.n_dim) =                 (int(x) for x in f.readline().rstrip('\n').split(' '))
            self.embed = np.zeros((self.n_words, self.n_dim))
            for i, line in enumerate(f):
                elems = line.rstrip('\n').split(' ')
                self.word2id[elems[0]] = i
                self.embed[i] = elems[1:self.n_dim+1]
                self.id2word.append(elems[0])
        
        # Used in translate_inverted_softmax()
        self.softmax_denominators = None
        
        if transform is not None:
            print('Applying transformation to embedding')
            self.apply_transform(transform)

    def apply_transform(self, transform):
        """
        Apply the given transformation to the vector space
        Right-multiplies given transform with embeddings E:
            E = E * transform
        Transform can either be a string with a filename to a
        text file containing a ndarray (compat. with np.loadtxt)
        or a numpy ndarray.
        """
        transmat = np.loadtxt(transform) if isinstance(transform, str) else transform
        self.embed = np.matmul(self.embed, transmat)

    def export(self, outpath):
        """
        Transforming a large matrix of WordVectors is expensive. 
        This method lets you write the transformed matrix back to a file for future use
        :param The path to the output file to be written 
        """
        fout = open(outpath, "w")

        # Header takes the guesswork out of loading by recording how many lines, vector dims
        fout.write(str(self.n_words) + " " + str(self.n_dim) + "\n")
        for token in self.id2word:
            vector_components = ["%.6f" % number for number in self[token]]
            vector_as_string = " ".join(vector_components)

            out_line = token + " " + vector_as_string + "\n"
            fout.write(out_line)

        fout.close()

    def translate_nearest_neighbour(self, source_vector):
        """Obtain translation of source_vector using nearest neighbour retrieval"""
        similarity_vector = np.matmul(FastVector.normalised(self.embed), source_vector)
        target_id = np.argmax(similarity_vector)
        return self.id2word[target_id]

    def translate_inverted_softmax(self, source_vector, source_space, nsamples,
                                   beta=10., batch_size=100, recalculate=True):
        """
        Obtain translation of source_vector using sampled inverted softmax retrieval
        with inverse temperature beta.
        nsamples vectors are drawn from source_space in batches of batch_size
        to calculate the inverted softmax denominators.
        Denominators from previous call are reused if recalculate=False. This saves
        time if multiple words are translated from the same source language.
        """
        embed_normalised = FastVector.normalised(self.embed)
        # calculate contributions to softmax denominators in batches
        # to save memory
        if self.softmax_denominators is None or recalculate is True:
            self.softmax_denominators = np.zeros(self.embed.shape[0])
            while nsamples > 0:
                # get batch of randomly sampled vectors from source space
                sample_vectors = source_space.get_samples(min(nsamples, batch_size))
                # calculate cosine similarities between sampled vectors and
                # all vectors in the target space
                sample_similarities =                     np.matmul(embed_normalised,
                              FastVector.normalised(sample_vectors).transpose())
                # accumulate contribution to denominators
                self.softmax_denominators                     += np.sum(np.exp(beta * sample_similarities), axis=1)
                nsamples -= batch_size
        # cosine similarities between source_vector and all target vectors
        similarity_vector = np.matmul(embed_normalised,
                                      source_vector/np.linalg.norm(source_vector))
        # exponentiate and normalise with denominators to obtain inverted softmax
        softmax_scores = np.exp(beta * similarity_vector) /                          self.softmax_denominators
        # pick highest score as translation
        target_id = np.argmax(softmax_scores)
        return self.id2word[target_id]

    def get_samples(self, nsamples):
        """Return a matrix of nsamples randomly sampled vectors from embed"""
        sample_ids = np.random.choice(self.embed.shape[0], nsamples, replace=False)
        return self.embed[sample_ids]

    @classmethod
    def normalised(cls, mat, axis=-1, order=2):
        """Utility function to normalise the rows of a numpy array."""
        norm = np.linalg.norm(
            mat, axis=axis, ord=order, keepdims=True)
        norm[norm == 0] = 1
        return mat / norm
    
    @classmethod
    def cosine_similarity(cls, vec_a, vec_b):
        """Compute cosine similarity between vec_a and vec_b"""
        return np.dot(vec_a, vec_b) /             (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

    def __contains__(self, key):
        return key in self.word2id

    def __getitem__(self, key):
        return self.embed[self.word2id[key]]


# In[6]:


#import numpy as np
#from fasttext import FastVector

# from https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
def normalized(a, axis=-1, order=2):
    """Utility function to normalize the rows of a numpy array."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def make_training_matrices(source_dictionary, target_dictionary, bilingual_dictionary):
    """
    Source and target dictionaries are the FastVector objects of
    source/target languages. bilingual_dictionary is a list of 
    translation pair tuples [(source_word, target_word), ...].
    """
    source_matrix = []
    target_matrix = []

    for (source, target) in bilingual_dictionary:
        if source in source_dictionary and target in target_dictionary:
            source_matrix.append(source_dictionary[source])
            target_matrix.append(target_dictionary[target])

    # return training matrices
    return np.array(source_matrix), np.array(target_matrix)

def learn_transformation(source_matrix, target_matrix, normalize_vectors=True):
    """
    Source and target matrices are numpy arrays, shape
    (dictionary_length, embedding_dimension). These contain paired
    word vectors from the bilingual dictionary.
    """
    # optionally normalize the training vectors
    if normalize_vectors:
        source_matrix = normalized(source_matrix)
        target_matrix = normalized(target_matrix)

    # perform the SVD
    product = np.matmul(source_matrix.transpose(), target_matrix)
    U, s, V = np.linalg.svd(product)

    # return orthogonal transformation which aligns source language to the target
    return np.matmul(U, V)


# # We create dictionaries for both languages based off of the FastText Wiki vectors for each language: 

# In[7]:



fr_dictionary = FastVector(vector_file='wiki.fr.vec')
en_dictionary = FastVector(vector_file='wiki.en.vec')


# # We create a bilingual dictionary based on overlappings between the two languages:

# In[7]:


fr_words = set(fr_dictionary.word2id.keys())
en_words = set(en_dictionary.word2id.keys())
overlap = list(en_words & fr_words)
bilingual_dictionary = [(entry, entry) for entry in overlap]


# In[8]:


# form the training matrices
source_matrix, target_matrix = make_training_matrices(
    fr_dictionary, en_dictionary, bilingual_dictionary)


# # We align the FR dictionary with the EN one:

# In[9]:


# learn and apply the transformation
transform = learn_transformation(source_matrix, target_matrix)
fr_dictionary.apply_transform(transform)


# In[ ]:


fr_words = set(fr_dictionary.word2id.keys())


# In[10]:



import re
import numpy as np
import pandas as pd
from pprint import pprint
from nltk.tokenize import sent_tokenize, word_tokenize
import os
#The OS module in Python provides a way of using operating system dependent functionality. 
#The functions that the OS module provides allows you to interface with the underlying operating system 
#that Python is running on – be that Windows, Mac or Linux.

from os import listdir
from os.path import isfile, join

# Gensim
import gensim
import gensim.corpora as corpora
from gensim import models, corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
get_ipython().run_line_magic('matplotlib', 'inline')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[11]:


import nltk as nltk 
nltk.download('stopwords')


# In[ ]:


#stopwords = nltk.corpus.stopwords.words("stopwords_Latin.txt")


# In[12]:


def tokenize(text):
    tokens = word_tokenize(text)
    tokens = _pre_clean(tokens)
    tokens = [token for token in tokens if len(token) > 0]
    tokens = [token for token in tokens if token not in stopwords]
    #tokens = [get_lemma(token) for token in tokens]
    return tokens


# In[13]:


def _pre_clean(list_of_text):
        '''
        preliminary cleaning of the text
        - remove new line character i.e. \n or \r
        - remove tabs i.e. \t
        - remove extra spaces
        '''
        cleaned_list = []
        for text in list_of_text:
            # print("original:", text)
            text = text.replace('\\n', ' ')
            text = text.replace('\\r', ' ')
            text = text.replace('\\t', ' ')
            pattern = re.compile(r'\s+')
            text = re.sub(pattern, ' ', text)
            text = text.strip()
            text = text.lower()
            # check for empty strings
            if text != '' and text is not None:
                cleaned_list.append(text)

        return cleaned_list


# In[14]:



stopwords = nltk.corpus.stopwords.words('stop_words_poetry.txt')

stopwords.append('...')
stopwords.append("'d")
stopwords.append('...')
stopwords.append("&")
stopwords.append("upon")
stopwords.append("also")
stopwords.append("hath")
stopwords.append("must")
stopwords.append("therefore")
stopwords.append("doth")
stopwords.append("could")
stopwords.append("would")
stopwords.append("another")
stopwords.append("much")
#stopwords.append("give")
stopwords.append("like")
stopwords.append("since")
#stopwords.append("many")
stopwords.append("without")
#stopwords.append("first")
stopwords.append("though")
#stopwords.append("well")
stopwords.append("often")
#stopwords.append("great")
stopwords.append("either")
stopwords.append("even")
stopwords.append("shall")
#stopwords.append("they")
stopwords.append("what")
stopwords.append("their")
#stopwords.append("more")
#stopwords.append("there")
#stopwords.append("your")
stopwords.append("them")


# In[ ]:


stopwords.extend(['a', 'like', 'you', 'they', 'he', 'be', 'it', 'your', 'her', 'of', 'more', 'there', 'no', 'not', '’', 'what', 'my', 'his', 'she', 'to', 'our', 'me', 'we', 'in', 'can', 'us', 'an', 'if', 'do', 'this', '”', 'because', 'who', 'hand', 'but', 'him'])


# In[15]:


cd fastText_multilingual-master


# # We open, label, and tokenize all EN poems:

# In[16]:


HOME = os.getcwd()

TEXTS_DIR = HOME + "/cannes_&_stuff/"

filelabels_en = {}

texts_data = []

files = [f for f in os.listdir(TEXTS_DIR) if os.path.isfile(os.path.join(TEXTS_DIR, f))]

import string
from string import punctuation

remove_punct_map = dict.fromkeys(map(ord, string.punctuation))

tokens_total = []

count = -1
 
os.chdir(TEXTS_DIR)
    
for f in files:
    #os.chdir(TEXTS_DIR)
    with open(f, "r", encoding='utf-8', errors = 'ignore') as openf:
        tokens = []
        count = count + 1
        filelabels_en[count] = os.path.basename(openf.name)
        for line in openf:
            sent_text = nltk.sent_tokenize(line)
            for sentence in sent_text:
                tokens1 = tokenize(sentence)
                tokens1 = [item.translate(remove_punct_map)
                      for item in tokens1]
                #filter_object = filter(lambda x: x != "", tokens1)
                tokens1 = [x for x in tokens1 if x!= ""]
                tokens1 = [x.lower() for x in tokens1]
                for token in tokens1:
                    tokens.append(token)
                    tokens_total.append(token)
                #if random.random() > .99:
                #print(tokens)
    #print(tokens_total)
    texts_data.append(tokens)

print(filelabels_en)


# In[17]:


filelabels1 = list(filelabels_en)


# In[ ]:



#pwd


# In[18]:


for i in range(len(filelabels1)):
    texts_data[i] = [x for x in texts_data[i] if x not in stopwords]


# In[ ]:


#vect_en = []


# In[19]:


def l2_norm(x):
   return np.sqrt(np.sum(x**2))

def div_norm(x):
   norm_value = l2_norm(x)
   if norm_value > 0:
       return x * ( 1.0 / norm_value)
   else:
       return x


# In[ ]:


# WE HAVE TO CONVERT EN_WORDS & FR_WORDS TO LISTS (BEFORE RUNNING THE CELLS BELOW) otherwise the code below is not reliable


# In[20]:


en_words = list(en_words)
fr_words = list(fr_words)


# In[ ]:



# Just checking, you don't have to run this one, might jam up your notebook (it's a lot to print out)
for j in range(len(texts_data[1])):
    if texts_data[1][j] in en_words:
        print(div_norm(la_dictionary[texts_data[1][j]]))
    else:
        continue


# # We generate vectors for all EN poems:

# In[21]:


vect_en = []

for i in range(len(filelabels1)):
        vect1 = []
        for j in range(len(texts_data[i])):
            if texts_data[i][j] in en_words:
                vect1.append(div_norm(en_dictionary[texts_data[i][j]]))
            else:
                continue
        vect0 = sum(vect1) / len(texts_data[i])
        vect_en.append(vect0)


# In[22]:



len(vect_en)


# In[ ]:


#print(vect_en[21])


# In[23]:



cd ..


# In[24]:



HOME = os.getcwd()


# In[ ]:


pwd


# # We move on to the FR subcorpus:

# In[25]:



TEXTS_DIR = HOME + "/cannes_fr/"


# In[26]:



stopwords = nltk.corpus.stopwords.words("french1.txt")


# In[27]:


len(filelabels1)


# In[28]:


filelabels_fr = {}

texts_data = []

files = [f for f in os.listdir(TEXTS_DIR) if os.path.isfile(os.path.join(TEXTS_DIR, f))]

import string
from string import punctuation

remove_punct_map = dict.fromkeys(map(ord, string.punctuation))

tokens_total = []

count = len(filelabels1) - 1
 
os.chdir(TEXTS_DIR)
    
for f in files:
    #os.chdir(TEXTS_DIR)
    if count <= 300:
        with open(f, "r", encoding='utf-8', errors = 'ignore') as openf:
            tokens = []
            count = count + 1
            filelabels_fr[count] = os.path.basename(openf.name)
            for line in openf:
                sent_text = nltk.sent_tokenize(line)
                for sentence in sent_text:
                    tokens1 = tokenize(sentence)
                    tokens1 = [item.translate(remove_punct_map)
                      for item in tokens1]
                #filter_object = filter(lambda x: x != "", tokens1)
                    tokens1 = [x for x in tokens1 if x!= ""]
                    for token in tokens1:
                        tokens.append(token)
                        tokens_total.append(token)
                #if random.random() > .99:
                #print(tokens)
    #print(tokens_total)
        texts_data.append(tokens)

print(filelabels_fr)


# In[29]:


len(texts_data)


# In[30]:



filelabels2 = list(filelabels_fr)


# In[31]:



for i in range(100):
    texts_data[i] = [x for x in texts_data[i] if x not in stopwords]


# # We generate vectors representing every single FR poem:

# In[32]:


vect_fr = []

for i in range(100):
        vect1 = []
        for j in range(len(texts_data[i])):
            if texts_data[i][j] in fr_words:
                vect1.append(div_norm(fr_dictionary[texts_data[i][j]]))
            else:
                continue
        vect0 = sum(vect1) / len(texts_data[i])
        vect_fr.append(vect0)


# In[31]:



len(vect_fr)


# # We consolidate EN and FR vectors into one single list:

# In[33]:



vect_total = [*vect_en, *vect_fr]


# In[34]:



len(vect_total)


# In[ ]:


# We create a list of all labels as well, EN & FR together:


# In[35]:



#labels = filelabels1 + filelabels2

labels = [*filelabels1, *filelabels2]


# In[35]:


len(labels)


# In[36]:



dt = [('correlation', float)]


# In[37]:



vect_mat = np.mat(vect_total)


# # We calculate the matrix of similiarities between all vectors, EN & FR. Then we generate the network representing that matrix: the nodes are all EN & FR poem-vectors and the edges represent the similarities between every two nodes:

# In[38]:



similarity_matrix = np.matrix((vect_mat * vect_mat.T).A, dtype=dt)


# In[39]:



import networkx as nx

G = nx.from_numpy_matrix(similarity_matrix)

weights = [(G[tpl[0]][tpl[1]]['correlation']) for tpl in G.edges()]


# In[40]:



e = [(x, x) for x in G.nodes()] 
G.remove_edges_from(e)


# In[56]:


def draw_graph(G):
    weights = [(G[tpl[0]][tpl[1]]['correlation']) for tpl in G.edges()]
    normalized_weights = [400*weight/sum(weights) for weight in weights]
    fig, ax = plt.subplots(figsize=(25, 16))
    pos=nx.spring_layout(G)
    labels1 = dict([x for x in enumerate(labels)])
    #labels=labels
    nx.draw_networkx(
        G,
        pos,
        edges=G.edges(),
        width=normalized_weights,
        labels=labels1,
        with_labels=True,
        node_size=800,
        node_color='r',
        alpha=1,
        font_color = 'w',
        font_size=20
    )
    #plt.show()
    return


# In[57]:


import matplotlib.pyplot as plt


# In[58]:


draw_graph(G) # Here is our bilingual corpus


# # This is our bilingual English and French corpus; the nodes are the poems represented as correlated vectors based off of the wiki multilingual word embeddings that we aligned in FastText. 

# In[41]:



len(list(G.nodes))


# In[42]:



len(list(G.edges))


# In[ ]:



list(G.edges)[0]


# In[43]:



weights = [(tpl,(G[tpl[0]][tpl[1]]['correlation'])) for tpl in G.edges()]


# # We sort the edges in the decreasing order of similarities between the poem-nodes they connect:

# In[44]:



Sorted_weights = sorted(weights, key = lambda t: t[1], reverse = True)


# In[45]:


print(Sorted_weights[0]) # the strongest correlation


# In[46]:



print(Sorted_weights[(len(Sorted_weights)-1)]) # the weakest correlation


# In[ ]:



#VECTORS AND FILE LABELS NEED BE SWITCED TO EN & FR


# In[63]:



filelabels_en[44]


# In[64]:



filelabels_en[181]


# In[66]:



filelabels_en[171]


# In[79]:



filelabels_en[3]


# In[48]:



filelabels_total = {}


# In[49]:


filelabels_total.update(filelabels_en)


# In[50]:


filelabels_total.update(filelabels_fr)


# In[51]:



Degrees = G.degree(weight = "correlation")
Sorted_degrees = sorted(Degrees, key = lambda t: t[1], reverse = True)


# In[71]:



Sorted_degrees[0] #node with the highest degree


# In[72]:


Sorted_degrees[len(Sorted_degrees)-1] #node with the lowest degree


# In[73]:



filelabels_total[22]


# In[75]:



#print(filelabels_total)


# In[76]:


print(Sorted_degrees)


# In[77]:



clo_cen = nx.closeness_centrality(G)
import operator
c = sorted(clo_cen.items(), key=operator.itemgetter(1), reverse=True)
print("Closeness centralities for G:", c)


# In[80]:


# Weighted Closeness Centrality:
clo_cen_w = nx.closeness_centrality(G, distance = 'correlation')
c_w = sorted(clo_cen_w.items(), key=operator.itemgetter(1), reverse=True)
print("Weighted closeness centralities for G in decreasing order", c_w)


# In[81]:


#Betweeness centrality
bet_cen = nx.betweenness_centrality(G, weight = "correlation")
bet = sorted(bet_cen.items(), key=operator.itemgetter(1), reverse=True)
print("Betweenness centralities for G in decreasing order:", bet)


# In[82]:


#Eigenvector centrality
eigenvector_centrality = nx.eigenvector_centrality(G, weight = "correlation")
eigenvector = sorted(eigenvector_centrality.items(), key=operator.itemgetter(1), reverse=True)
print("Eigenvector centralities for G in decreasing order:", eigenvector)


# In[ ]:



#dag_longest_path(G, weight = "correlation")


# In[ ]:



len(filelabels_total)


# In[116]:


len(filelabels_en)


# In[117]:


len(filelabels_fr)


# In[118]:



filelabels_fr


# In[119]:


filelabels_total


# In[52]:



E = list(G.edges)


# In[53]:



weights_list = [(e, (G[e[0]][e[1]]['correlation'])) for e in E]


# # We are trying to find a route that would alternate between EN & FR poems in the decreasing order of similarity (i.e., weight of connecting edges) without crossing the same node (i.e., poem) twice:

# Here are all the edges connecting EN to FR poems:

# In[54]:



weights_en_to_fr = [(e, (G[e[0]][e[1]]['correlation'])) for e in E if e[0] in filelabels_en and e[1] in filelabels_fr]


# In[ ]:



weights_en_to_fr[0]


# In[87]:



len(weights_en_to_fr)


# In[ ]:



print(weights_en_to_fr[0])


# In[55]:



Sorted_weights_en_to_fr = sorted(weights_en_to_fr, key = lambda t: t[1], reverse = True)


# In[53]:



Sorted_weights_en_to_fr[0]


# In[90]:


filelabels_total[22]


# In[91]:


filelabels_total[287]


# In[56]:



List_poem_itinerary = []


# In[82]:





List_poem_itinerary.extend([(22, filelabels_total[22]), (287, filelabels_total[287])])


# In[94]:



len(List_poem_itinerary)


# In[123]:


weights_to_en_287 = [e for e in Sorted_weights_en_to_fr if e[0][1]==287]


# In[124]:


weights_to_en_287


# # Let us identify the poem-nodes on such a route and the order in which they need to be crossed:

# In[57]:



List_poem_itinerary = []
List_poem_itinerary.extend([(22, filelabels_total[22]), (287, filelabels_total[287])])

i = List_poem_itinerary[(len(List_poem_itinerary) - 1)][0]

while len(List_poem_itinerary) <= len(filelabels_total):
    if i < 200:
        weights_to_fr = []
        weights_to_fr = [e for e in Sorted_weights_en_to_fr if e[0][0]==i]
        for j in range(1, len(weights_to_fr)):
            if (weights_to_fr[j][0][1], filelabels_total[weights_to_fr[j][0][1]]) not in List_poem_itinerary:
                List_poem_itinerary.append((weights_to_fr[j][0][1], filelabels_total[weights_to_fr[j][0][1]]))
                i = weights_to_fr[j][0][1]
                break
        else:
            break
    else:
        weights_to_en = []
        weights_to_en = [e for e in Sorted_weights_en_to_fr if e[0][1]==i]
        for k in range(1, len(weights_to_en)): 
            if (weights_to_en[k][0][0], filelabels_total[weights_to_en[k][0][0]]) not in List_poem_itinerary:
                List_poem_itinerary.append((weights_to_en[k][0][0], filelabels_total[weights_to_en[k][0][0]]))
                i = weights_to_en[k][0][0]
                break
        else:
            break


# In[58]:


print(List_poem_itinerary)


# In[66]:


len(List_poem_itinerary)


# In[87]:


Itinerary_names = []


# In[88]:


for i in range(len(List_poem_itinerary)):
    Itinerary_names.append(List_poem_itinerary[i][1])


# In[ ]:





# In[ ]:





# # We are looking for a line in each of these poems along our route--stronger to weaker cross-language links, to the least aligned--a line best representing the poem vector-prosody-wise, that is, a line whose vector has the greatest cosine similarity to the vector of the poem as a whole

# Let's start with an example and then generalize to the whole itinerary:

# In[ ]:



Oceano = ['Les particules d’eau se collent à la peau', 'regorgent d’abstraction', 'Cette sensation de fluidité intrinsèque à chaque atome', 'perpètre la cénesthésie créative des neuf premiers mois', 'L’imaginaire se berce dans la pensée', 'se concentre sur la caresse du geste répété', 'se laisse flotter comme un objet éloigné de tout', 'C’est ainsi que je survis', 'sur une île de bonheur submergée par les ondes', 'Je regarde le monde à travers ce voile', 'le seul que je peux supporter', 'je constate sa beauté et mon indifférence', 'ses couleurs se font de plus en plus intenses', 'mes yeux sont aveuglés par la tendre tempête', 'J’ai enfin appris à nager']


# In[ ]:


texts_tokens = []

for y in Oceano:
    #os.chdir(TEXTS_DIR)
    #with open(f, "r", encoding='utf-8', errors = 'ignore') as openf:
        tokens = []
        #count = count + 1
        #filelabels_fr[count] = os.path.basename(openf.name)
        #for line in y:
            #sent_text = nltk.sent_tokenize(line)
        
        tokens1 = tokenize(y)
        tokens1 = [item.translate(remove_punct_map)
                      for item in tokens1]
                #filter_object = filter(lambda x: x != "", tokens1)
        tokens1 = [x for x in tokens1 if x!= ""]
        for token in tokens1:
                    tokens.append(token)
                    #tokens_total.append(token)
                #if random.random() > .99:
                #print(tokens)
    #print(tokens_total)
        texts_tokens.append(tokens)


# In[ ]:



for i in range(len(texts_tokens)):
    texts_tokens[i] = [x for x in texts_tokens[i] if x not in stopwords]


# In[ ]:



len(texts_tokens)


# In[ ]:



print(texts_tokens)


# In[ ]:



vect_oceano = []

for i in range(len(texts_tokens)):
        vect1 = []
        for j in range(len(texts_tokens[i])):
            if texts_tokens[i][j] in fr_words:
                vect1.append(div_norm(fr_dictionary[texts_tokens[i][j]]))
            else:
                continue
        vect0 = sum(vect1) / len(texts_tokens[i])
        vect_oceano.append(vect0)


# In[ ]:



# from numpy import dot
# from numpy.linalg import norm


# In[ ]:


# We compute cosine similarity between all the [vectors of the] lines making up the poem [vect_oceano] 
# and the overall poem they belong in [oceanotherapie_sybille_rembard.txt / node no. 400]

cos_list = []

for i in range(len(vect_oceano)):
    cos_sim = 0
    cos_sim = dot(vect_oceano[i], vect_total[400])/(norm(vect_oceano[i])*norm(vect_total[400]))
    cos_list.append((i, cos_sim))


# In[ ]:



print(cos_list)


# In[ ]:



sorted_cos_list = sorted(cos_list, key = lambda t: t[1], reverse = True)


# In[ ]:



sorted_cos_list[0]


# In[ ]:


# Here is the line that best represents the poem in terms of vector prosody
print(Oceano[6])


# In[ ]:


pwd


# In[59]:



cd ..


# In[60]:


cd cannes_en_fr


# In[ ]:


HOME = os.getcwd()


# In[81]:



pwd


# In[82]:


cd ..


# # We need to move all--and only--the poems on our route into a new directory. We start with the EN ones and then do the same with the FR poems on the route.

# In[83]:


import shutil 
import os 
import logging


# In[84]:


HOME = os.getcwd()


# In[85]:


source = HOME + "/cannes_&_stuff/"


# In[91]:


destination = HOME + "/cannes_itinerary"


# In[89]:


print(Itinerary_names)


# In[97]:


len(Itinerary_names)


# In[90]:


files = os.listdir(source)


# In[93]:


for f in files:
    if f in Itinerary_names:
        shutil.move(source+f, destination)


# Now the FR ones:

# In[94]:



source = HOME + "/cannes_fr/"


# In[95]:


files = os.listdir(source)


# In[96]:


for f in files:
    if f in Itinerary_names:
        shutil.move(source+f, destination)


# # Now that we got all the poems needed into one directory, we convert each of them into a list of lines (in which every element is a line in that specific poem) and consolidate these lists into a list of lists whose order of elements is the one of destinations on our route:

# In[226]:



HOME = os.getcwd()

Poems_Dir = HOME + "/cannes_itinerary/"
#Poems_Dir = HOME

os.chdir(Poems_Dir)
lines = [i for i in range(len(Itinerary_names))]

for i in range(len(List_poem_itinerary)):
    lines_1 = []
    with open (List_poem_itinerary[i][1], 'rt') as file:
        for line in file:
                line = line.replace('\\n', ' ')
                line = line.replace('\\r', ' ')
                line = line.replace('\\t', ' ')
                pattern = re.compile(r'\s+')
                line = re.sub(pattern, ' ', line)
                line = line.strip()
                line = line.lower()
                # check for empty strings
                if line != '' and line is not None:
                    lines_1.append(line)
        lines[i] = lines_1
    
    
    

        


# In[227]:



lines[0]


# In[65]:


List_poem_itinerary[0]


# In[221]:


poems_tokens = []


# In[222]:



for line in lines:
    poems_tokens.append(line)

# Or poems_tokens.extend(lines)


# In[ ]:





# In[120]:



import copy

#poems_tokens = copy.copy(lines)


# In[69]:



# Let's make sure they are not one and the same object:

poems_tokens is lines


# In[77]:


lines[1]


# 
# We tokenize the lines of each poem separately and thus convert the lines into lists of tokens:

# In[223]:



for i in range(len(lines)):
    for j in range(len(lines[i])):
        tokens = []
        tokens1 = tokenize(lines[i][j])
        tokens1 = [item.translate(remove_punct_map)
                      for item in tokens1]
        tokens1 = [x for x in tokens1 if x!= ""]
        for token in tokens1:
                    tokens.append(token)
        poems_tokens[i][j] = tokens


# In[ ]:



poems_tokens[0]


# In[ ]:


lines[189]


# In[228]:



stopwords_0 = nltk.corpus.stopwords.words('stop_words_poetry.txt')


# In[229]:



for i in range(len(poems_tokens)):
    if i % 2 == 0:
        for j in range(len(poems_tokens[i])):
            poems_tokens[i][j] = [x for x in poems_tokens[i][j] if x not in stopwords_0]
    else:
        for j in range(len(poems_tokens[i])):
            poems_tokens[i][j] = [x for x in poems_tokens[i][j] if x not in stopwords]


# In[230]:


poems_tokens[0]


# In[126]:


len(poems_tokens)


# In[ ]:


#poems_tokens[1]


# In[231]:



for poem in poems_tokens:
    for k in poem:
        if k == []:
            poem.remove(k)
        else:
            if len(k) == 0:
                poem.remove(k)


# In[232]:


for i in range(len(poems_tokens)):
    if len(poems_tokens[i]) == 0:
        print(i)


# In[233]:


# Don't need to run this one if the one above worked

for i in range(len(poems_tokens)):
    if len(poems_tokens[i]) == 0:
        poems_tokens.remove(poems_tokens[i])
    else:   
        for j in range(len(poems_tokens[i])):
            if len(poems_tokens[i][j]) == 0:
                poems_tokens[i].remove(poems_tokens[i][j])


# In[132]:



from numpy import dot
from numpy.linalg import norm


# In[134]:



import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


# # We compute vectors for every single line in every single poem:

# In[234]:


vectors_of_lines1 = copy.copy(lines)

for i in range(len(poems_tokens)):
    vectors_of_lines1[i] = []
    if i % 2 == 0:
        for j in range(len(poems_tokens[i])):
            vect1 = []
            for k in range(len(poems_tokens[i][j])):
                    if poems_tokens[i][j][k] in en_words:
                        vect1.append(div_norm(en_dictionary[poems_tokens[i][j][k]]))
                    else:
                        continue
            if len(poems_tokens[i][j]) != 0:
                vect0 = sum(vect1) / len(poems_tokens[i][j])
            else:
                vect0 = 0
            vectors_of_lines1[i].append((j, vect0))
    else:
        for j in range(len(poems_tokens[i])):
            vect1 = []
            for k in range(len(poems_tokens[i][j])):
                    if poems_tokens[i][j][k] in fr_words:
                        vect1.append(div_norm(fr_dictionary[poems_tokens[i][j][k]]))
                    else:
                        continue
            if len(poems_tokens[i][j]) != 0:
                vect0 = sum(vect1) / len(poems_tokens[i][j])
            else:
                vect0 = 0
            vectors_of_lines1[i].append((j, vect0))


# In[ ]:





# In[235]:


vectors_of_lines1[0]


# In[236]:


for i in range(len(vectors_of_lines1)):
    if len(vectors_of_lines1[i]) != len(lines[i]):
        print(i, len(vectors_of_lines1[i]), len(lines[i]))


# In[162]:


lines[1]


# In[163]:


vectors_of_lines1[1]


# # We compute cosine similarity between [the vector of] every line in a poem and [the vector] that specific poem. Every poem will now be represented by a list of line numbers and values of [line and poem] cosine similarity:

# In[237]:


cos_list_total = copy.copy(vectors_of_lines1)

for i in range(len(vectors_of_lines1)):
    cos_list = []
    #sorted_cos_list = []
    for j in range(len(vectors_of_lines1[i])):
        cos_sim = 0
        cos_sim = dot(vectors_of_lines1[i][j][1], vect_total[List_poem_itinerary[i][0]])/(norm(vectors_of_lines1[i][j][1])*norm(vect_total[List_poem_itinerary[i][0]]))
        cos_list.append((j, cos_sim))
    cos_list_total[i] = cos_list


# In[238]:


cos_list_total[0]


# In[239]:


cos_list_total[1]


# In[240]:



sorted(cos_list_total[0], key = lambda t: t[1], reverse = True)


# In[241]:


cos_max0 = sorted(cos_list_total[0], key = lambda t: t[1], reverse = True)


# In[242]:


cos_max0


# In[243]:


cos_list_total


# In[244]:


cos_list_total1 = copy.copy(cos_list_total)

for i in range(len(cos_list_total1)):
    for j in range(len(cos_list_total1[i])):
        if type(cos_list_total1[i][j][1]) == np.ndarray:
            cos_list_total1[i][j] = (j, 0)


# In[245]:


cos_list_total1


# # We sort the cosine similarity lists representing the poems and keep only the maximum value and the number of the line whose cosine similarity to the poem it belongs in is maximum. Every poem on our route will be now represented by a tuple consisting of the line number and the cosine similarity value:

# In[246]:


cos_max = []
cos_sorted = list(range(len(cos_list_total1)))

for i in range(len(cos_list_total1)):
    cos_sorted[i] = sorted(cos_list_total1[i], key = lambda t: t[1], reverse = True)
    cos_max.append(cos_sorted[i][0])


# In[247]:


cos_max[0]


# In[248]:


cos_max[1]


# # Our "Bilingual Corpus Journey Poem" will include from each poem on the FR-EN itinerary the line with the maximum cosine similarity:

# In[249]:



Bilingual_Corpus_Journey_Poem = []

for i in range(len(lines)):
    k = cos_max[i][0]
    Bilingual_Corpus_Journey_Poem.append(lines[i][k])


# In[250]:




len(Bilingual_Corpus_Journey_Poem)


# In[251]:



print(Bilingual_Corpus_Journey_Poem, sep = "\n")


# Here is our 

# # Bilingual Corpus Journey Poem

# In[252]:



for line_of_verse in Bilingual_Corpus_Journey_Poem:
        print(line_of_verse)


# In[ ]:




