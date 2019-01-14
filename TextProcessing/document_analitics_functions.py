#http://josearcosaneas.github.io/python/r/procesamiento/lenguaje/2017/01/02/procesamiento-lenguaje-natural-0.html
#https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
#https://medium.com/@sabber/classifying-yelp-review-comments-using-lstm-and-word-embeddings-part-1-eb2275e4066b





"""Module to explore data.
Contains functions to help study, visualize and understand datasets.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


from string import punctuation
from nltk import wordpunct_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import itertools
import re
from wordcloud import WordCloud, STOPWORDS 

import spacy
from polyglot.text import Text
#**************for polyglot install********************
# pip install pyicu
# pip install polyglot
# pip install pycld2
# pip install morfessor
# polyglot download embeddings2.es
# polyglot download ner2.es
# polyglot download pos2.es

from nltk.parse import CoreNLPParser
#**************for stanford core NLP*******************
# https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK
#


def  lemmatization():
    nlp = spacy.load('es_core_news_sm') #es_core_news_md , es_core_news_sm
    doc = nlp(string)
    for token in doc:
        print(token.text, token.lemma_)
    
    return 

def pos(text_list,part='VERB',lib='spacy', lemma=False): #part =  VERB  , NOUN
    parts=[]
    for text in text_list:
        if lib=='spacy':
            nlp = spacy.load('es_core_news_sm') #es_core_news_md , es_core_news_sm
            doc = nlp(text)
            part_sample = []
            for token in doc:
                if part=='VERB':
                    if token.pos_== 'VERB':
                        if lemma:
                            part_sample.append(token.lemma_)
                        else:
                            part_sample.append(token.text)
                        #print(token.text, token.pos_)
                if part=='NOUN':
                    if token.pos_== 'NOUN':
                        if lemma:
                            part_sample.append(token.lemma_)
                        else:
                            part_sample.append(token.text)
                        #print(token.text, token.pos_)
            parts.append(' '.join(part_sample))
                        
        if lib=='polyglot':
            ptext = Text(string,hint_language_code='es')
            ptext.pos_tags    
        if lib=='corenlp':
            # For Stanford CoreNLP use, start the server with:
            # java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
            # -serverProperties StanfordCoreNLP-spanish.properties \
            # -preload tokenize,ssplit,pos,ner,parse \
            # -status_port 9003  -port 9003 -timeout 15000
            pos_tagger = CoreNLPParser('http://localhost:9003', tagtype='pos')
            print(pos_tagger.tag(string.split()))    
    
    
    return parts


def stemmer(tokens):
    Snowball_stemmer = SnowballStemmer('spanish')
    stemmers = [Snowball_stemmer.stem(word) for word in tokens]
    stemm_tokens = [stem for stem in stemmers if stem.isalpha() and len(stem) > 1]
    return stemm_tokens
    

def filter_text(text_list,stopw=True, stem=True, clean_numbers=True, punct=True, stopw_list=[]):
    text_list_filtered=[]
    for index, text in enumerate(text_list):   
        # Convert words to lower case 
        text_enc = text.encode('utf-8','ignore').lower().decode('utf-8') 
        if clean_numbers:           
            # Clean numbers in the text
            text_wo_numbers = ''.join([i for i in text_enc if not i.isdigit()])

        tokens = [word for word in wordpunct_tokenize(text_wo_numbers)]

        remove_word_list=[]
        stopw = [w.encode('utf-8') for w in stopwords.words('spanish')]


        if punct:
            remove_word_list = [u'.', u'[', ']', u',', u';', u'', u')', u'),', u' ', u'(',u'?',u'¿',u'!',u'¡',u'_',u'-']
            stopw.extend(remove_word_list) 
            words = [token for token in tokens if not token in stopw]
            tokens=words
        if stopw:     
            stopw.extend(stopw_list)   
            words = [token for token in tokens if not token in stopw]
            tokens=words

        # Optionally, shorten words to their stems
        if stem:
            tokens=stemmer(tokens)
        
        text_filtered=' '.join(tokens)
        text_list_filtered.append(text_filtered) 
        #print(text,text_filtered)
    
    # Return a list of strings filtered
    return text_list_filtered


def get_num_classes(labels):
    """Gets the total number of classes.
    # Arguments
        labels: list, label values.
            There should be at lease one sample for values in the
            range (0, num_classes -1)
    # Returns
        int, total number of classes.
    # Raises
        ValueError: if any label value in the range(0, num_classes - 1)
            is missing or if number of classes is <= 1.
    """

    num_classes=len(Counter(labels).keys()) # equals to list(set(words))
    num_examples=sum(Counter(labels).values()) # counts the elements' frequency    

#     num_classes = max(labels) + 1
#     missing_classes = [i for i in range(num_classes) if i not in labels]
#     if len(missing_classes):
#         raise ValueError('Missing samples with label value(s) '
#                          '{missing_classes}. Please make sure you have '
#                          'at least one sample for every label value '
#                          'in the range(0, {max_class})'.format(
#                             missing_classes=missing_classes,
#                             max_class=num_classes - 1))

#     if num_classes <= 1:
#         raise ValueError('Invalid number of labels: {num_classes}.'
#                          'Please make sure there are at least two classes '
#                          'of samples'.format(num_classes=num_classes))
    return num_classes


def get_num_words_per_sample(sample_texts):
    """Gets the median number of words per sample given corpus.
    # Arguments
        sample_texts: list, sample texts.
    # Returns
        int, median number of words per sample.
    """
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)


def frequency_distribution_of_ngrams(sample_texts, ngram_range=(1, 2), num_ngrams=50):
    """Plots the frequency distribution of n-grams.
    # Arguments
        samples_texts: list, sample texts.
        ngram_range: tuple (min, mplt), The range of n-gram values to consider.
            Min and mplt are the lower and upper bound values for the range.
        num_ngrams: int, number of n-grams to plot.
            Top `num_ngrams` frequent n-grams will be plotted.
    """
    # Create args required for vectorizing.
    kwargs = {
            'ngram_range': ngram_range,
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': 'word',  # Split text into word tokens.
    }
    vectorizer = CountVectorizer(**kwargs)

    # This creates a vocabulary (dict, where keys are n-grams and values are
    # idxices). This also converts every text to an array the length of
    # vocabulary, where every element idxicates the count of the n-gram
    # corresponding at that idxex in vocabulary.
    if len( list(filter(None, sample_texts)))==0:
            sample_texts.append(' '.join(['none']))
    vectorized_texts = vectorizer.fit_transform(sample_texts)
    
    # This is the list of all n-grams in the index order from the vocabulary.
    all_ngrams = list(vectorizer.get_feature_names())
    num_ngrams = min(num_ngrams, len(all_ngrams))
    # ngrams = all_ngrams[:num_ngrams]

    # Add up the counts per n-gram ie. column-wise
    all_counts = vectorized_texts.sum(axis=0).tolist()[0]

    # Sort n-grams and counts by frequency and get top `num_ngrams` ngrams.
    all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(
        zip(all_counts, all_ngrams), reverse=True)])
    ngrams = list(all_ngrams)[:num_ngrams]
    counts = list(all_counts)[:num_ngrams]
    
    return ngrams, counts, num_ngrams
    

def plot_frequency_distribution_of_ngrams(ngrams=[], counts=[], num_ngrams=0, name='',save=False,save_path=''):    
    idx = np.arange(num_ngrams)
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    ax.bar(idx, counts, align='center', width=0.9, color='g', alpha=0.8)
    plt.xlabel('N-grams')
    plt.ylabel('Frequencies')
    plt.title('N-grams frequency distribution of class %s'%name)
    plt.xticks(idx,ngrams,rotation='vertical')
    # plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
    plt.tight_layout()
    plt.rc('grid', linestyle="-", color='black')
    plt.grid()
    if save:
        plt.savefig(save_path+'frequency_distribution_of_ngrams_(%s).png' % name,dpi=900)
    else:
        plt.show()


def plot_sample_length_distribution(sample_texts,name='', save=False, word_count=True,save_path=''):
    """Plots the sample length distribution.
    # Arguments
        samples_texts: list, sample texts.
    """
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    if word_count:
        ax.hist([len(s.split()) for s in sample_texts], 50, width=0.9, color='b', alpha=0.7) #word count
        plt.xlabel('Length (words) of a sample')
    else:    
        ax.hist([len(s) for s in sample_texts], 50, width=0.9, color='b', alpha=0.7) #character count
        plt.xlabel('Length (characters) of a sample')
        

    plt.ylabel('Number of samples')
    plt.title('Sample length distribution of class %s'%name)
    # plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
    
    
    mean_length= get_num_words_per_sample(sample_texts)
    
    xl1, yl1 = [mean_length, mean_length], [0, len(sample_texts)]
    plt.plot(xl1, yl1,label='mean length', marker = 'o',color='red')
    plt.legend()
      
    
    plt.tight_layout()
    
    
    plt.text(mean_length,len(sample_texts)/2, "%d" % mean_length)
        
    plt.rc('grid', linestyle="-", color='black')
    plt.grid()
    if save:
        plt.savefig(save_path+'sample_length_distribution_(%s).png' % name,dpi=900)
    else:
        plt.show()


def plot_class_distribution(labels, name='', save=False,save_path=''):
    """Plots the class distribution.
    # Arguments
        labels: list, label values.
            There should be at lease one sample for values in the
            range (0, num_classes -1)
    """
    num_classes = get_num_classes(labels)
    count_map = Counter(labels)    #{'blue': 3, 'red': 2, 'yellow': 1}
    counts=[]
    classes=[]
    for key, value in count_map.items():
        counts.append(value)
        classes.append(key)
    
    idx = np.arange(num_classes)   
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(111)
    ax.bar(idx, counts, align='center', width=0.9, color='b', alpha=0.4)
    plt.xlabel('Classes')
    plt.ylabel('Number of samples')
    plt.title('Class distribution of %s'%name)
    plt.xticks(idx,classes,rotation='vertical')
    # plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
    plt.tight_layout()
    
    mean_counts=float(sum(counts)) / max(len(counts), 1)
    
    xl1, yl1 = [0, num_classes], [15, 15]
    xl2, yl2 = [0, num_classes], [mean_counts,mean_counts]
    xl3, yl3 = [0, num_classes], [100,100]
    plt.plot(xl1, yl1,label='recommended min', marker = 'o')
    plt.plot(xl2, yl2,label='data mean', marker = 'o')
    plt.plot(xl3, yl3,label='recommended max', marker = 'o')
    plt.legend()
    
    plt.text(num_classes, 15, "%d" % 15)
    plt.text(num_classes, int(mean_counts), "%d" % mean_counts)
    plt.text(num_classes, 100, "%d" % 100)
    
    plt.rc('grid', linestyle="-", color='black')
    plt.grid()
    if save:
        plt.savefig(save_path+'class_distribution_%s.png' % name,dpi=900)
    else:
        plt.show()
    
def plot_keywords_vs_classes(scatter_data, name='', save=False,save_path=''):

    idx = np.arange(2)   
    fig = plt.figure(figsize=(15,70))
    ax = fig.add_subplot(111)
    marker_size=100
    b=ax.scatter(scatter_data[:,0],  scatter_data[:,1],marker_size,c=scatter_data[:,2])
    plt.xlabel('Classes')
    plt.ylabel('Words')
    plt.title('Frequent words distribution of %s'%name)
    plt.xticks(scatter_data[:,0],scatter_data[:,0],rotation='vertical')
    cbar= plt.colorbar(b)
    cbar.set_label("word count", labelpad=+1)
    # plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
    plt.tight_layout()
    plt.rc('grid', linestyle="-", color='black')
    plt.grid()
    if save:
        plt.savefig(save_path+'keywords_vs_classes_%s.png' % name,dpi=900)
    else:
        plt.show()
        
        
def plot_wordcloud(dic,name='', save=False,save_path=''): 

    wordcloud = WordCloud(font_path='/Library/Fonts/Verdana.ttf',
                          relative_scaling = 1.0,
                          stopwords = {'a', 'de'} # set or space-separated string
                          )
    wordcloud.generate_from_frequencies(frequencies=dic)
    plt.figure( figsize=(10,7) )
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    if save:
        plt.savefig(save_path+'word_cloud_(%s).png' % name,dpi=900)
    else:
        plt.show()       