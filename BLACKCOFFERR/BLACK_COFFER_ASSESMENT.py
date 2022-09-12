#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Import required libraries
import pandas as pd
import requests
import bs4


import warnings
warnings.simplefilter('ignore')

from nltk import word_tokenize,SyllableTokenizer,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re
from collections import Counter
lemma = WordNetLemmatizer()

#Load the data
data = pd.read_csv('Input.csv')
Records,fields = data.shape

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0',}



#Saving the data for output
url_id = []
base_url = []
Pos_score = []
Neg_score = []
Pol_score = []
Sub_score = []
Average_sen_length = []
Percentage_of_com_words = []
Fog_Ind = []
Average_words_per_sen_length = []
Number_of_com_word = []
Total_number_of_words_aft_cleaned = []
Syll_Count_per_word = []
pron_count = []
Avg_word_length = []
    

#Data scraping and Text Analysis    
for ID in range(Records):
    
    #Loading URL and Scraping using BeautifulSoup
    Base_url = data['URL'][ID]
    base_url.append(Base_url)
    scrap_url = requests.get(Base_url,headers=headers)
    soup = bs4.BeautifulSoup(scrap_url.text,'lxml')
    
    #Scraping title
    soup_title = soup.select('h1')
    title = soup_title[0].text
    split_title = title.split(' ')
    
    #Scraping article
    soup_article = soup.select('p')
    
    article_list = []
    for i in range(len(soup_article)):
        article = soup_article[i].text
        article_list.append(article)
        
    #Concatenting the two lists and getting back into string by using join method 
    #(Adding '?' in the articel to easily extract the title and text to do the text analysis)
    if title[-1] == '?':
        Total_text = split_title + article_list    
        Article = ' '.join(Total_text).replace('\xa0','') 
    else:
        Total_text = split_title + ['?'] + article_list
        Article = ' '.join(Total_text).replace('\xa0','')
    
    #Saving the file
    URL_ID = data['URL_ID'][ID]
    file_name = 'URL_{}.txt'.format(URL_ID)
    url_id.append(URL_ID)
    with open(file_name,mode='w',encoding='utf_8') as f:      #Saving file to txtfile
        f.write(Article)
        
        
    #Text Anlaysis
    
    #Importing data
    with open(file_name,mode='r',encoding='utf_8') as f:
        Original_text = f.read()
        f.close()
     
    
    # Cleaning Data 
    #Splitting it by '?' to get only text an (Leaving Title)Incase if we have more than two "?" in text
    #We are getting list from index 1
    
    Text = Original_text.split('?')[1:]   
    get_back_string = ' '.join(Text)
    Text_lower_case = get_back_string.lower()
    cleaned_Text = re.sub('[^a-zA-Z]',' ',Text_lower_case) 
    cleaned_words = word_tokenize(cleaned_Text)
    
    
    # Analysis of Readability
    Number_of_words = len(cleaned_words)
    Number_of_sentences = len(sent_tokenize(Text_lower_case))

    Average_sentence_length = round((Number_of_words) / (Number_of_sentences))
    Average_sen_length.append(Average_sentence_length)
    

    #Here, Hard Words/Complex words = words with more than two syllables.
    Number_of_Complex_words = 0
    for word in cleaned_words:
        tk = SyllableTokenizer()
        syllables = tk.tokenize(word)
        if len(syllables) > 2:
            Number_of_Complex_words += 1

    Percentage_of_complex_words = round((Number_of_Complex_words)/(Number_of_words),2)
    Percentage_of_com_words.append(Percentage_of_complex_words)

    #Fog Index
    Fog_Index = round(0.4 * (Average_sentence_length + Percentage_of_complex_words),2)
    Fog_Ind.append(Fog_Index)
 
    
    
    #Average Number of words per Sentence
    Number_of_words = len(cleaned_words)
    Number_of_sentences = len(sent_tokenize(Text_lower_case))
    
    Average_words_per_sentence_length = round((Number_of_words) / (Number_of_sentences))
    Average_words_per_sen_length.append(Average_words_per_sentence_length)
        
    
    #Complex words
    Number_of_Complex_words = 0
    for word in cleaned_words:
        tk = SyllableTokenizer()
        syllables = tk.tokenize(word)
        if len(syllables) > 2:
            Number_of_Complex_words += 1    
    Number_of_com_word.append(Number_of_Complex_words)
    
    #Word Count = Removing Stopwords (Using stopwords class of nltk package)
    
    final_words = [word for word in cleaned_words if word not in stopwords.words('english')]
    Total_number_of_words_after_cleaned = len(final_words)
    Total_number_of_words_aft_cleaned.append(Total_number_of_words_after_cleaned)
    
    
    #Syllable count Per word 
    count = 0
    for word in cleaned_words:
        length_of_word = len(word)
        for char in word:
            if char in ['a','e','i','o','u']:
                count += 1

    if word[length_of_word - 2] == 'e':
        if word[length_of_word - 1] in ['s','d']:
            count -= 1
    Syllable_Count_per_word = round((count / Number_of_words))
    Syll_Count_per_word.append(Syllable_Count_per_word)
    
    
    #Calculating personal Pronouns in text
    regexp = re.sub('[^a-zA-Z]',' ',Text_lower_case)
    regexp_words = word_tokenize(regexp)

    pronouns_count = 0
    Personal_Pronouns = ['I','WE','MY','OURS','us','i','we','my','ours','We','My','Ours','Us']
    for word in regexp_words:
        if word in Personal_Pronouns:
            pronouns_count += 1
    pron_count.append(pronouns_count)
                    
    #Average Word Length
    Sum_of_characters_word = len([i for i in cleaned_Text])
    Total_number_of_words = len(cleaned_words)

    Average_word_length = round((Sum_of_characters_word / Total_number_of_words))
    Avg_word_length.append(Average_word_length)
                    
                    
    # Sentiment Analysis
    
    cleaned_words_after_stopwords = [word for word in cleaned_words if word not in stopwords.words('english')]
    lemmatizer = [lemma.lemmatize(l) for l in cleaned_words_after_stopwords]
      
    with open('Positive Words.txt',mode='r',encoding='utf_8') as P:
        Positive_words = P.read().split()
    
    with open('Negative Words.txt',mode='r',encoding='utf_8') as N:
        Negative_words = N.read().split()
    
    Positive_words_count = len([P for P in lemmatizer if P in Positive_words])
    Negative_words_count = len([N for N in lemmatizer if N in Negative_words])

    Polarity_score = (Positive_words_count - Negative_words_count) / ((Positive_words_count + Negative_words_count) + 0.000001)
    Subjectivity_score = (Positive_words_count + Negative_words_count) / (len(cleaned_words) + 0.000001)

    Pos_score.append(Positive_words_count)
    Neg_score.append(Negative_words_count)
    Pol_score.append(Polarity_score)
    Sub_score.append(Subjectivity_score)
    
    
    #Saving into xlsx file
    Dic = {'URL_ID':url_id,
           'URL':base_url,
           'POSITIVE SCORE':Pos_score,
           'NEGATIVE SCORE':Neg_score,
           'POLARITY SCORE':Pol_score,
           'SUBJECTIVITY SCORE':Sub_score,
           'AVG SENTENCE LENGTH':Average_sen_length,
          'PERCENTAGE OF COMPLEX WORDS':Percentage_of_com_words,
           'FOG INDEX':Fog_Ind,
           'AVG NUMBER OF WORDS PER SENTENCE':Average_words_per_sen_length,
          'COMPLEX WORD COUNT':Number_of_com_word,
          'WORD COUNT':Total_number_of_words_aft_cleaned,
          'SYLLABLE PER WORD':Syll_Count_per_word,
          'PERSONAL PRONOUNS':pron_count,
          'AVG WORD LENGTH':Avg_word_length}
    
#Saving into csv  file
dataframe = pd.DataFrame(Dic)
dataframe.to_csv('Output Data Structure.csv',index=False)


# In[ ]:




