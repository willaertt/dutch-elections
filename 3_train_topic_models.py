'''
Pipeline for training BERTopic models for different min topic size tresholds as well
https://huggingface.co/cointegrated/rubert-tiny2 
https://huggingface.co/ai-forever/sbert_large_mt_nlu_ru

'''

#import libraries
import numpy as np
import os
import pandas as pd
from bertopic import BERTopic

if __name__ == "__main__":

    #specify path to data files
    csv_sample_file = "/home/tom/Documents/code/GitHub/dutch-elections/DATA/Parties_and_Leaders_74128_20251027_143525.csv"

    #specify path to embeddings
    embeddings_folder = "embeddings"
   
    #load data
    print('load data')
    telegram_df = pd.read_csv(csv_sample_file)

    docs = list(telegram_df['text']) #get list of messages
    sample_size = print('number of docs', len(docs))

    #train the topic model with different minimum topic sizes and for different embedding models
    min_topic_sizes = [50, 60, 70, 80, 90, 100]

    for topic_size in min_topic_sizes:
        for embedding_file in os.listdir(embeddings_folder):

            #get name for embedding file 
            embedding_name = embedding_file.split('.')[0]

            #check if the directory for storing the topic model exists, otherwise create it
            topic_model_folder = "BERTopic_models/" + embedding_name + "_topic_model_" + str(topic_size)
            if not os.path.isdir(topic_model_folder):
                print('create topic model folder')
                os.mkdir(topic_model_folder)

            print('load embeddings')
            embedding_path = os.path.join(embeddings_folder, embedding_file)
            embedding = np.load(embedding_path)

            #fit the model to the messages
            print('fit topic model with min size' , str(topic_size), 'and model', embedding_name)
            topic_model = BERTopic(min_topic_size = topic_size, verbose =True) #language is not English
            topics, probabilities = topic_model.fit_transform(docs, embedding)

            print('save topic model')
            topic_model.save(topic_model_folder, serialization="safetensors", save_ctfidf=True)