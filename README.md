
## HOW TO RUN THE CODE?
- run the pos_tagger.py file as, 
 python3 pos_tagger.py <model_type>
- model_type = ['-f', '-r']
- -f for fffnn 
- -r for lstm
- the pre-trained model is saved in the same directory as the above file 


## About Code
Here we are doing an implementation of POS Tagging using a feedforward neural network and an LSTM rnn network. There are 2 ipynb files which have all the code about vocabulary creation, preprocessing, model building, training, evaluating, testing and plotting graphs and evaluation metrics. 
There is a main.py which has all the code on both the neural networks. and there is a pos_tagger.py which utilises the saved models and runs the predictions on an input sentence. 
There are also the dataset Treebank train, dev, test files.

## Vocabulary Creation
For creating vocabulary, we do different preprocessing for the 2 models.
* For FFNN, we separate out the train sentences and train tags from the .conllu file and similarly for dev and test. We add 4 special tokens PAD, UNK, START, END. START and END tokens are added to each sentence of the train and dev set and PAD tokens are prepended and appended similarly for the pos tags of those sentences to keep the sentence length consistent with its respective pos tags lenght. Then we create a vocabulary by adding all the sentences in train and dev set. We count the frequency of all the words and replace UNK token with those words whose word frequency is less than 3 so that the model can learn and generalise better and handle out of vocabulary (OOV) words.  
* for LSTM, vocab was created by adding all the sentences of the train and dev set and by handling oov words in the similar way as in FFNN. 

## Data Preparation
I used glove 6b, 100 dim word embeddings, downloaded it and stored it in a folder and got all the embeddings of all the words in the vocab. Created tag_to_ix list having all the pos tags with PAD being assigned 0 index and UNK the length of the list. 
Created a dataset class which created a context window of size (p+s+1) and padded it with PAD token if words were not there and oov words were assigned UNK tag. 

p=1, s=1 was used for fffnn model.

Finally, the code defines hyperparameters such as the batch size, number of epochs, and optimizer. It trains the model on the training dataset and evaluates its performance on the validation dataset. It uses several metrics such as accuracy, recall, precision, and F1 score to evaluate the model's performance.
