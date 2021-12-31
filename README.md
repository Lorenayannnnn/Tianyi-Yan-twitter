# Tianyi-Yan-twitter

## Personal Information:
1. First name: Tianyi (Lorena)
2. Last name: Yan
3. Email address: tianyiy@usc.edu

## Steps to run the project
1. Please put the data source file under the same directory where main.py and load_tweets_data.py are located to runt the project
2. Install used libraries (Please see reference at the end of this ReadMe file)
3. Please download GloVe word embeddings (glove.6B.50.txt) from https://nlp.stanford.edu/ and put it directly under the project directory
4. Run main.py


## Classification or regression:
I chose to do a classification, because since we are classifying all tweets based on valence values, which include 0, 2, 4 and are all discrete values, so I chose to treat it as a categorical value and thus do a classification task.

## Data preprocessing:
1. Use Pandas to load the csv data source file
2. Use Keras to tokenize the tweets
3. Use Keras to pad all tweet sequences 
4. Load pretrained word embeddings and get word embeddings for all words in each tweet. GloVe  word embeddings are used in this case.
5. Perform a simple one hot encoding for the valence target value. (Although there’s only 0s and 4s for valence in the data source file, I still converted 0, 2, and 4 to binary vectors and preprocess all valences as the final labels.)  

## Post preprocessing: 
 ---- | Column | Domain |Type
:----:|:-----:|:----:|:----:
Feature|tweet	|array of int (word embeddings)|array of int (word embeddings)
Target|valence|array of binary int (one-hot encoding)|array of binary int (one-hot encoding)

## Machine Learning model:
1. Type of model: supervised neural network
2. Input layer: 
	1. Embedding layer with size of length of word_index from the tokenizer (i.e., numbers of unique words that are tokenized), and with output size of embedding dimension of 50. 
	2. Reason: Need to convert word indices to GloVe word embeddings before further processing
3. Hidden layers: 
	1. Add several LSTM layers with size of embedding dimension of 50 (as it’s the output size of the embedding layer) with dropout rate of 0.2 and 0.5 and Tanh as the activation function.
	2. Reason: 
		- Use LSTM as we are processing tweets which involve sequencing and timing issue (sequence of the words).
		- Use dropout rate to avoid overfitting. 
		- Use Tanh to avoid vanishing gradient problem and also to facilitate converging speed to some extent 
4. Output layer: 
	1. Use a Dense layer with size of 3 and SoftMax as the activation function
	2. Reason:
		- Size of 3: because we have 3 categories for valence: 0, 2, and 4
		- SoftMax: since we have 3 different categories/classes, SoftMax would be useful to standardize the output. 

## Loss and Optimizer:
1. Loss: categorical_crossentropy
2. Optimizer: RMSprop

## Split data: 
I used train_test_split from sklearn to shuffle and split all data into training and test data, and then split all training data to train and data for validation. The ratio between train, validate, and test data is 3:1:1.


## Reference:
1. GloVe word embedding: https://nlp.stanford.edu/projects/glove/
2. Libraries used:
	- Pandas
	- Numpy
	- Tensor flow – Keras
	- Sklearn
