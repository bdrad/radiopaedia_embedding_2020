# Radiopaedia Embedding
Repository to accompany paper [cite here]

## Relevant Files
Note: Each of the following file descriptions will include an 'Optional Arguments' section if there are optional arguments that can be passed on its execution. If omitted, the values of these arguments will always default to the values we used in the manuscript.

### data_collection.py
Run this python file to collect Articles section from the Radiopedia website. It creates a folder 'data,' and places inside a csv file 'articles.csv,' where each row contains the article text and a list of system labels.

### train_glove.py
Run this python file to train custom word embeddings using the text scraped from Radiopedia. It generates four of these embeddings of the same dimensions used by Stanford NLP's glove6B model: 50d, 100d, 200d, 300d. It creates a folder 'custom' inside the 'vectors' folder [See Instructions for Running Locally] including a .txt representation for each of the custom word embeddings, identical to the glove6B format.

#### Optional Arguments
* epochs: The number of epochs used to train each embedding. Default: 25.

### evaluate.py
Run this python file to perform 5x2 cross validation on a Radiopedia article label classification model to compare results between the custom word embeddings and glove6B. This file automatically uses a gpu if one is available and visible [make sure to correctly set the CUDA_VISIBLE_DEVICES environment variable]. Otherwise, it uses the cpu. It creates a folder 'results' where it puts three files: analysis.pkl, a pandas dataframe containing info about each model output in the test set, results.pkl, a pandas dataframe containing info about model performance, and results.txt, a plaintext representation of results.pkl.

#### Optional Arguments
* batch_size: The batch size to use when training the classifier. Default: 32.
* val_split: The fraction of the training data to reserve for validation. Default: 0.05.
* min_freq: The minimum frequency required to include a token in the training vocabulary. Default: 1.
* num_epochs: The number of epochs on which to train the classifier. Default: 10.
* init_lr: The initial learning rate for training. Default: 0.05.
* step: The step size for applying learning rate decay. Default: 2.
* decay: The learning rate decay factor. Default: 0.9.
* replicate: Call this argument to use the same five seeds we used in the manuscript. Otherwise Omit.

## Instructions for Running Locally
* Clone this repository and install all the requirements.
* Install Python 3.6 or above
* Install everything in the requirements.txt file.
* Install Maciej Kula's implementation of the GloVe training pipeline in python (included in the requirements.txt file), either by cloning the [repository](https://github.com/maciejkula/glove-python) into this repo, or by installing from pypi: `pip install glove_python`.
* Download nltk punkt and stopwords. This can be done by manually installing them through the nltk downloader program or by running the following commands in any python interpreter on your system:  
    `nltk.download('punkt')`  
    `nltk.download('stopwords')`
* Download the glove.6B vectors [here](http://nlp.stanford.edu/data/glove.6B.zip), and extract the glove.6B folder into the repo directory.
* Run `DataCollection.py` to scrape the data from Radiopedia.
* Run `train_glove.py` to generate the custom word embeddings.
* Run `evaluate.py` to run the classifier and compare the custom embeddings with glove.6B.
* Download pre-trained Radiopaedia embeddings [here](https://bit.ly/3e0t4lQ)

## License
Embeddings and Analogy dataset:
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/3.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/3.0/">Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License</a>.

Other non dataset source code:
Licensed under GNU General Public License v3.0
