##### Copyright 2024 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Classify text with embeddings


<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/Classify_text_with_embeddings.ipynb"><img src="../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>

## Overview

In this notebook, you'll learn to use the embeddings produced by the Gemini API to train a model that can classify different types of newsgroup posts based on the topic.



```
!pip install -U -q "google-generativeai>=0.7.2"
```


```
import re
import tqdm
import keras
import numpy as np
import pandas as pd

import google.generativeai as genai

from google.colab import userdata

import seaborn as sns
import matplotlib.pyplot as plt

from keras import layers
from matplotlib.ticker import MaxNLocator
from sklearn.datasets import fetch_20newsgroups
import sklearn.metrics as skmetrics
```

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) quickstart for an example.


```
API_KEY=userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=API_KEY)
```

## Dataset

The [20 Newsgroups Text Dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html){:.external} contains 18,000 newsgroups posts on 20 topics divided into training and test sets. The split between the training and test datasets are based on messages posted before and after a specific date. For this tutorial, you will be using the subsets of the training and test datasets. You will preprocess and organize the data into Pandas dataframes.


```
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# View list of class names for dataset
newsgroups_train.target_names
```




    ['alt.atheism',
     'comp.graphics',
     'comp.os.ms-windows.misc',
     'comp.sys.ibm.pc.hardware',
     'comp.sys.mac.hardware',
     'comp.windows.x',
     'misc.forsale',
     'rec.autos',
     'rec.motorcycles',
     'rec.sport.baseball',
     'rec.sport.hockey',
     'sci.crypt',
     'sci.electronics',
     'sci.med',
     'sci.space',
     'soc.religion.christian',
     'talk.politics.guns',
     'talk.politics.mideast',
     'talk.politics.misc',
     'talk.religion.misc']



Here is an example of what a data point from the training set looks like.


```
idx = newsgroups_train.data[0].index('Lines')
print(newsgroups_train.data[0][idx:])
```

    Lines: 15
    
     I was wondering if anyone out there could enlighten me on this car I saw
    the other day. It was a 2-door sports car, looked to be from the late 60s/
    early 70s. It was called a Bricklin. The doors were really small. In addition,
    the front bumper was separate from the rest of the body. This is 
    all I know. If anyone can tellme a model name, engine specs, years
    of production, where this car is made, history, or whatever info you
    have on this funky looking car, please e-mail.
    
    Thanks,
    - IL
       ---- brought to you by your neighborhood Lerxst ----
    
    
    
    
    
    

Now you will begin preprocessing the data for this tutorial. Remove any sensitive information like names, email, or redundant parts of the text like `"From: "` and `"\nSubject: "`. Organize the information into a Pandas dataframe so it is more readable.


```
def preprocess_newsgroup_data(newsgroup_dataset):
  # Apply functions to remove names, emails, and extraneous words from data points in newsgroups.data
  newsgroup_dataset.data = [re.sub(r'[\w\.-]+@[\w\.-]+', '', d) for d in newsgroup_dataset.data] # Remove email
  newsgroup_dataset.data = [re.sub(r"\([^()]*\)", "", d) for d in newsgroup_dataset.data] # Remove names
  newsgroup_dataset.data = [d.replace("From: ", "") for d in newsgroup_dataset.data] # Remove "From: "
  newsgroup_dataset.data = [d.replace("\nSubject: ", "") for d in newsgroup_dataset.data] # Remove "\nSubject: "

  # Cut off each text entry after 5,000 characters
  newsgroup_dataset.data = [d[0:5000] if len(d) > 5000 else d for d in newsgroup_dataset.data]

  # Put data points into dataframe
  df_processed = pd.DataFrame(newsgroup_dataset.data, columns=['Text'])
  df_processed['Label'] = newsgroup_dataset.target
  # Match label to target name index
  df_processed['Class Name'] = ''
  for idx, row in df_processed.iterrows():
    df_processed.at[idx, 'Class Name'] = newsgroup_dataset.target_names[row['Label']]

  return df_processed
```


```
# Apply preprocessing function to training and test datasets
df_train = preprocess_newsgroup_data(newsgroups_train)
df_test = preprocess_newsgroup_data(newsgroups_test)

df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Text</th>
      <th>Label</th>
      <th>Class Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>WHAT car is this!?\nNntp-Posting-Host: rac3.w...</td>
      <td>7</td>
      <td>rec.autos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SI Clock Poll - Final Call\nSummary: Final ca...</td>
      <td>4</td>
      <td>comp.sys.mac.hardware</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PB questions...\nOrganization: Purdue Univers...</td>
      <td>4</td>
      <td>comp.sys.mac.hardware</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Re: Weitek P9000 ?\nOrganization: Harris Comp...</td>
      <td>1</td>
      <td>comp.graphics</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Re: Shuttle Launch Question\nOrganization: Sm...</td>
      <td>14</td>
      <td>sci.space</td>
    </tr>
  </tbody>
</table>
</div>



Next, you will sample some of the data by taking 100 data points in the training dataset, and dropping a few of the categories to run through this tutorial. Choose the science categories to compare.


```
def sample_data(df, num_samples, classes_to_keep):
  df = df.groupby('Label', as_index = False).apply(lambda x: x.sample(num_samples)).reset_index(drop=True)

  df = df[df['Class Name'].str.contains(classes_to_keep)]

  # Reset the encoding of the labels after sampling and dropping certain categories
  df['Class Name'] = df['Class Name'].astype('category')
  df['Encoded Label'] = df['Class Name'].cat.codes

  return df
```


```
TRAIN_NUM_SAMPLES = 100
TEST_NUM_SAMPLES = 25
CLASSES_TO_KEEP = 'sci' # Class name should contain 'sci' in it to keep science categories
df_train = sample_data(df_train, TRAIN_NUM_SAMPLES, CLASSES_TO_KEEP)
df_test = sample_data(df_test, TEST_NUM_SAMPLES, CLASSES_TO_KEEP)
```


```
df_train.value_counts('Class Name')
```




    Class Name
    sci.crypt          100
    sci.electronics    100
    sci.med            100
    sci.space          100
    Name: count, dtype: int64




```
df_test.value_counts('Class Name')
```




    Class Name
    sci.crypt          25
    sci.electronics    25
    sci.med            25
    sci.space          25
    Name: count, dtype: int64



## Create the embeddings

In this section, you will see how to generate embeddings for a piece of text using the embeddings from the Gemini API. To learn more about embeddings, visit the [embeddings guide](https://ai.google.dev/docs/embeddings_guide).

**NOTE**: Embeddings are computed one at a time, large sample sizes can take a long time!

### API changes to Embeddings embedding-001

For the new embeddings model, there is a new task type parameter and the optional title (only valid with task_type=`RETRIEVAL_DOCUMENT`).

These new parameters apply only to the newest embeddings models.The task types are:

Task Type | Description
---       | ---
RETRIEVAL_QUERY	| Specifies the given text is a query in a search/retrieval setting.
RETRIEVAL_DOCUMENT | Specifies the given text is a document in a search/retrieval setting.
SEMANTIC_SIMILARITY	| Specifies the given text will be used for Semantic Textual Similarity (STS).
CLASSIFICATION	| Specifies that the embeddings will be used for classification.
CLUSTERING	| Specifies that the embeddings will be used for clustering.


```
from tqdm.auto import tqdm
tqdm.pandas()

from google.api_core import retry

def make_embed_text_fn(model):

  @retry.Retry(timeout=300.0)
  def embed_fn(text: str) -> list[float]:
    # Set the task_type to CLASSIFICATION.
    embedding = genai.embed_content(model=model,
                                    content=text,
                                    task_type="classification")
    return embedding['embedding']

  return embed_fn

def create_embeddings(model, df):
  df['Embeddings'] = df['Text'].progress_apply(make_embed_text_fn(model))
  return df
```


```
model = 'models/embedding-001'
df_train = create_embeddings(model, df_train)
df_test = create_embeddings(model, df_test)
```


      0%|          | 0/400 [00:00<?, ?it/s]



      0%|          | 0/100 [00:00<?, ?it/s]



```
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Text</th>
      <th>Label</th>
      <th>Class Name</th>
      <th>Encoded Label</th>
      <th>Embeddings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1100</th>
      <td>freely distributable public key cryptography ...</td>
      <td>11</td>
      <td>sci.crypt</td>
      <td>0</td>
      <td>[0.03072634, -0.023314659, -0.094281256, -0.09...</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>Can DES code be shipped to Canada?\nArticle-I...</td>
      <td>11</td>
      <td>sci.crypt</td>
      <td>0</td>
      <td>[0.037030637, -0.036586225, -0.043221463, -0.0...</td>
    </tr>
    <tr>
      <th>1102</th>
      <td>"Clipper" an Infringement on Intergraph's Nam...</td>
      <td>11</td>
      <td>sci.crypt</td>
      <td>0</td>
      <td>[0.023385575, 0.0010962725, -0.0868093, -0.053...</td>
    </tr>
    <tr>
      <th>1103</th>
      <td>Re: Secret algorithm [Re: Clipper Chip and cr...</td>
      <td>11</td>
      <td>sci.crypt</td>
      <td>0</td>
      <td>[0.006500222, -0.021546165, -0.079286404, -0.0...</td>
    </tr>
    <tr>
      <th>1104</th>
      <td>(Stephan Neuhaus )Re: Do we need the clipper ...</td>
      <td>11</td>
      <td>sci.crypt</td>
      <td>0</td>
      <td>[-0.004582751, -0.05328875, -0.064975366, -0.0...</td>
    </tr>
  </tbody>
</table>
</div>



## Build a simple classification model
Here you will define a simple model with one hidden layer and a single class probability output. The prediction will correspond to the probability of a piece of text being a particular class of news. When you build your model, Keras will automatically shuffle the data points.


```
def build_classification_model(input_size: int, num_classes: int) -> keras.Model:
  inputs = x = keras.Input(input_size)
  x = layers.Dense(input_size, activation='relu')(x)
  x = layers.Dense(num_classes, activation='sigmoid')(x)
  return keras.Model(inputs=[inputs], outputs=x)
```


```
# Derive the embedding size from the first training element.
embedding_size = len(df_train['Embeddings'].iloc[0])

# Give your model a different name, as you have already used the variable name 'model'
classifier = build_classification_model(embedding_size, len(df_train['Class Name'].unique()))
classifier.summary()

classifier.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   optimizer = keras.optimizers.Adam(learning_rate=0.001),
                   metrics=['accuracy'])
```

    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_1 (InputLayer)        [(None, 768)]             0         
                                                                     
     dense (Dense)               (None, 768)               590592    
                                                                     
     dense_1 (Dense)             (None, 4)                 3076      
                                                                     
    =================================================================
    Total params: 593,668
    Trainable params: 593,668
    Non-trainable params: 0
    _________________________________________________________________
    

    2024-05-31 19:53:59.716281: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/lib64:/usr/local/nccl2/lib:/usr/local/cuda/extras/CUPTI/lib64
    2024-05-31 19:53:59.716323: W tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:265] failed call to cuInit: UNKNOWN ERROR (303)
    2024-05-31 19:53:59.716345: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (lucianomartins-io24labs): /proc/driver/nvidia/version does not exist
    2024-05-31 19:53:59.718205: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F AVX512_VNNI FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    


```
embedding_size
```




    768



## Train the model to classify newsgroups

Finally, you can train a simple model. Use a small number of epochs to avoid overfitting. The first epoch takes much longer than the rest, because the embeddings need to be computed only once.


```
NUM_EPOCHS = 20
BATCH_SIZE = 32

# Split the x and y components of the train and validation subsets.
y_train = df_train['Encoded Label']
x_train = np.stack(df_train['Embeddings'])
y_val = df_test['Encoded Label']
x_val = np.stack(df_test['Embeddings'])

# Train the model for the desired number of epochs.
callback = keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)

history = classifier.fit(x=x_train,
                         y=y_train,
                         validation_data=(x_val, y_val),
                         callbacks=[callback],
                         batch_size=BATCH_SIZE,
                         epochs=NUM_EPOCHS,)
```

    Epoch 1/20
    

    /opt/conda/lib/python3.10/site-packages/keras/backend.py:5585: UserWarning: "`sparse_categorical_crossentropy` received `from_logits=True`, but the `output` argument was produced by a Softmax activation and thus does not represent logits. Was this intended?
      output, from_logits = _get_logits(
    

    13/13 [==============================] - 1s 17ms/step - loss: 1.1849 - accuracy: 0.7575 - val_loss: 0.9315 - val_accuracy: 0.9300
    Epoch 2/20
    13/13 [==============================] - 0s 5ms/step - loss: 0.6942 - accuracy: 0.9625 - val_loss: 0.5221 - val_accuracy: 0.9500
    Epoch 3/20
    13/13 [==============================] - 0s 5ms/step - loss: 0.3545 - accuracy: 0.9650 - val_loss: 0.2969 - val_accuracy: 0.9700
    Epoch 4/20
    13/13 [==============================] - 0s 5ms/step - loss: 0.1971 - accuracy: 0.9750 - val_loss: 0.2094 - val_accuracy: 0.9500
    Epoch 5/20
    13/13 [==============================] - 0s 5ms/step - loss: 0.1255 - accuracy: 0.9775 - val_loss: 0.1555 - val_accuracy: 0.9600
    Epoch 6/20
    13/13 [==============================] - 0s 5ms/step - loss: 0.0935 - accuracy: 0.9850 - val_loss: 0.1488 - val_accuracy: 0.9400
    Epoch 7/20
    13/13 [==============================] - 0s 5ms/step - loss: 0.0710 - accuracy: 0.9900 - val_loss: 0.1109 - val_accuracy: 0.9900
    Epoch 8/20
    13/13 [==============================] - 0s 5ms/step - loss: 0.0535 - accuracy: 0.9975 - val_loss: 0.1133 - val_accuracy: 0.9800
    Epoch 9/20
    13/13 [==============================] - 0s 5ms/step - loss: 0.0444 - accuracy: 0.9950 - val_loss: 0.0896 - val_accuracy: 0.9900
    Epoch 10/20
    13/13 [==============================] - 0s 5ms/step - loss: 0.0346 - accuracy: 1.0000 - val_loss: 0.0902 - val_accuracy: 0.9800
    Epoch 11/20
    13/13 [==============================] - 0s 5ms/step - loss: 0.0269 - accuracy: 1.0000 - val_loss: 0.0783 - val_accuracy: 0.9900
    Epoch 12/20
    13/13 [==============================] - 0s 5ms/step - loss: 0.0225 - accuracy: 1.0000 - val_loss: 0.0780 - val_accuracy: 0.9800
    Epoch 13/20
    13/13 [==============================] - 0s 5ms/step - loss: 0.0189 - accuracy: 1.0000 - val_loss: 0.0734 - val_accuracy: 0.9800
    

## Evaluate model performance

Use Keras <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate"><code>Model.evaluate</code></a> to get the loss and accuracy on the test dataset.


```
classifier.evaluate(x=x_val, y=y_val, return_dict=True)
```

    4/4 [==============================] - 0s 3ms/step - loss: 0.0734 - accuracy: 0.9800
    




    {'loss': 0.07338692992925644, 'accuracy': 0.9800000190734863}



One way to evaluate your model performance is to visualize the classifier performance. Use `plot_history` to see the loss and accuracy trends over the epochs.


```
def plot_history(history):
  """
    Plotting training and validation learning curves.

    Args:
      history: model history with all the metric measures
  """
  fig, (ax1, ax2) = plt.subplots(1,2)
  fig.set_size_inches(20, 8)

  # Plot loss
  ax1.set_title('Loss')
  ax1.plot(history.history['loss'], label = 'train')
  ax1.plot(history.history['val_loss'], label = 'test')
  ax1.set_ylabel('Loss')

  ax1.set_xlabel('Epoch')
  ax1.legend(['Train', 'Validation'])

  # Plot accuracy
  ax2.set_title('Accuracy')
  ax2.plot(history.history['accuracy'],  label = 'train')
  ax2.plot(history.history['val_accuracy'], label = 'test')
  ax2.set_ylabel('Accuracy')
  ax2.set_xlabel('Epoch')
  ax2.legend(['Train', 'Validation'])

  plt.show()

plot_history(history)
```


    
![png](output_35_0.png)
    


Another way to view model performance, beyond just measuring loss and accuracy is to use a confusion matrix. The confusion matrix allows you to assess the performance of the classification model beyond accuracy. You can see what misclassified points get classified as. In order to build the confusion matrix for this multi-class classification problem, get the actual values in the test set and the predicted values.

Start by generating the predicted class for each example in the validation set using [`Model.predict()`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict).


```
y_hat = classifier.predict(x=x_val)
y_hat = np.argmax(y_hat, axis=1)
```

    4/4 [==============================] - 0s 2ms/step
    


```
labels_dict = dict(zip(df_test['Class Name'], df_test['Encoded Label']))
labels_dict
```




    {'sci.crypt': 0, 'sci.electronics': 1, 'sci.med': 2, 'sci.space': 3}




```
cm = skmetrics.confusion_matrix(y_val, y_hat)
disp = skmetrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels_dict.keys())
disp.plot(xticks_rotation='vertical')
plt.title('Confusion matrix for newsgroup test dataset');
plt.grid(False)
```


    
![png](output_39_0.png)
    

