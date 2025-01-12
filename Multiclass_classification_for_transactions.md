# Multiclass Classification for Transactions

For this notebook we will be looking to classify a public dataset of transactions into a number of categories that we have predefined. These approaches should be replicable to any multiclass classification use case where we are trying to fit transactional data into predefined categories, and by the end of running through this you should have a few approaches for dealing with both labelled and unlabelled datasets.

The different approaches we'll be taking in this notebook are:
- **Zero-shot Classification:** First we'll do zero shot classification to put transactions in one of five named buckets using only a prompt for guidance
- **Classification with Embeddings:** Following this we'll create embeddings on a labelled dataset, and then use a traditional classification model to test their effectiveness at identifying our categories
- **Fine-tuned Classification:** Lastly we'll produce a fine-tuned model trained on our labelled dataset to see how this compares to the zero-shot and few-shot classification approaches

## Setup


```python
%load_ext autoreload
%autoreload
%pip install openai 'openai[datalib]' 'openai[embeddings]' transformers

```


```python
import openai
import pandas as pd
import numpy as np
import json
import os

COMPLETIONS_MODEL = "gpt-4"

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if you didn't set as an env var>"))
```

### Load dataset

We're using a public transaction dataset of transactions over £25k for the Library of Scotland. The dataset has three features that we'll be using:
- Supplier: The name of the supplier
- Description: A text description of the transaction
- Value: The value of the transaction in GBP

**Source**:

https://data.nls.uk/data/organisational-data/transactions-over-25k/


```python
transactions = pd.read_csv('./data/25000_spend_dataset_current.csv', encoding= 'unicode_escape')
len(transactions)

```




    359




```python
transactions.head()

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
      <th>Date</th>
      <th>Supplier</th>
      <th>Description</th>
      <th>Transaction value (£)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21/04/2016</td>
      <td>M &amp; J Ballantyne Ltd</td>
      <td>George IV Bridge Work</td>
      <td>35098.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26/04/2016</td>
      <td>Private Sale</td>
      <td>Literary &amp; Archival Items</td>
      <td>30000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30/04/2016</td>
      <td>City Of Edinburgh Council</td>
      <td>Non Domestic Rates</td>
      <td>40800.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>09/05/2016</td>
      <td>Computacenter Uk</td>
      <td>Kelvin Hall</td>
      <td>72835.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>09/05/2016</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>64361.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
def request_completion(prompt):

    completion_response = openai.chat.completions.create(
                            prompt=prompt,
                            temperature=0,
                            max_tokens=5,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0,
                            model=COMPLETIONS_MODEL)

    return completion_response

def classify_transaction(transaction,prompt):

    prompt = prompt.replace('SUPPLIER_NAME',transaction['Supplier'])
    prompt = prompt.replace('DESCRIPTION_TEXT',transaction['Description'])
    prompt = prompt.replace('TRANSACTION_VALUE',str(transaction['Transaction value (£)']))

    classification = request_completion(prompt).choices[0].message.content.replace('\n','')

    return classification

# This function takes your training and validation outputs from the prepare_data function of the Finetuning API, and
# confirms that each have the same number of classes.
# If they do not have the same number of classes the fine-tune will fail and return an error

def check_finetune_classes(train_file,valid_file):

    train_classes = set()
    valid_classes = set()
    with open(train_file, 'r') as json_file:
        json_list = list(json_file)
        print(len(json_list))

    for json_str in json_list:
        result = json.loads(json_str)
        train_classes.add(result['completion'])
        #print(f"result: {result['completion']}")
        #print(isinstance(result, dict))

    with open(valid_file, 'r') as json_file:
        json_list = list(json_file)
        print(len(json_list))

    for json_str in json_list:
        result = json.loads(json_str)
        valid_classes.add(result['completion'])
        #print(f"result: {result['completion']}")
        #print(isinstance(result, dict))

    if len(train_classes) == len(valid_classes):
        print('All good')

    else:
        print('Classes do not match, please prepare data again')

```

## Zero-shot Classification

We'll first assess the performance of the base models at classifying these transactions using a simple prompt. We'll provide the model with 5 categories and a catch-all of "Could not classify" for ones that it cannot place.


```python
zero_shot_prompt = '''You are a data expert working for the National Library of Scotland.
You are analysing all transactions over £25,000 in value and classifying them into one of five categories.
The five categories are Building Improvement, Literature & Archive, Utility Bills, Professional Services and Software/IT.
If you can't tell what it is, say Could not classify

Transaction:

Supplier: SUPPLIER_NAME
Description: DESCRIPTION_TEXT
Value: TRANSACTION_VALUE

The classification is:'''

```


```python
# Get a test transaction
transaction = transactions.iloc[0]

# Interpolate the values into the prompt
prompt = zero_shot_prompt.replace('SUPPLIER_NAME',transaction['Supplier'])
prompt = prompt.replace('DESCRIPTION_TEXT',transaction['Description'])
prompt = prompt.replace('TRANSACTION_VALUE',str(transaction['Transaction value (£)']))

# Use our completion function to return a prediction
completion_response = request_completion(prompt)
print(completion_response.choices[0].text)

```

     Building Improvement
    

Our first attempt is correct, M & J Ballantyne Ltd are a house builder and the work they performed is indeed Building Improvement.

Lets expand the sample size to 25 and see how it performs, again with just a simple prompt to guide it


```python
test_transactions = transactions.iloc[:25]
test_transactions['Classification'] = test_transactions.apply(lambda x: classify_transaction(x,zero_shot_prompt),axis=1)

```

    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    


```python
test_transactions['Classification'].value_counts()

```




     Building Improvement    14
     Could not classify       5
     Literature & Archive     3
     Software/IT              2
     Utility Bills            1
    Name: Classification, dtype: int64




```python
test_transactions.head(25)

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
      <th>Date</th>
      <th>Supplier</th>
      <th>Description</th>
      <th>Transaction value (£)</th>
      <th>Classification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21/04/2016</td>
      <td>M &amp; J Ballantyne Ltd</td>
      <td>George IV Bridge Work</td>
      <td>35098.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26/04/2016</td>
      <td>Private Sale</td>
      <td>Literary &amp; Archival Items</td>
      <td>30000.0</td>
      <td>Literature &amp; Archive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30/04/2016</td>
      <td>City Of Edinburgh Council</td>
      <td>Non Domestic Rates</td>
      <td>40800.0</td>
      <td>Utility Bills</td>
    </tr>
    <tr>
      <th>3</th>
      <td>09/05/2016</td>
      <td>Computacenter Uk</td>
      <td>Kelvin Hall</td>
      <td>72835.0</td>
      <td>Software/IT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>09/05/2016</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>64361.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>5</th>
      <td>09/05/2016</td>
      <td>A McGillivray</td>
      <td>Causewayside Refurbishment</td>
      <td>53690.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>6</th>
      <td>16/05/2016</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>365344.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>7</th>
      <td>23/05/2016</td>
      <td>Computacenter Uk</td>
      <td>Kelvin Hall</td>
      <td>26506.0</td>
      <td>Software/IT</td>
    </tr>
    <tr>
      <th>8</th>
      <td>23/05/2016</td>
      <td>ECG Facilities Service</td>
      <td>Facilities Management Charge</td>
      <td>32777.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>9</th>
      <td>23/05/2016</td>
      <td>ECG Facilities Service</td>
      <td>Facilities Management Charge</td>
      <td>32777.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>10</th>
      <td>30/05/2016</td>
      <td>ALDL</td>
      <td>ALDL Charges</td>
      <td>32317.0</td>
      <td>Could not classify</td>
    </tr>
    <tr>
      <th>11</th>
      <td>10/06/2016</td>
      <td>Wavetek Ltd</td>
      <td>Kelvin Hall</td>
      <td>87589.0</td>
      <td>Could not classify</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10/06/2016</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>381803.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>13</th>
      <td>28/06/2016</td>
      <td>ECG Facilities Service</td>
      <td>Facilities Management Charge</td>
      <td>32832.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>14</th>
      <td>30/06/2016</td>
      <td>Glasgow City Council</td>
      <td>Kelvin Hall</td>
      <td>1700000.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>15</th>
      <td>11/07/2016</td>
      <td>Wavetek Ltd</td>
      <td>Kelvin Hall</td>
      <td>65692.0</td>
      <td>Could not classify</td>
    </tr>
    <tr>
      <th>16</th>
      <td>11/07/2016</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>139845.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>17</th>
      <td>15/07/2016</td>
      <td>Sotheby'S</td>
      <td>Literary &amp; Archival Items</td>
      <td>28500.0</td>
      <td>Literature &amp; Archive</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18/07/2016</td>
      <td>Christies</td>
      <td>Literary &amp; Archival Items</td>
      <td>33800.0</td>
      <td>Literature &amp; Archive</td>
    </tr>
    <tr>
      <th>19</th>
      <td>25/07/2016</td>
      <td>A McGillivray</td>
      <td>Causewayside Refurbishment</td>
      <td>30113.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>20</th>
      <td>31/07/2016</td>
      <td>ALDL</td>
      <td>ALDL Charges</td>
      <td>32317.0</td>
      <td>Could not classify</td>
    </tr>
    <tr>
      <th>21</th>
      <td>08/08/2016</td>
      <td>ECG Facilities Service</td>
      <td>Facilities Management Charge</td>
      <td>32795.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>22</th>
      <td>15/08/2016</td>
      <td>Creative Video Productions Ltd</td>
      <td>Kelvin Hall</td>
      <td>26866.0</td>
      <td>Could not classify</td>
    </tr>
    <tr>
      <th>23</th>
      <td>15/08/2016</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>196807.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24/08/2016</td>
      <td>ECG Facilities Service</td>
      <td>Facilities Management Charge</td>
      <td>32795.0</td>
      <td>Building Improvement</td>
    </tr>
  </tbody>
</table>
</div>



Initial results are pretty good even with no labelled examples! The ones that it could not classify were tougher cases with few clues as to their topic, but maybe if we clean up the labelled dataset to give more examples we can get better performance.

## Classification with Embeddings

Lets create embeddings from the small set that we've classified so far - we've made a set of labelled examples by running the zero-shot classifier on 101 transactions from our dataset and manually correcting the 15 **Could not classify** results that we got

### Create embeddings

This initial section reuses the approach from the [Get_embeddings_from_dataset Notebook](Get_embeddings_from_dataset.ipynb) to create embeddings from a combined field concatenating all of our features


```python
df = pd.read_csv('./data/labelled_transactions.csv')
df.head()

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
      <th>Date</th>
      <th>Supplier</th>
      <th>Description</th>
      <th>Transaction value (£)</th>
      <th>Classification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15/08/2016</td>
      <td>Creative Video Productions Ltd</td>
      <td>Kelvin Hall</td>
      <td>26866</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29/05/2017</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>74806</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29/05/2017</td>
      <td>Morris &amp; Spottiswood Ltd</td>
      <td>George IV Bridge Work</td>
      <td>56448</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31/05/2017</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>164691</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24/07/2017</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>27926</td>
      <td>Building Improvement</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['combined'] = "Supplier: " + df['Supplier'].str.strip() + "; Description: " + df['Description'].str.strip() + "; Value: " + str(df['Transaction value (£)']).strip()
df.head(2)

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
      <th>Date</th>
      <th>Supplier</th>
      <th>Description</th>
      <th>Transaction value (£)</th>
      <th>Classification</th>
      <th>combined</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15/08/2016</td>
      <td>Creative Video Productions Ltd</td>
      <td>Kelvin Hall</td>
      <td>26866</td>
      <td>Other</td>
      <td>Supplier: Creative Video Productions Ltd; Desc...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29/05/2017</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>74806</td>
      <td>Building Improvement</td>
      <td>Supplier: John Graham Construction Ltd; Descri...</td>
    </tr>
  </tbody>
</table>
</div>




```python
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

df['n_tokens'] = df.combined.apply(lambda x: len(tokenizer.encode(x)))
len(df)

```




    101




```python
embedding_path = './data/transactions_with_embeddings_100.csv'

```


```python
from utils.embeddings_utils import get_embedding

df['babbage_similarity'] = df.combined.apply(lambda x: get_embedding(x, model='gpt-4'))
df['babbage_search'] = df.combined.apply(lambda x: get_embedding(x, model='gpt-4'))
df.to_csv(embedding_path)

```

### Use embeddings for classification

Now that we have our embeddings, let see if classifying these into the categories we've named gives us any more success.

For this we'll use a template from the [Classification_using_embeddings](Classification_using_embeddings.ipynb) notebook


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from ast import literal_eval

fs_df = pd.read_csv(embedding_path)
fs_df["babbage_similarity"] = fs_df.babbage_similarity.apply(literal_eval).apply(np.array)
fs_df.head()

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
      <th>Unnamed: 0</th>
      <th>Date</th>
      <th>Supplier</th>
      <th>Description</th>
      <th>Transaction value (£)</th>
      <th>Classification</th>
      <th>combined</th>
      <th>n_tokens</th>
      <th>babbage_similarity</th>
      <th>babbage_search</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>15/08/2016</td>
      <td>Creative Video Productions Ltd</td>
      <td>Kelvin Hall</td>
      <td>26866</td>
      <td>Other</td>
      <td>Supplier: Creative Video Productions Ltd; Desc...</td>
      <td>136</td>
      <td>[-0.009802100248634815, 0.022551486268639565, ...</td>
      <td>[-0.00232666521333158, 0.019198870286345482, 0...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>29/05/2017</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>74806</td>
      <td>Building Improvement</td>
      <td>Supplier: John Graham Construction Ltd; Descri...</td>
      <td>140</td>
      <td>[-0.009065819904208183, 0.012094118632376194, ...</td>
      <td>[0.005169447045773268, 0.00473341578617692, -0...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>29/05/2017</td>
      <td>Morris &amp; Spottiswood Ltd</td>
      <td>George IV Bridge Work</td>
      <td>56448</td>
      <td>Building Improvement</td>
      <td>Supplier: Morris &amp; Spottiswood Ltd; Descriptio...</td>
      <td>141</td>
      <td>[-0.009000026620924473, 0.02405017428100109, -...</td>
      <td>[0.0028343256562948227, 0.021166473627090454, ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>31/05/2017</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>164691</td>
      <td>Building Improvement</td>
      <td>Supplier: John Graham Construction Ltd; Descri...</td>
      <td>140</td>
      <td>[-0.009065819904208183, 0.012094118632376194, ...</td>
      <td>[0.005169447045773268, 0.00473341578617692, -0...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>24/07/2017</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>27926</td>
      <td>Building Improvement</td>
      <td>Supplier: John Graham Construction Ltd; Descri...</td>
      <td>140</td>
      <td>[-0.009065819904208183, 0.012094118632376194, ...</td>
      <td>[0.005169447045773268, 0.00473341578617692, -0...</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train, X_test, y_train, y_test = train_test_split(
    list(fs_df.babbage_similarity.values), fs_df.Classification, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
probas = clf.predict_proba(X_test)

report = classification_report(y_test, preds)
print(report)

```

                          precision    recall  f1-score   support
    
    Building Improvement       0.92      1.00      0.96        11
    Literature & Archive       1.00      1.00      1.00         3
                   Other       0.00      0.00      0.00         1
             Software/IT       1.00      1.00      1.00         1
           Utility Bills       1.00      1.00      1.00         5
    
                accuracy                           0.95        21
               macro avg       0.78      0.80      0.79        21
            weighted avg       0.91      0.95      0.93        21
    
    

    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

Performance for this model is pretty strong, so creating embeddings and using even a simpler classifier looks like an effective approach as well, with the zero-shot classifier helping us do the initial classification of the unlabelled dataset.

Lets take it one step further and see if a fine-tuned model trained on this same labelled datasets gives us comparable results

## Fine-tuned Transaction Classification

For this use case we're going to try to improve on the few-shot classification from above by training a fine-tuned model on the same labelled set of 101 transactions and applying this fine-tuned model on group of unseen transactions

### Building Fine-tuned Classifier

We'll need to do some data prep first to get our data ready. This will take the following steps:
- First we'll list out our classes and replace them with numeric identifiers. Making the model predict a single token rather than multiple consecutive ones like 'Building Improvement' should give us better results
- We also need to add a common prefix and suffix to each example to aid the model in making predictions - in our case our text is already started with 'Supplier' and we'll add a suffix of '\n\n###\n\n'
- Lastly we'll aid a leading whitespace onto each of our target classes for classification, again to aid the model


```python
ft_prep_df = fs_df.copy()
len(ft_prep_df)

```




    101




```python
ft_prep_df.head()

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
      <th>Unnamed: 0</th>
      <th>Date</th>
      <th>Supplier</th>
      <th>Description</th>
      <th>Transaction value (£)</th>
      <th>Classification</th>
      <th>combined</th>
      <th>n_tokens</th>
      <th>babbage_similarity</th>
      <th>babbage_search</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>15/08/2016</td>
      <td>Creative Video Productions Ltd</td>
      <td>Kelvin Hall</td>
      <td>26866</td>
      <td>Other</td>
      <td>Supplier: Creative Video Productions Ltd; Desc...</td>
      <td>12</td>
      <td>[-0.009630300104618073, 0.009887108579277992, ...</td>
      <td>[-0.008217384107410908, 0.025170527398586273, ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>29/05/2017</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>74806</td>
      <td>Building Improvement</td>
      <td>Supplier: John Graham Construction Ltd; Descri...</td>
      <td>16</td>
      <td>[-0.006144719664007425, -0.0018709596479311585...</td>
      <td>[-0.007424891460686922, 0.008475713431835175, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>29/05/2017</td>
      <td>Morris &amp; Spottiswood Ltd</td>
      <td>George IV Bridge Work</td>
      <td>56448</td>
      <td>Building Improvement</td>
      <td>Supplier: Morris &amp; Spottiswood Ltd; Descriptio...</td>
      <td>17</td>
      <td>[-0.005225738976150751, 0.015156379900872707, ...</td>
      <td>[-0.007611643522977829, 0.030322374776005745, ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>31/05/2017</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>164691</td>
      <td>Building Improvement</td>
      <td>Supplier: John Graham Construction Ltd; Descri...</td>
      <td>16</td>
      <td>[-0.006144719664007425, -0.0018709596479311585...</td>
      <td>[-0.007424891460686922, 0.008475713431835175, ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>24/07/2017</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>27926</td>
      <td>Building Improvement</td>
      <td>Supplier: John Graham Construction Ltd; Descri...</td>
      <td>16</td>
      <td>[-0.006144719664007425, -0.0018709596479311585...</td>
      <td>[-0.007424891460686922, 0.008475713431835175, ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
classes = list(set(ft_prep_df['Classification']))
class_df = pd.DataFrame(classes).reset_index()
class_df.columns = ['class_id','class']
class_df  , len(class_df)

```




    (   class_id                 class
     0         0  Literature & Archive
     1         1         Utility Bills
     2         2  Building Improvement
     3         3           Software/IT
     4         4                 Other,
     5)




```python
ft_df_with_class = ft_prep_df.merge(class_df,left_on='Classification',right_on='class',how='inner')

# Adding a leading whitespace onto each completion to help the model
ft_df_with_class['class_id'] = ft_df_with_class.apply(lambda x: ' ' + str(x['class_id']),axis=1)
ft_df_with_class = ft_df_with_class.drop('class', axis=1)

# Adding a common separator onto the end of each prompt so the model knows when a prompt is terminating
ft_df_with_class['prompt'] = ft_df_with_class.apply(lambda x: x['combined'] + '\n\n###\n\n',axis=1)
ft_df_with_class.head()

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
      <th>Unnamed: 0</th>
      <th>Date</th>
      <th>Supplier</th>
      <th>Description</th>
      <th>Transaction value (£)</th>
      <th>Classification</th>
      <th>combined</th>
      <th>n_tokens</th>
      <th>babbage_similarity</th>
      <th>babbage_search</th>
      <th>class_id</th>
      <th>prompt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>15/08/2016</td>
      <td>Creative Video Productions Ltd</td>
      <td>Kelvin Hall</td>
      <td>26866</td>
      <td>Other</td>
      <td>Supplier: Creative Video Productions Ltd; Desc...</td>
      <td>12</td>
      <td>[-0.009630300104618073, 0.009887108579277992, ...</td>
      <td>[-0.008217384107410908, 0.025170527398586273, ...</td>
      <td>4</td>
      <td>Supplier: Creative Video Productions Ltd; Desc...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51</td>
      <td>31/03/2017</td>
      <td>NLS Foundation</td>
      <td>Grant Payment</td>
      <td>177500</td>
      <td>Other</td>
      <td>Supplier: NLS Foundation; Description: Grant P...</td>
      <td>11</td>
      <td>[-0.022305507212877274, 0.008543581701815128, ...</td>
      <td>[-0.020519884303212166, 0.01993306167423725, -...</td>
      <td>4</td>
      <td>Supplier: NLS Foundation; Description: Grant P...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>70</td>
      <td>26/06/2017</td>
      <td>British Library</td>
      <td>Legal Deposit Services</td>
      <td>50056</td>
      <td>Other</td>
      <td>Supplier: British Library; Description: Legal ...</td>
      <td>11</td>
      <td>[-0.01019938476383686, 0.015277703292667866, -...</td>
      <td>[-0.01843327097594738, 0.03343546763062477, -0...</td>
      <td>4</td>
      <td>Supplier: British Library; Description: Legal ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>71</td>
      <td>24/07/2017</td>
      <td>ALDL</td>
      <td>Legal Deposit Services</td>
      <td>27067</td>
      <td>Other</td>
      <td>Supplier: ALDL; Description: Legal Deposit Ser...</td>
      <td>11</td>
      <td>[-0.008471488021314144, 0.004098685923963785, ...</td>
      <td>[-0.012966590002179146, 0.01299362163990736, 0...</td>
      <td>4</td>
      <td>Supplier: ALDL; Description: Legal Deposit Ser...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100</td>
      <td>24/07/2017</td>
      <td>AM Phillip</td>
      <td>Vehicle Purchase</td>
      <td>26604</td>
      <td>Other</td>
      <td>Supplier: AM Phillip; Description: Vehicle Pur...</td>
      <td>10</td>
      <td>[-0.003459023078903556, 0.004626389592885971, ...</td>
      <td>[-0.0010945454705506563, 0.008626140654087067,...</td>
      <td>4</td>
      <td>Supplier: AM Phillip; Description: Vehicle Pur...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# This step is unnecessary if you have a number of observations in each class
# In our case we don't, so we shuffle the data to give us a better chance of getting equal classes in our train and validation sets
# Our fine-tuned model will error if we have less classes in the validation set, so this is a necessary step

import random

labels = [x for x in ft_df_with_class['class_id']]
text = [x for x in ft_df_with_class['prompt']]
ft_df = pd.DataFrame(zip(text, labels), columns = ['prompt','class_id']) #[:300]
ft_df.columns = ['prompt','completion']
ft_df['ordering'] = ft_df.apply(lambda x: random.randint(0,len(ft_df)), axis = 1)
ft_df.set_index('ordering',inplace=True)
ft_df_sorted = ft_df.sort_index(ascending=True)
ft_df_sorted.head()

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
      <th>prompt</th>
      <th>completion</th>
    </tr>
    <tr>
      <th>ordering</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Supplier: Sothebys; Description: Literary &amp; Ar...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Supplier: Sotheby'S; Description: Literary &amp; A...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Supplier: City Of Edinburgh Council; Descripti...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Supplier: John Graham Construction Ltd; Descri...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Supplier: John Graham Construction Ltd; Descri...</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# This step is to remove any existing files if we've already produced training/validation sets for this classifier
#!rm transactions_grouped*

# We output our shuffled dataframe to a .jsonl file and run the prepare_data function to get us our input files
ft_df_sorted.to_json("transactions_grouped.jsonl", orient='records', lines=True)
!openai tools fine_tunes.prepare_data -f transactions_grouped.jsonl -q

```


```python
# This functions checks that your classes all appear in both prepared files
# If they don't, the fine-tuned model creation will fail
check_finetune_classes('transactions_grouped_prepared_train.jsonl','transactions_grouped_prepared_valid.jsonl')

```

    31
    8
    All good
    


```python
# This step creates your model
!openai api fine_tunes.create -t "transactions_grouped_prepared_train.jsonl" -v "transactions_grouped_prepared_valid.jsonl" --compute_classification_metrics --classification_n_classes 5 -m curie

# You can use following command to get fine tuning job status and model name, replace the job name with your job
#!openai api fine_tunes.get -i ft-YBIc01t4hxYBC7I5qhRF3Qdx

```


```python
# Congrats, you've got a fine-tuned model!
# Copy/paste the name provided into the variable below and we'll take it for a spin
fine_tuned_model = 'curie:ft-personal-2022-10-20-10-42-56'

```

### Applying Fine-tuned Classifier

Now we'll apply our classifier to see how it performs. We only had 31 unique observations in our training set and 8 in our validation set, so lets see how the performance is


```python
test_set = pd.read_json('transactions_grouped_prepared_valid.jsonl', lines=True)
test_set.head()

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
      <th>prompt</th>
      <th>completion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Supplier: Wavetek Ltd; Description: Kelvin Hal...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Supplier: ECG Facilities Service; Description:...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Supplier: M &amp; J Ballantyne Ltd; Description: G...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Supplier: Private Sale; Description: Literary ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Supplier: Ex Libris; Description: IT equipment...</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_set['predicted_class'] = test_set.apply(lambda x: openai.chat.completions.create(model=fine_tuned_model, prompt=x['prompt'], max_tokens=1, temperature=0, logprobs=5),axis=1)
test_set['pred'] = test_set.apply(lambda x : x['predicted_class']['choices'][0]['text'],axis=1)

```


```python
test_set['result'] = test_set.apply(lambda x: str(x['pred']).strip() == str(x['completion']).strip(), axis = 1)

```


```python
test_set['result'].value_counts()

```




    True     4
    False    4
    Name: result, dtype: int64



Performance is not great - unfortunately this is expected. With only a few examples of each class, the above approach with embeddings and a traditional classifier worked better.

A fine-tuned model works best with a great number of labelled observations. If we had a few hundred or thousand we may get better results, but lets do one last test on a holdout set to confirm that it doesn't generalise well to a new set of observations


```python
holdout_df = transactions.copy().iloc[101:]
holdout_df.head()

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
      <th>Date</th>
      <th>Supplier</th>
      <th>Description</th>
      <th>Transaction value (£)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>101</th>
      <td>23/10/2017</td>
      <td>City Building LLP</td>
      <td>Causewayside Refurbishment</td>
      <td>53147.0</td>
    </tr>
    <tr>
      <th>102</th>
      <td>30/10/2017</td>
      <td>ECG Facilities Service</td>
      <td>Facilities Management Charge</td>
      <td>35758.0</td>
    </tr>
    <tr>
      <th>103</th>
      <td>30/10/2017</td>
      <td>ECG Facilities Service</td>
      <td>Facilities Management Charge</td>
      <td>35758.0</td>
    </tr>
    <tr>
      <th>104</th>
      <td>06/11/2017</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>134208.0</td>
    </tr>
    <tr>
      <th>105</th>
      <td>06/11/2017</td>
      <td>ALDL</td>
      <td>Legal Deposit Services</td>
      <td>27067.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
holdout_df['combined'] = "Supplier: " + holdout_df['Supplier'].str.strip() + "; Description: " + holdout_df['Description'].str.strip() + '\n\n###\n\n' # + "; Value: " + str(df['Transaction value (£)']).strip()
holdout_df['prediction_result'] = holdout_df.apply(lambda x: openai.chat.completions.create(model=fine_tuned_model, prompt=x['combined'], max_tokens=1, temperature=0, logprobs=5),axis=1)
holdout_df['pred'] = holdout_df.apply(lambda x : x['prediction_result']['choices'][0]['text'],axis=1)

```


```python
holdout_df.head(10)

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
      <th>Date</th>
      <th>Supplier</th>
      <th>Description</th>
      <th>Transaction value (£)</th>
      <th>combined</th>
      <th>prediction_result</th>
      <th>pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>101</th>
      <td>23/10/2017</td>
      <td>City Building LLP</td>
      <td>Causewayside Refurbishment</td>
      <td>53147.0</td>
      <td>Supplier: City Building LLP; Description: Caus...</td>
      <td>{'id': 'cmpl-63YDadbYLo8xKsGY2vReOFCMgTOvG', '...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>102</th>
      <td>30/10/2017</td>
      <td>ECG Facilities Service</td>
      <td>Facilities Management Charge</td>
      <td>35758.0</td>
      <td>Supplier: ECG Facilities Service; Description:...</td>
      <td>{'id': 'cmpl-63YDbNK1D7UikDc3xi5ATihg5kQEt', '...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>103</th>
      <td>30/10/2017</td>
      <td>ECG Facilities Service</td>
      <td>Facilities Management Charge</td>
      <td>35758.0</td>
      <td>Supplier: ECG Facilities Service; Description:...</td>
      <td>{'id': 'cmpl-63YDbwfiHjkjMWsfTKNt6naeqPzOe', '...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>104</th>
      <td>06/11/2017</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>134208.0</td>
      <td>Supplier: John Graham Construction Ltd; Descri...</td>
      <td>{'id': 'cmpl-63YDbWAndtsRqPTi2ZHZtPodZvOwr', '...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>105</th>
      <td>06/11/2017</td>
      <td>ALDL</td>
      <td>Legal Deposit Services</td>
      <td>27067.0</td>
      <td>Supplier: ALDL; Description: Legal Deposit Ser...</td>
      <td>{'id': 'cmpl-63YDbDu7WM3svYWsRAMdDUKtSFDBu', '...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>106</th>
      <td>27/11/2017</td>
      <td>Maggs Bros Ltd</td>
      <td>Literary &amp; Archival Items</td>
      <td>26500.0</td>
      <td>Supplier: Maggs Bros Ltd; Description: Literar...</td>
      <td>{'id': 'cmpl-63YDbxNNI8ZH5CJJNxQ0IF9Zf925C', '...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>107</th>
      <td>30/11/2017</td>
      <td>Glasgow City Council</td>
      <td>Kelvin Hall</td>
      <td>42345.0</td>
      <td>Supplier: Glasgow City Council; Description: K...</td>
      <td>{'id': 'cmpl-63YDb8R1FWu4bjwM2xE775rouwneV', '...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>108</th>
      <td>11/12/2017</td>
      <td>ECG Facilities Service</td>
      <td>Facilities Management Charge</td>
      <td>35758.0</td>
      <td>Supplier: ECG Facilities Service; Description:...</td>
      <td>{'id': 'cmpl-63YDcAPsp37WhbPs9kwfUX0kBk7Hv', '...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>109</th>
      <td>11/12/2017</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>159275.0</td>
      <td>Supplier: John Graham Construction Ltd; Descri...</td>
      <td>{'id': 'cmpl-63YDcML2welrC3wF0nuKgcNmVu1oQ', '...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>110</th>
      <td>08/01/2018</td>
      <td>ECG Facilities Service</td>
      <td>Facilities Management Charge</td>
      <td>35758.0</td>
      <td>Supplier: ECG Facilities Service; Description:...</td>
      <td>{'id': 'cmpl-63YDc95SSdOHnIliFB2cjMEEm7Z2u', '...</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
holdout_df['pred'].value_counts()

```




     2    231
     0     27
    Name: pred, dtype: int64



Well those results were similarly underwhelming - so we've learned that with a dataset with a small number of labelled observations, either zero-shot classification or traditional classification with embeddings return better results than a fine-tuned model.

A fine-tuned model is still a great tool, but is more effective when you have a larger number of labelled examples for each class that you're looking to classify
