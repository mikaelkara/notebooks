# Detecting Issues in a Text Dataset with Cleanlab


Authored by: [Aravind Putrevu](https://huggingface.co/aravindputrevu)


In this 5-minute quickstart tutorial, we use Cleanlab to detect various issues in an intent classification dataset composed of (text) customer service requests at an online bank. We consider a subset of the [Banking77-OOS Dataset](https://arxiv.org/abs/2106.04564) containing 1,000 customer service requests which are classified into 10 categories based on their intent (you can run this same code on any text classification dataset). [Cleanlab](https://github.com/cleanlab/cleanlab) automatically identifies bad examples in our dataset, including mislabeled data, out-of-scope examples (outliers), or otherwise ambiguous examples. Consider filtering or correcting such bad examples before you dive deep into modeling your data!

**Overview of what we'll do in this tutorial:**

- Use a pretrained transformer model to extract the text embeddings from the customer service requests

- Train a simple Logistic Regression model on the text embeddings to compute out-of-sample predicted probabilities

- Run Cleanlab's `Datalab` audit with these predictions and embeddings in order to identify problems like: label issues, outliers, and near duplicates in the dataset.


## Quickstart

    
Already have (out-of-sample) `pred_probs` from a model trained on an existing set of labels? Maybe you have some numeric `features` as well? Run the code below to find any potential label errors in your dataset.

**Note:** If running on Colab, may want to use GPU (select: Runtime > Change runtime type > Hardware accelerator > GPU)



```python
from cleanlab import Datalab

lab = Datalab(data=your_dataset, label_name="column_name_of_labels")
lab.find_issues(pred_probs=your_pred_probs, features=your_features)

lab.report()
lab.get_issues()

```

## Install required dependencies


You can use `pip` to install all packages required for this tutorial as follows:



```python
!pip install -U scikit-learn sentence-transformers datasets
!pip install -U "cleanlab[datalab]"
```


```python
import re
import string
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

from cleanlab import Datalab
```


```python
import random
import numpy as np

pd.set_option("display.max_colwidth", None)

SEED = 123456  # for reproducibility
np.random.seed(SEED)
random.seed(SEED)
```

## Load and format the text dataset



```python
from datasets import load_dataset

dataset = load_dataset("PolyAI/banking77", split="train")
data = pd.DataFrame(dataset[:1000])
data.head()
```





  <div id="df-bb25ddea-7d53-4ee3-bdc9-92ba5f185022" class="colab-df-container">
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
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I am still waiting on my card?</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What can I do if my card still hasn't arrived after 2 weeks?</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I have been waiting over a week. Is the card still coming?</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Can I track my card while it is in the process of delivery?</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>How do I know if I will get my card, or if it is lost?</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-bb25ddea-7d53-4ee3-bdc9-92ba5f185022')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-bb25ddea-7d53-4ee3-bdc9-92ba5f185022 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-bb25ddea-7d53-4ee3-bdc9-92ba5f185022');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-51d80644-98e0-40b3-89a9-c326994791c6">
  <button class="colab-df-quickchart" onclick="quickchart('df-51d80644-98e0-40b3-89a9-c326994791c6')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-51d80644-98e0-40b3-89a9-c326994791c6 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
raw_texts, labels = data["text"].values, data["label"].values
num_classes = len(set(labels))

print(f"This dataset has {num_classes} classes.")
print(f"Classes: {set(labels)}")
```

    This dataset has 7 classes.
    Classes: {32, 34, 36, 11, 13, 46, 17}
    

Let's view the i-th example in the dataset:


```python
i = 1  # change this to view other examples from the dataset
print(f"Example Label: {labels[i]}")
print(f"Example Text: {raw_texts[i]}")
```

    Example Label: 11
    Example Text: What can I do if my card still hasn't arrived after 2 weeks?
    

The data is stored as two numpy arrays:

1. `raw_texts` stores the customer service requests utterances in text format
2. `labels` stores the intent categories (labels) for each example

<div class="alert alert-info">
Bringing Your Own Data (BYOD)?

You can easily replace the above with your own text dataset, and continue with the rest of the tutorial.

</div>

Next we convert the text strings into vectors better suited as inputs for our ML models.

We will use numeric representations from a pretrained Transformer model as embeddings of our text. The [Sentence Transformers](https://huggingface.co/docs/hub/sentence-transformers) library offers simple methods to compute these embeddings for text data. Here, we load the pretrained `electra-small-discriminator` model, and then run our data through network to extract a vector embedding of each example.


```python
transformer = SentenceTransformer('google/electra-small-discriminator')
text_embeddings = transformer.encode(raw_texts)
```

Our subsequent ML model will directly operate on elements of `text_embeddings` in order to classify the customer service requests.

## Define a classification model and compute out-of-sample predicted probabilities

A typical way to leverage pretrained networks for a particular classification task is to add a linear output layer and fine-tune the network parameters on the new data. However this can be computationally intensive. Alternatively, we can freeze the pretrained weights of the network and only train the output layer without having to rely on GPU(s). Here we do this conveniently by fitting a scikit-learn linear model on top of the extracted embeddings.

To identify label issues, cleanlab requires a probabilistic prediction from your model for each datapoint. However these predictions will be _overfit_ (and thus unreliable) for datapoints the model was previously trained on. cleanlab is intended to only be used with **out-of-sample** predicted class probabilities, i.e. on datapoints held-out from the model during the training.

Here we obtain out-of-sample predicted class probabilities for every example in our dataset using a Logistic Regression model with cross-validation.
Make sure that the columns of your `pred_probs` are properly ordered with respect to the ordering of classes, which for Datalab is: lexicographically sorted by class name.


```python
model = LogisticRegression(max_iter=400)

pred_probs = cross_val_predict(model, text_embeddings, labels, method="predict_proba")
```

## Use Cleanlab to find issues in your dataset

Given feature embeddings and the (out-of-sample) predicted class probabilities obtained from any model you have, cleanlab can quickly help you identify low-quality examples in your dataset.

Here, we use Cleanlab's `Datalab` to find issues in our data. Datalab offers several ways of loading the data; we’ll simply wrap the training features and noisy labels in a dictionary.


```python
data_dict = {"texts": raw_texts, "labels": labels}
```

All that is need to audit your data is to call `find_issues()`. We pass in the predicted probabilities and the feature embeddings obtained above, but you do not necessarily need to provide all of this information depending on which types of issues you are interested in. The more inputs you provide, the more types of issues `Datalab` can detect in your data. Using a better model to produce these inputs will ensure cleanlab more accurately estimates issues.


```python
lab = Datalab(data_dict, label_name="labels")
lab.find_issues(pred_probs=pred_probs, features=text_embeddings)
```

The output would look like:

```bash
Finding null issues ...
Finding label issues ...
Finding outlier issues ...
Fitting OOD estimator based on provided features ...
Finding near_duplicate issues ...
Finding non_iid issues ...
Finding class_imbalance issues ...
Finding underperforming_group issues ...

Audit complete. 62 issues found in the dataset.
```

After the audit is complete, review the findings using the `report` method:


```python
lab.report()
```

    Here is a summary of the different kinds of issues found in the data:
    
        issue_type  num_issues
           outlier          37
    near_duplicate          14
             label          10
           non_iid           1
    
    Dataset Information: num_examples: 1000, num_classes: 7
    
    
    ---------------------- outlier issues ----------------------
    
    About this issue:
    	Examples that are very different from the rest of the dataset 
        (i.e. potentially out-of-distribution or rare/anomalous instances).
        
    
    Number of examples with this issue: 37
    Overall dataset quality in terms of this issue: 0.3671
    
    Examples representing most severe instances of this issue:
         is_outlier_issue  outlier_score
    791              True       0.024866
    601              True       0.031162
    863              True       0.060738
    355              True       0.064199
    157              True       0.065075
    
    
    ------------------ near_duplicate issues -------------------
    
    About this issue:
    	A (near) duplicate issue refers to two or more examples in
        a dataset that are extremely similar to each other, relative
        to the rest of the dataset.  The examples flagged with this issue
        may be exactly duplicated, or lie atypically close together when
        represented as vectors (i.e. feature embeddings).
        
    
    Number of examples with this issue: 14
    Overall dataset quality in terms of this issue: 0.5961
    
    Examples representing most severe instances of this issue:
         is_near_duplicate_issue  near_duplicate_score near_duplicate_sets  distance_to_nearest_neighbor
    459                     True              0.009544               [429]                      0.000566
    429                     True              0.009544               [459]                      0.000566
    501                     True              0.046044          [412, 517]                      0.002781
    412                     True              0.046044               [501]                      0.002781
    698                     True              0.054626               [607]                      0.003314
    
    
    ----------------------- label issues -----------------------
    
    About this issue:
    	Examples whose given label is estimated to be potentially incorrect
        (e.g. due to annotation error) are flagged as having label issues.
        
    
    Number of examples with this issue: 10
    Overall dataset quality in terms of this issue: 0.9930
    
    Examples representing most severe instances of this issue:
         is_label_issue  label_score  given_label  predicted_label
    379           False     0.025486           32               11
    100           False     0.032102           11               36
    300           False     0.037742           32               46
    485            True     0.057666           17               34
    159            True     0.059408           13               11
    
    
    ---------------------- non_iid issues ----------------------
    
    About this issue:
    	Whether the dataset exhibits statistically significant
        violations of the IID assumption like:
        changepoints or shift, drift, autocorrelation, etc.
        The specific violation considered is whether the
        examples are ordered such that almost adjacent examples
        tend to have more similar feature values.
        
    
    Number of examples with this issue: 1
    Overall dataset quality in terms of this issue: 0.0000
    
    Examples representing most severe instances of this issue:
         is_non_iid_issue  non_iid_score
    988              True       0.563774
    975             False       0.570179
    997             False       0.571891
    967             False       0.572357
    956             False       0.577413
    
    Additional Information: 
    p-value: 0.0
    

### Label issues

The report indicates that cleanlab identified many label issues in our dataset. We can see which examples are flagged as likely mislabeled and the label quality score for each example using the `get_issues` method, specifying `label` as an argument to focus on label issues in the data.


```python
label_issues = lab.get_issues("label")
label_issues.head()
```





  <div id="df-5a26d9e1-9d7e-4327-9c38-a2fa12e59f28" class="colab-df-container">
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
      <th>is_label_issue</th>
      <th>label_score</th>
      <th>given_label</th>
      <th>predicted_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>0.903926</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>0.860544</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>0.658309</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>0.697085</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>0.434934</td>
      <td>11</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5a26d9e1-9d7e-4327-9c38-a2fa12e59f28')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-5a26d9e1-9d7e-4327-9c38-a2fa12e59f28 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5a26d9e1-9d7e-4327-9c38-a2fa12e59f28');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-4dd98a1c-b39c-464d-af07-2f385d2054b1">
  <button class="colab-df-quickchart" onclick="quickchart('df-4dd98a1c-b39c-464d-af07-2f385d2054b1')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-4dd98a1c-b39c-464d-af07-2f385d2054b1 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




| | is_label_issue | label_score | given_label | predicted_label |
|----------------|-------------|-------------|-----------------|-----------------|
| 0              | False       | 0.903926    | 11              | 11 |
| 1              | False       | 0.860544    | 11              | 11 |
| 2              | False       | 0.658309    | 11              | 11 |
| 3              | False       | 0.697085    | 11              | 11 |
| 4              | False       | 0.434934    | 11              | 11 |


This method returns a dataframe containing a label quality score for each example. These numeric scores lie between 0 and 1, where lower scores indicate examples more likely to be mislabeled. The dataframe also contains a boolean column specifying whether or not each example is identified to have a label issue (indicating it is likely mislabeled).

We can get the subset of examples flagged with label issues, and also sort by label quality score to find the indices of the 5 most likely mislabeled examples in our dataset.


```python
identified_label_issues = label_issues[label_issues["is_label_issue"] == True]
lowest_quality_labels = label_issues["label_score"].argsort()[:5].to_numpy()

print(
    f"cleanlab found {len(identified_label_issues)} potential label errors in the dataset.\n"
    f"Here are indices of the top 5 most likely errors: \n {lowest_quality_labels}"
)
```

    cleanlab found 10 potential label errors in the dataset.
    Here are indices of the top 5 most likely errors: 
     [379 100 300 485 159]
    

Let's review some of the most likely label errors.

Here we display the top 5 examples identified as the most likely label errors in the dataset, together with their given (original) label and a suggested alternative label from cleanlab.



```python
data_with_suggested_labels = pd.DataFrame(
    {"text": raw_texts, "given_label": labels, "suggested_label": label_issues["predicted_label"]}
)
data_with_suggested_labels.iloc[lowest_quality_labels]
```





  <div id="df-b3e89f50-5593-4907-a4ca-3b41c4a5e72c" class="colab-df-container">
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
      <th>text</th>
      <th>given_label</th>
      <th>suggested_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>379</th>
      <td>Is there a specific source that the exchange rate for the transfer I'm planning on making is pulled from?</td>
      <td>32</td>
      <td>11</td>
    </tr>
    <tr>
      <th>100</th>
      <td>can you share card tracking number?</td>
      <td>11</td>
      <td>36</td>
    </tr>
    <tr>
      <th>300</th>
      <td>If I need to cash foreign transfers, how does that work?</td>
      <td>32</td>
      <td>46</td>
    </tr>
    <tr>
      <th>485</th>
      <td>Was I charged more than I should of been for a currency exchange?</td>
      <td>17</td>
      <td>34</td>
    </tr>
    <tr>
      <th>159</th>
      <td>Is there any way to see my card in the app?</td>
      <td>13</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b3e89f50-5593-4907-a4ca-3b41c4a5e72c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-b3e89f50-5593-4907-a4ca-3b41c4a5e72c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b3e89f50-5593-4907-a4ca-3b41c4a5e72c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-8e310624-5a8c-4409-96e1-9757bd48d51a">
  <button class="colab-df-quickchart" onclick="quickchart('df-8e310624-5a8c-4409-96e1-9757bd48d51a')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-8e310624-5a8c-4409-96e1-9757bd48d51a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




  The output to the above command would like below:
  
|      | text                                                                                                      | given_label    | suggested_label |
|------|-----------------------------------------------------------------------------------------------------------|----------------|-----------------|
| 379  | Is there a specific source that the exchange rate for the transfer I'm planning on making is pulled from? | 32             | 11              |
| 100  | can you share card tracking number?                                                                       | 11             | 36              |
| 300  | If I need to cash foreign transfers, how does that work?                                                  | 32             | 46              |
| 485  | Was I charged more than I should of been for a currency exchange?                                         | 17             | 34              |
| 159  | Is there any way to see my card in the app?                                                               | 13             | 11              |


These are very clear label errors that cleanlab has identified in this data! Note that the `given_label` does not correctly reflect the intent of these requests, whoever produced this dataset made many mistakes that are important to address before modeling the data.

### Outlier issues

According to the report, our dataset contains some outliers.
We can see which examples are outliers (and a numeric quality score quantifying how typical each example appears to be) via `get_issues`. We sort the resulting DataFrame by cleanlab's outlier quality score to see the most severe outliers in our dataset.


```python
outlier_issues = lab.get_issues("outlier")
outlier_issues.sort_values("outlier_score").head()
```





  <div id="df-b302658a-5cf7-4fcc-a69d-150ef156f2ce" class="colab-df-container">
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
      <th>is_outlier_issue</th>
      <th>outlier_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>791</th>
      <td>True</td>
      <td>0.024866</td>
    </tr>
    <tr>
      <th>601</th>
      <td>True</td>
      <td>0.031162</td>
    </tr>
    <tr>
      <th>863</th>
      <td>True</td>
      <td>0.060738</td>
    </tr>
    <tr>
      <th>355</th>
      <td>True</td>
      <td>0.064199</td>
    </tr>
    <tr>
      <th>157</th>
      <td>True</td>
      <td>0.065075</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b302658a-5cf7-4fcc-a69d-150ef156f2ce')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-b302658a-5cf7-4fcc-a69d-150ef156f2ce button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b302658a-5cf7-4fcc-a69d-150ef156f2ce');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-6b51f708-0b0d-469d-b68a-156d487823c5">
  <button class="colab-df-quickchart" onclick="quickchart('df-6b51f708-0b0d-469d-b68a-156d487823c5')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-6b51f708-0b0d-469d-b68a-156d487823c5 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




Output would look like below:

|   | is_outlier_issue | outlier_score |
|---| ----------------|---------------|
| 791 | True             | 0.024866      |
| 601 | True             | 0.031162      |
| 863 | True             | 0.060738      |
| 355 | True             | 0.064199      |
| 157 | True             | 0.065075      |


```python
lowest_quality_outliers = outlier_issues["outlier_score"].argsort()[:5]

data.iloc[lowest_quality_outliers]
```





  <div id="df-30ae3a5c-a5ac-48e2-be07-1bc96f12dd24" class="colab-df-container">
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
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>791</th>
      <td>withdrawal pending meaning?</td>
      <td>46</td>
    </tr>
    <tr>
      <th>601</th>
      <td>$1 charge in transaction.</td>
      <td>34</td>
    </tr>
    <tr>
      <th>863</th>
      <td>My atm withdraw is stillpending</td>
      <td>46</td>
    </tr>
    <tr>
      <th>355</th>
      <td>explain the interbank exchange rate</td>
      <td>32</td>
    </tr>
    <tr>
      <th>157</th>
      <td>lost card found, want to put it back in app</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-30ae3a5c-a5ac-48e2-be07-1bc96f12dd24')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-30ae3a5c-a5ac-48e2-be07-1bc96f12dd24 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-30ae3a5c-a5ac-48e2-be07-1bc96f12dd24');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-bddbc2a7-3f55-4d4b-9f5b-bad055b943b6">
  <button class="colab-df-quickchart" onclick="quickchart('df-bddbc2a7-3f55-4d4b-9f5b-bad055b943b6')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-bddbc2a7-3f55-4d4b-9f5b-bad055b943b6 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




A sample output for the lowest quality outliers would look like below:

|index|text|label|
|---|---|---|
|791|withdrawal pending meaning?|46|
|601|$1 charge in transaction\.|34|
|863|My atm withdraw is stillpending|46|
|355|explain the interbank exchange rate|32|
|157|lost card found, want to put it back in app|13|


We see that cleanlab has identified entries in this dataset that do not appear to be proper customer requests. Outliers in this dataset appear to be out-of-scope customer requests and other nonsensical text which does not make sense for intent classification. Carefully consider whether such outliers may detrimentally affect your data modeling, and consider removing them from the dataset if so.

### Near-duplicate issues

According to the report, our dataset contains some sets of nearly duplicated examples.
We can see which examples are (nearly) duplicated (and a numeric quality score quantifying how dissimilar each example is from its nearest neighbor in the dataset) via `get_issues`. We sort the resulting DataFrame by cleanlab's near-duplicate quality score to see the text examples in our dataset that are most nearly duplicated.


```python
duplicate_issues = lab.get_issues("near_duplicate")
duplicate_issues.sort_values("near_duplicate_score").head()
```





  <div id="df-d081a3cf-0a30-4c25-9122-25edb4f5cb8d" class="colab-df-container">
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
      <th>is_near_duplicate_issue</th>
      <th>near_duplicate_score</th>
      <th>near_duplicate_sets</th>
      <th>distance_to_nearest_neighbor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>459</th>
      <td>True</td>
      <td>0.009544</td>
      <td>[429]</td>
      <td>0.000566</td>
    </tr>
    <tr>
      <th>429</th>
      <td>True</td>
      <td>0.009544</td>
      <td>[459]</td>
      <td>0.000566</td>
    </tr>
    <tr>
      <th>501</th>
      <td>True</td>
      <td>0.046044</td>
      <td>[412, 517]</td>
      <td>0.002781</td>
    </tr>
    <tr>
      <th>412</th>
      <td>True</td>
      <td>0.046044</td>
      <td>[501]</td>
      <td>0.002781</td>
    </tr>
    <tr>
      <th>698</th>
      <td>True</td>
      <td>0.054626</td>
      <td>[607]</td>
      <td>0.003314</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d081a3cf-0a30-4c25-9122-25edb4f5cb8d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-d081a3cf-0a30-4c25-9122-25edb4f5cb8d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d081a3cf-0a30-4c25-9122-25edb4f5cb8d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-87faf38c-7402-4ae8-b190-5cecbc434665">
  <button class="colab-df-quickchart" onclick="quickchart('df-87faf38c-7402-4ae8-b190-5cecbc434665')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-87faf38c-7402-4ae8-b190-5cecbc434665 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




The results above show which examples cleanlab considers nearly duplicated (rows where `is_near_duplicate_issue == True`). Here, we see that example 459 and 429 are nearly duplicated, as are example 501 and 412.

Let's view these examples to see how similar they are.


```python
data.iloc[[459, 429]]
```





  <div id="df-262b7208-d7d7-4fba-9d6c-7bda298822bc" class="colab-df-container">
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
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>459</th>
      <td>I purchased something abroad and the incorrect exchange rate was applied.</td>
      <td>17</td>
    </tr>
    <tr>
      <th>429</th>
      <td>I purchased something overseas and the incorrect exchange rate was applied.</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-262b7208-d7d7-4fba-9d6c-7bda298822bc')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-262b7208-d7d7-4fba-9d6c-7bda298822bc button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-262b7208-d7d7-4fba-9d6c-7bda298822bc');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-af242cbb-1eba-4356-88ea-3d159158902a">
  <button class="colab-df-quickchart" onclick="quickchart('df-af242cbb-1eba-4356-88ea-3d159158902a')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-af242cbb-1eba-4356-88ea-3d159158902a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




Sample output:

|index|text|label|
|---|---|---|
|459|I purchased something abroad and the incorrect exchange rate was applied\.|17|
|429|I purchased something overseas and the incorrect exchange rate was applied\.|17|


```python
data.iloc[[501, 412]]
```





  <div id="df-85c08b5b-da74-4092-97e6-f8908702eeaf" class="colab-df-container">
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
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>501</th>
      <td>The exchange rate you are using is really bad.This can't be the official interbank exchange rate.</td>
      <td>17</td>
    </tr>
    <tr>
      <th>412</th>
      <td>The exchange rate you are using is bad.This can't be the official interbank exchange rate.</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-85c08b5b-da74-4092-97e6-f8908702eeaf')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-85c08b5b-da74-4092-97e6-f8908702eeaf button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-85c08b5b-da74-4092-97e6-f8908702eeaf');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-74bfcb88-0e60-4aec-a441-90ce1d28370b">
  <button class="colab-df-quickchart" onclick="quickchart('df-74bfcb88-0e60-4aec-a441-90ce1d28370b')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-74bfcb88-0e60-4aec-a441-90ce1d28370b button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




Sample output:

|index|text|label|
|---|---|---|
|501|The exchange rate you are using is really bad\.This can't be the official interbank exchange rate\.|17|
|412|The exchange rate you are using is bad\.This can't be the official interbank exchange rate\.|17|

We see that these two sets of request are indeed very similar to one another! Including near duplicates in a dataset may have unintended effects on models, and be wary about splitting them across training/test sets. Learn more about handling near duplicates in a dataset from [the FAQ](https://docs.cleanlab.ai/stable/tutorials/faq.html#How-to-handle-near-duplicate-data-identified-by-Datalab?).

### Non-IID issues (data drift)
According to the report, our dataset does not appear to be Independent and Identically Distributed (IID).  The overall non-iid score for the dataset (displayed below) corresponds to the `p-value` of a statistical test for whether the ordering of samples in the dataset appears related to the similarity between their feature values.  A low `p-value` strongly suggests that the dataset violates the IID assumption, which is a key assumption required for conclusions (models) produced from the dataset to generalize to a larger population.


```python
p_value = lab.get_info('non_iid')['p-value']
p_value
```




    0.0



Here, our dataset was flagged as non-IID because the rows happened to be sorted by class label in the original data. This may be benign if we remember to shuffle rows before model training and data splitting. But if you don't know why your data was flagged as non-IID, then you should be worried about potential data drift or unexpected interactions between data points (their values may not be statistically independent). Think carefully about what future test data may look like (and whether your data is representative of the population you care about). You should not shuffle your data before the non-IID test runs (will invalidate its conclusions).

As demonstrated above, cleanlab can automatically shortlist the most likely issues in your dataset to help you better curate your dataset for subsequent modeling. With this shortlist, you can decide whether to fix these label issues or remove nonsensical or duplicated examples from your dataset to obtain a higher-quality dataset for training your next ML model. cleanlab's issue detection can be run with outputs from *any* type of model you initially trained.


### Cleanlab Opensource Project

[Cleanlab](https://github.com/cleanlab/cleanlab) is a standard Data-centric AI package designed to address data quality issues for messy, real-world data.

Do consider giving Cleanlab Github Repository a Star, and we welcome [contributions](https://github.com/cleanlab/cleanlab/issues?q=is:issue+is:open+label:%22good+first+issue%22) to the project.
