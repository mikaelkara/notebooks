<h1 align ="center"> Dynamic Prompting for task completion</h1>
<hr>

Recent papers such as [Do Prompt-Based Models Really Understand the Meaning of their Prompts?](https://arxiv.org/abs/2109.01247) and [What Makes Good In-Context Examples for GPT-3?](https://aclanthology.org/2022.deelio-1.10.pdf) have shown that using dynamic set of examples instead of fixed set of examples help GPT-3 to perfom the task with higher accuracy.


```python
# if needed, upgrade to the latest version of the OpenAI Python library
%pip install --upgrade openai
%pip install --upgrade torch
%pip install --upgrade sentence_transformers
%pip install --upgrade numpy
%pip install --upgrade datasets
%pip install --upgrade scikit-learn
%pip install --upgrade transformers
```


```python
# import os module & the OpenAI Python library for calling the OpenAI API
# please make sure you have installed required libraries via pip install -r requirements.txt
import os
import dotenv
import openai
from sentence_transformers import SentenceTransformer, util
import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
dotenv.load_dotenv()
```




    True



## Dataset Summary

The Text REtrieval Conference (TREC) Question Classification dataset is a dataset for question classification consisting of open-domain, fact-based questions divided into broad semantic categories. It contains 5500 labeled questions in training set and another 500 for test set.

The dataset has 6 coarse class labels and 50 fine class labels. Average length of each sentence is 10, vocabulary size of 8700.


```python
# load dataset from Huggingface's dataset library
dataset = load_dataset("trec")
```


```python
dataset
```




    DatasetDict({
        train: Dataset({
            features: ['text', 'coarse_label', 'fine_label'],
            num_rows: 5452
        })
        test: Dataset({
            features: ['text', 'coarse_label', 'fine_label'],
            num_rows: 500
        })
    })




```python
# name of the text and label column
label_type = "coarse_label"
text_key = "text"
# create mapping of ids2class and class2id
id2class = dict((i, label) for i, label in enumerate(dataset['train'].features[label_type].names))
class2id = dict((label, i) for i, label in enumerate(dataset['train'].features[label_type].names))
# create a dictionary with classes as key and containing all the training examples within that class
class2TrainDataset = dict((label, []) for label in dataset['train'].features[label_type].names)
for example in dataset['train']:
    label = id2class[example[label_type]]
    class2TrainDataset[label].append(example[text_key])
```

# Task Prompt


```python
# a prompt for asking LLM to perform a task
task_prompt = "As a Question Answering agent, your goal is to categorize questions into different semantic classes that impose constraints on potential answers, so that they can be utilized in later stages of the question answering process.\nFollowing are the semantic classes: ["
task_prompt += ", ".join([label for label in class2TrainDataset]) + "]"
# a prompt for asking LLM to generate the output for current task
query_prompt = "\nClassify the following question into one of the above classes. Please answer in a single word.\nquestion: "
answer_prompt = "\noutput: "
```

# Setup OpenAI API


```python
# Setting up the deployment name
deployment_name = os.environ['COMPLETIONS_MODEL']

# The base URL for your Azure OpenAI resource. e.g. "https://<your resource name>.openai.azure.com"
# This is the value of the endpoint for your Azure OpenAI resource
azure_endpoint = os.environ['AZURE_OPENAI_ENDPOINT']

# The API key for your Azure OpenAI resource.
api_key = os.environ['AZURE_OPENAI_API_KEY']

# Currently OPENAI API have the following versions available: https://learn.microsoft.com/azure/ai-services/openai/reference
api_version = os.environ['OPENAI_API_VERSION']

client = openai.AzureOpenAI(
  api_key=api_key,  
  azure_endpoint=azure_endpoint,
  api_version=api_version
)
```


```python
# Text completion using GPT
def trim_text(text):
    return text.strip().strip('\n').strip('\\n')

def generate_using_gpt(prompt):
    generated_sentence = ""
    try:
        # Create a completion for the provided prompt and parameters
        # To know more about the parameters, checkout this documentation: https://learn.microsoft.com/en-us/azure/cognitive-services/openai/reference
        response = client.completions.create(
            model=deployment_name,
            prompt=prompt, 
            max_tokens=3,
            temperature=0,
            top_p=1,
            stop=None,
            frequency_penalty=0,
            presence_penalty=0.0)
        
        choices = response.choices
        if len(choices) == 0 or not hasattr(choices[0], "text"):
            print("Text not generated properly")
        generated_sentence = choices[0].text.lstrip('\\n').rstrip('\\n').lstrip('\n\n').rstrip('\n\n').lstrip('\n').rstrip('\n')

    except openai.APITimeoutError as e:
        # Handle request timeout
        print(f"Request timed out: {e}")
    
    except openai.AuthenticationError as e:
        # Handle Authentication error here, e.g. invalid API key
        print(f"OpenAI API returned an Authentication Error: {e}")

    except openai.APIConnectionError as e:
        # Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")

    except openai.BadRequestError as e:
        # Handle connection error here
        print(f"Invalid Request Error: {e}")
        
    except openai.RateLimitError as e:
        # Handle rate limit error
        print(f"OpenAI API request exceeded rate limit: {e}")

    except openai.InternalServerError as e:
        # Handle Service Unavailable error
        print(f"Service Unavailable: {e}")

    except openai.APIError as e:
        # Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")

    return generated_sentence
```

# Zero-shot Prompt

### Example of the zero-shot prompt


```python
zeroshot_prompt = task_prompt +  query_prompt + dataset['test'][0][text_key] + answer_prompt
print(zeroshot_prompt)
```

    As a Question Answering agent, your goal is to categorize questions into different semantic classes that impose constraints on potential answers, so that they can be utilized in later stages of the question answering process.
    Following are the semantic classes: [ABBR, ENTY, DESC, HUM, LOC, NUM]
    Classify the following question into one of the above classes. Please answer in a single word.
    question: How far is it from Denver to Aspen ?
    output: 
    


```python
# prompt without any examples from the training dataset
labels = []
predictions = []
for example in dataset['test']:
    zeroshot_prompt = task_prompt +  query_prompt + example[text_key] + answer_prompt
    pred = generate_using_gpt(zeroshot_prompt)
    pred=trim_text(pred)
    labels.append(example[label_type])
    if pred not in class2id:
        predictions.append(-1)
    else:
        predictions.append(class2id[pred])
        
report = classification_report(labels, predictions) 
```


```python
print(report)
```

                  precision    recall  f1-score   support
    
               0       0.69      1.00      0.82         9
               1       0.30      0.69      0.42        94
               2       0.70      0.15      0.25       138
               3       0.88      0.35      0.51        65
               4       0.70      0.90      0.78        81
               5       0.84      0.80      0.82       113
    
        accuracy                           0.56       500
       macro avg       0.69      0.65      0.60       500
    weighted avg       0.68      0.56      0.54       500
    
    

# Few-shot Prompt


```python
# function to selection few examples in each of the classes from the training dataset
def generateFewshotPrompt(class2TrainDataset, N=3):
    fewshot_prompt = "\nFollowing are some examples."
    for label in class2TrainDataset:
        for example in class2TrainDataset[label][:N]:
            fewshot_prompt += "\nquestion: " + example
            fewshot_prompt += "\noutput: " + label
    return fewshot_prompt
```

### Example of the few-shot prompt 


```python
# prompt with one example in each of the classes
fewshot_examples = generateFewshotPrompt(class2TrainDataset, N=1)
fewshot_prompt = task_prompt +  fewshot_examples + query_prompt + dataset['test'][0][text_key] + answer_prompt
print(fewshot_prompt)
```

    As a Question Answering agent, your goal is to categorize questions into different semantic classes that impose constraints on potential answers, so that they can be utilized in later stages of the question answering process.
    Following are the semantic classes: [ABBR, ENTY, DESC, HUM, LOC, NUM]
    Following are some examples.
    question: What is the full form of .com ?
    output: ABBR
    question: What films featured the character Popeye Doyle ?
    output: ENTY
    question: How did serfdom develop in and then leave Russia ?
    output: DESC
    question: What contemptible scoundrel stole the cork from my lunch ?
    output: HUM
    question: What sprawling U.S. state boasts the most airports ?
    output: LOC
    question: When was Ozzy Osbourne born ?
    output: NUM
    Classify the following question into one of the above classes. Please answer in a single word.
    question: How far is it from Denver to Aspen ?
    output: 
    


```python
# prompt is created by adding one example in each of the classes 
labels = []
predictions = []
for example in dataset['test']:
    fewshot_prompt = task_prompt + fewshot_examples + query_prompt + example[text_key] + answer_prompt
    pred = generate_using_gpt(fewshot_prompt)
    pred=trim_text(pred)
    labels.append(example[label_type])
    if pred not in class2id:
        predictions.append(-1)
    else:
        predictions.append(class2id[pred])
        
report = classification_report(labels, predictions) 
```


```python
print(report)
```

                  precision    recall  f1-score   support
    
               0       0.75      1.00      0.86         9
               1       0.41      0.57      0.48        94
               2       0.80      0.51      0.62       138
               3       0.96      0.69      0.80        65
               4       0.93      0.93      0.93        81
               5       0.80      1.00      0.89       113
    
        accuracy                           0.73       500
       macro avg       0.77      0.78      0.76       500
    weighted avg       0.77      0.73      0.73       500
    
    

# Extract Embeddings for Training dataset


```python
# loading Sentence Transformer based model
model = SentenceTransformer('all-mpnet-base-v2', device=device)

# extract embeddings for a set of examples
def ExtractEmbeddings(examples):
    embedding_ls = []
    for example in examples:
        embedding = model.encode(example)     
        embedding_ls.append(embedding)
    return embedding_ls

# extract embeddings for all the training examples
class2TrainDatasetWithEmbedding = {}
for label in class2TrainDataset:
    embeddings = ExtractEmbeddings(class2TrainDataset[label])
    class2TrainDatasetWithEmbedding[label] = [class2TrainDataset[label], embeddings]
```

# Dynamic Few-shot Prompt


```python
# extract similar queries for a given input text from each of the classes
def getSimilarExamples(input_text, dataset, dataset_embedding):
    input_embedding = model.encode(input_text)
    sim_score = util.dot_score(input_embedding, dataset_embedding)[0]
    topN_ids = np.argsort(-sim_score)
    return [dataset[i] for i in topN_ids]
    
def getClasswiseSimilarExamples(input_text, class2TrainDatasetWithEmbedding):
    classwiseSimilarExamples = {}
    for label in class2TrainDataset:
        similarExamples = getSimilarExamples(input_text, class2TrainDatasetWithEmbedding[label][0], class2TrainDatasetWithEmbedding[label][1])
        classwiseSimilarExamples[label] = similarExamples
    return classwiseSimilarExamples
```


```python
# generate a prompt with similar examples in each of the classes
def generateDynamicPrompt(input_text, class2TrainDatasetWithEmbedding, N=3):
    classwiseSimilarExamples = getClasswiseSimilarExamples(input_text, class2TrainDatasetWithEmbedding)
    dynamic_prompt = "\nFollowing are some examples."
    for label in classwiseSimilarExamples:
        for example in classwiseSimilarExamples[label][:N]:
            dynamic_prompt += "\nquestion: " + example
            dynamic_prompt += "\noutput: " + label
    return dynamic_prompt
```

### Example of the dynamic prompt


```python
# dynamic prompt with one similar example in each of the classes
fewshot_examples = generateDynamicPrompt(dataset['test'][0][text_key], class2TrainDatasetWithEmbedding, N=1)
dynamic_prompt = task_prompt + fewshot_examples + query_prompt + dataset['test'][0][text_key] + answer_prompt
print(dynamic_prompt)
```


```python
labels = []
predictions = []
for example in dataset['test']:
    fewshot_examples = generateDynamicPrompt(example[text_key], class2TrainDatasetWithEmbedding, N=1)
    dynamic_prompt = task_prompt + fewshot_examples + query_prompt + example[text_key] + answer_prompt
    pred = generate_using_gpt(dynamic_prompt)
    pred=trim_text(pred)
    labels.append(example[label_type])
    if pred not in class2id:
        predictions.append(-1)
    else:
        predictions.append(class2id[pred])
        
report = classification_report(labels, predictions) 
```


```python
print(report)
```

                  precision    recall  f1-score   support
    
               0       0.69      1.00      0.82         9
               1       0.66      0.76      0.70        94
               2       0.88      0.71      0.79       138
               3       0.95      0.91      0.93        65
               4       0.94      0.90      0.92        81
               5       0.88      0.99      0.93       113
    
        accuracy                           0.84       500
       macro avg       0.83      0.88      0.85       500
    weighted avg       0.85      0.84      0.84       500
    
    
