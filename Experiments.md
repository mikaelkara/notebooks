# Swap LLMs and Validate App Performance

You're an AI Engineer recently tasked with switching to the latest LLM release? Follow this tutorial to ship your changes **confidently** with Literal AI ! 🚀


To show the experiments flow, we will go through these steps:
- [Setup](#setup)
- [Get dataset](#get-dataset)
- [Run experiment against `gpt-4o-mini`](#run-experiment)
   - [Load embedding model](#load-embedding-model)
   - [Create experiment](#create-experiment)
   - [Test each sample](#test-each-sample)

We deployed a chatbot to answer user questions about the Chainlit framework. In short, it's a **R**etrieval **A**ugmented **G**eneration (RAG) application, with access to the Chainlit [documentation](https://docs.chainlit.io/) and [cookbooks](https://github.com/Chainlit/cookbook). 


<div style="flex-direction: column; display: flex; justify-content: center; align-items: center">
    <figure style="margin-left: 150px; margin-right: 100px">
        <center>
            <figcaption><b>RAG chatbot in Chainlit</b></figcaption>
        </center>
        <br/>
        <img src="images/chainlit-rag-copilot.jpg" width="440"/>
    </figure>
    <figure>
        <center>
            <figcaption><b>Thread details in Literal AI</b></figcaption>
        </center>
        <br/>
        <img src="images/thread-details-rag.jpg" width="525"/>
    </figure>
</div>


We focus on the iteration loop to validate a parameter change we make on the application.<br/>
For instance, OpenAI released `gpt-4o-mini` yesterday. How do we confidently switch from `gpt-4o` to that cheaper version?

In this notebook, we will show the flow by checking how dissimilar `gpt-4o-mini` answers are from expected ground truths. <br/>
In a real-world scenario, you would have a few metrics you check against before shipping the change: context relevancy, answer similarity, latency, cost, latency, etc.

<a id="setup"></a>
## Setup

Get your [API key](https://docs.getliteral.ai/get-started/installation#how-to-get-my-api-key) and connect to Literal AI!

The below cell will prompt you for your `LITERAL_API_KEY` and create a `LiteralClient` which we will use to get our dataset and push the result of our experiments 🤗


```python
import os
import getpass

from literalai import LiteralClient

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("LITERAL_API_KEY")

literal_client = LiteralClient()
```

    LITERAL_API_KEY:  ········
    

<a id="get-dataset"></a>
## Get dataset

Here's what our dataset looks like on Literal AI. It contains:
1. the questions in the **Input** column
2. the answers (ground truths) in the **Output** column
3. the intermediate steps taken by our RAG agent in the dashed box

We will fetch the whole dataset, but focus on `input` and `output` for this tutorial.

<center>
    <img src="images/faq-dataset.jpg" width=80%/>
</center>



```python
# Adapt below to your own dataset
dataset = literal_client.api.get_dataset(name="Test dataset to ship RAG")

print(f"Number of samples in dataset = {len(dataset.items)}")
```

    Number of samples in dataset = 5
    

<a id="run-experiment"></a>
## Run experiment against `gpt-4o-mini`

<a id="load-embedding-model"></a>
### Load embedding model

We compute Answer Semantic Similarity using [gte-base-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) hosted on HuggingFace 🤗🤗

Check out the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) to pick the right embedding model for your task.


```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True);
```

<a id="create-experiment"></a>
### Create experiment

Let's start with creating a new experiment for our dataset. 

It's good practice to provide a meaningful name summarizing the changes you made. <br/>
In the `params` field, you can pass the exhaustive list of parameters that characterize the experiment you are about to run.



```python
experiment = dataset.create_experiment(
    name="Trial with gpt-4o-mini",
    params={ 
        "model": "gpt-4o-mini",
        "type": "output similarity", 
        "embedding-model": "Alibaba-NLP/gte-base-en-v1.5", 
        "commit": "830a6d1ee79e395e9cdcc487a6ec923887c29713"
    }
)
```

<a id="test-each-sample"></a>
### Test each sample

Simply loop on the dataset and for each entry:
- send the `question` to the locally modified version of our application (using `gpt-4o-mini`)
- compute the cosine similarity between **ground truth** and the **reached answer**
- log the resulting value as a score on our experiment!



```python
import requests
from tqdm import tqdm

for item in tqdm(dataset.items):
    question = item.input["content"]["args"][0]

    # Reached answer - based on locally modified version of the RAG application
    response = requests.get(f"http://localhost/app/{question}")
    answer = response.json()["answer"]
    answer_embedding = model.encode(answer)

    # Ground truth
    ground_truth = item.expected_output["content"]
    ground_truth_embedding = model.encode(ground_truth)

    similarity = float(cos_sim(answer_embedding, ground_truth_embedding))
    
    experiment.log({
        "datasetItemId": item.id,
        "scores": [ {
            "name": "Answer similarity",
            "type": "AI",
            "value": similarity
        } ],
        "input": { "question": question },
        "output": { "answer": answer }
    })

```

    100%|████████████████████████████████████████████████████████████████████████| 5/5 [00:26<00:00,  5.37s/it]

    Experiment finished and all 5 logged on Literal AI! 🎉
    

    
    

## Compare experiments on Literal AI 🎉🎉🎉

Here is the comparison between the `gpt-4o` and `gpt-4o-mini` experiments on Literal AI!

<center>
    <img src="images/experiments-comparison.jpg" width=80%/>
</center>


```python

```
