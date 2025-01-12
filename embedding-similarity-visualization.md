```
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Visualizing embedding similarity from text documents using t-SNE plots

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/embeddings/embedding-similarity-visualization.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/embeddings/embedding-similarity-visualization.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/embeddings/embedding-similarity-visualization.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Gabe Rives-Corbett](https://github.com/grivescorbett) |

This notebook demonstrates how vector similarity is relevant to LLM-generated embeddings. You will embed a collection of labelled documents and then plot the embeddings on a two-dimensional t-SNE plot to observe how similar documents tend to cluster together based on their embeddings.

## Getting started

### Install libraries


```
%pip install --user langchain==0.0.315 \
                    google-cloud-aiplatform==1.35.0 \
                    scikit-learn==1.3.1
```

### Restart current runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which will restart the current kernel.


```
# Restart kernel after installs so that your environment can access the new packages
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Please wait until it is finished before continuing to the next step. ⚠️</b>
</div>


### Authenticate your notebook environment (Colab only)

If you are running this notebook on Google Colab, you will need to authenticate your environment. To do this, run the new cell below. This step is not required if you are using [Vertex AI Workbench](https://cloud.google.com/vertex-ai-workbench).


```
import sys

if "google.colab" in sys.modules:
    # Define project information
    PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
    LOCATION = "us-central1"  # @param {type:"string"}

    # Authenticate user to Google Cloud
    from google.colab import auth

    auth.authenticate_user()
```

### Import libraries


```
import re

from google.api_core import retry
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from vertexai.language_models import TextEmbeddingModel

tqdm.pandas()
```

## Fetch and clean the data

In this example, you will use the open source [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) dataset, a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups


```
categories = ["comp.graphics", "sci.space", "sci.med", "rec.sport.hockey"]
newsgroups = fetch_20newsgroups(categories=categories)
```


```
raw_data = pd.DataFrame()
raw_data["text"] = newsgroups.data
raw_data["target"] = [newsgroups.target_names[x] for x in newsgroups.target]
```

Because of the 8k input token limit, in this example you will exclude all documents that have a length outside this limit.

Even though tokens typically are >=1 characters, for simplicity, you can just filter for documents that have <= 8000 _characters_.


```
filtered = raw_data.loc[raw_data["text"].str.len() <= 8000]
```

Subsample the dataset into 500 data points, stratified on the label


```
x_subsample, _, y_subsample, _ = train_test_split(
    raw_data["text"], raw_data["target"], stratify=raw_data["target"], train_size=500
)
```

Clean out the text removing by emails, names, etc. This will help improve the data that will then be converted into embeddings.


```
x_subsample = [re.sub(r"[\w\.-]+@[\w\.-]+", "", d) for d in x_subsample]  # Remove email
x_subsample = [re.sub(r"\([^()]*\)", "", d) for d in x_subsample]  # Remove names
x_subsample = [d.replace("From: ", "") for d in x_subsample]  # Remove "From: "
x_subsample = [
    d.replace("\nSubject: ", "") for d in x_subsample
]  # Remove "\nSubject: "
```


```
df = pd.DataFrame()
df["text"] = x_subsample
df["target"] = list(y_subsample)
```

You now have 500 data points roughly evenly distributed across the categories:


```
df["target"].value_counts()
```

## Create and visualize the embeddings using a t-SNE plot

Load the text embedding model from Vertex AI ([documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text-embeddings)).


```
model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
```


```
# Retrieve embeddings from the specified model with retry logic


def make_embed_text_fn(model):
    @retry.Retry(timeout=300.0)
    def embed_fn(text):
        return model.get_embeddings([text])[0].values

    return embed_fn
```

Create the embeddings. This may take a minute or two.


```
df["embeddings"] = df["text"].progress_apply(make_embed_text_fn(model))
```


```
df.head()
```

The vectors generate by our model are 768 dimensions, and so visualizing across 768 dimensions is impossible. Instead, you can use [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) to reduce to 2 dimensions.


```
embeddings_array = np.array(df["embeddings"].to_list(), dtype=np.float32)
tsne = TSNE(random_state=0, n_iter=1000)
tsne_results = tsne.fit_transform(embeddings_array)
```


```
df_tsne = pd.DataFrame(tsne_results, columns=["TSNE1", "TSNE2"])
df_tsne["target"] = df["target"]  # Add labels column from df_train to df_tsne
```


```
df_tsne.head()
```

Plot the data points. It should now be visually clear how the documents from the same newsgroup show up close to each other in the vector space with text embeddings.


```
fig, ax = plt.subplots(figsize=(8, 6))  # Set figsize
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.scatterplot(data=df_tsne, x="TSNE1", y="TSNE2", hue="target", palette="hls")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.title("Scatter plot of news using t-SNE")
plt.xlabel("TSNE1")
plt.ylabel("TSNE2")
plt.axis("equal")
```
