```
# Copyright 2024 Google LLC
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

# Multimodal Retrieval Augmented Generation (RAG) using Gemini API in Vertex AI

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/retrieval-augmented-generation/intro_multimodal_rag.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fuse-cases%2Fretrieval-augmented-generation%2Fintro_multimodal_rag.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/retrieval-augmented-generation/intro_multimodal_rag.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/use-cases/retrieval-augmented-generation/intro_multimodal_rag.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>    
</table>

| | |
|-|-|
|Author(s) | [Lavi Nigam](https://github.com/lavinigam-gcp) |

<div class="alert alert-block alert-warning">
<b>⚠️ There is a new version of this notebook with new data and some modifications here:  ⚠️</b>
</div>

[**building_DIY_multimodal_qa_system_with_mRAG.ipynb**](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/qa-ops/building_DIY_multimodal_qa_system_with_mRAG.ipynb)

You can, however, still use this notebook as it is fully functional and has updated Gemini and text-embedding models.

## Overview

Retrieval augmented generation (RAG) has become a popular paradigm for enabling LLMs to access external data and also as a mechanism for grounding to mitigate against hallucinations.

In this notebook, you will learn how to perform multimodal RAG where you will perform Q&A over a financial document filled with both text and images.

### Gemini

Gemini is a family of generative AI models developed by Google DeepMind that is designed for multimodal use cases. The Gemini API gives you access to the Gemini 1.0 Pro Vision and Gemini 1.0 Pro models.

### Comparing text-based and multimodal RAG

Multimodal RAG offers several advantages over text-based RAG:

1. **Enhanced knowledge access:** Multimodal RAG can access and process both textual and visual information, providing a richer and more comprehensive knowledge base for the LLM.
2. **Improved reasoning capabilities:** By incorporating visual cues, multimodal RAG can make better informed inferences across different types of data modalities.

This notebook shows you how to use RAG with Gemini API in Vertex AI, [text embeddings](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text-embeddings), and [multimodal embeddings](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/multimodal-embeddings), to build a document search engine.

Through hands-on examples, you will discover how to construct a multimedia-rich metadata repository of your document sources, enabling search, comparison, and reasoning across diverse information streams.

### Objectives

This notebook provides a guide to building a document search engine using multimodal retrieval augmented generation (RAG), step by step:

1. Extract and store metadata of documents containing both text and images, and generate embeddings the documents
2. Search the metadata with text queries to find similar text or images
3. Search the metadata with image queries to find similar images
4. Using a text query as input, search for contextual answers using both text and images

### Costs

This tutorial uses billable components of Google Cloud:

- Vertex AI

Learn about [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) and use the [Pricing Calculator](https://cloud.google.com/products/calculator/) to generate a cost estimate based on your projected usage.

## Getting Started

### Install Vertex AI SDK for Python and other dependencies


```
%pip install --upgrade --user google-cloud-aiplatform pymupdf rich colorama
```

### Restart current runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which will restart the current kernel.


```
# Restart kernel after installs so that your environment can access the new packages
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```




    {'status': 'ok', 'restart': True}



<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Please wait until it is finished before continuing to the next step. ⚠️</b>
</div>


### Authenticate your notebook environment (Colab only)

If you are running this notebook on Google Colab, run the following cell to authenticate your environment. This step is not required if you are using [Vertex AI Workbench](https://cloud.google.com/vertex-ai-workbench) or [Colab Enterprise](https://cloud.google.com/colab/docs).


```
import sys

# Additional authentication is required for Google Colab
if "google.colab" in sys.modules:
    # Authenticate user to Google Cloud
    from google.colab import auth

    auth.authenticate_user()
```

### Define Google Cloud project information


```
# Define project information

import sys

PROJECT_ID = "YOUR_PROJECT_ID"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

# if not running on Colab, try to get the PROJECT_ID automatically
if "google.colab" not in sys.modules:
    import subprocess

    PROJECT_ID = subprocess.check_output(
        ["gcloud", "config", "get-value", "project"], text=True
    ).strip()

print(f"Your project ID is: {PROJECT_ID}")
```


```
import sys

# Initialize Vertex AI
import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Import libraries


```
from IPython.display import Markdown, display
from rich.markdown import Markdown as rich_Markdown
from vertexai.generative_models import GenerationConfig, GenerativeModel, Image
```

### Load the Gemini 1.5 Pro and Gemini 1.5 Flash models


```
text_model = GenerativeModel("gemini-1.5-pro")
multimodal_model = GenerativeModel("gemini-1.5-pro")
multimodal_model_flash = GenerativeModel("gemini-1.5-flash")
```

### Download custom Python utilities & required files

The cell below will download a helper functions needed for this notebook, to improve readability. It also downloads other required files. You can also view the code for the utils here: (`intro_multimodal_rag_utils.py`) directly on [GitHub](https://storage.googleapis.com/github-repo/rag/intro_multimodal_rag/intro_multimodal_rag_old_version/intro_multimodal_rag_utils.py).


```
# download documents and images used in this notebook
!gsutil -m rsync -r gs://github-repo/rag/intro_multimodal_rag/intro_multimodal_rag_old_version .
print("Download completed")
```

    
    WARNING: gsutil rsync uses hashes when modification time is not available at
    both the source and destination. Your crcmod installation isn't using the
    module's C extension, so checksumming will run very slowly. If this is your
    first rsync since updating gsutil, this rsync can take significantly longer than
    usual. For help installing the extension, please see "gsutil help crcmod".
    
    Building synchronization state...
    Starting synchronization...
    Copying gs://github-repo/rag/intro_multimodal_rag/intro_multimodal_rag_old_version/data/google-10k-sample-part2.pdf...
    Copying gs://github-repo/rag/intro_multimodal_rag/intro_multimodal_rag_old_version/class_a_share.png...
    Copying gs://github-repo/rag/intro_multimodal_rag/intro_multimodal_rag_old_version/intro_multimodal_rag_utils.py...
    Copying gs://github-repo/rag/intro_multimodal_rag/intro_multimodal_rag_old_version/data/google-10k-sample-part1.pdf...
    Copying gs://github-repo/rag/intro_multimodal_rag/intro_multimodal_rag_old_version/tac_table_revenue.png...
    / [5/5 files][882.3 KiB/882.3 KiB] 100% Done                                    
    Operation completed over 5 objects/882.3 KiB.                                    
    Download completed
    

## Building metadata of documents containing text and images

### The data

The source data that you will use in this notebook is a modified version of [Google-10K](https://abc.xyz/assets/investor/static/pdf/20220202_alphabet_10K.pdf) which provides a comprehensive overview of the company's financial performance, business operations, management, and risk factors. As the original document is rather large, you will be using a modified version with only 14 pages, split into two parts - [Part 1](https://storage.googleapis.com/github-repo/rag/intro_multimodal_rag/intro_multimodal_rag_old_version/data/google-10k-sample-part1.pdf) and [Part 2](https://storage.googleapis.com/github-repo/rag/intro_multimodal_rag/intro_multimodal_rag_old_version/data/google-10k-sample-part2.pdf) instead. Although it's truncated, the sample document still contains text along with images such as tables, charts, and graphs.

### Import helper functions to build metadata

Before building the multimodal RAG system, it's important to have metadata of all the text and images in the document. For references and citations purposes, the metadata should contain essential elements, including page number, file name, image counter, and so on. Hence, as a next step, you will generate embeddings from the metadata, which will is required to perform similarity search when querying the data.


```
from intro_multimodal_rag_utils import get_document_metadata
```

### Extract and store metadata of text and images from a document

You just imported a function called `get_document_metadata()`. This function extracts text and image metadata from a document, and returns two dataframes, namely *text_metadata* and *image_metadata*, as outputs. If you want to find out more about how `get_document_metadata()` function is implemented using Gemini and the embedding models, you can take look at the [source code](https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/use-cases/retrieval-augmented-generation/utils/intro_multimodal_rag_utils.py) directly.

The reason for extraction and storing both text metadata and image metadata is that just by using either of the two alone is not sufficient to come out with a relevent answer. For example, the relevant answers could be in visual form within a document, but text-based RAG won't be able to take into consideration of the visual images. You will also be exploring this example later in this notebook.

At the next step, you will use the function to extract and store metadata of text and images froma document. Please note that the following cell may take a few minutes to complete:

Note:

The current implementation works best:

* if your documents are a combination of text and images.
* if the tables in your documents are available as images.
* if the images in the document don't require too much context.

Additionally,

* If you want to run this on text-only documents, use normal RAG
* If your documents contain particular domain knowledge, pass that information in the prompt below.

<div class="alert alert-block alert-warning">
<b>⚠️ Do not send more than 50 pages in the logic below, its not degined to do that and you will get into quota issue. ⚠️</b>
</div>


```
# Specify the PDF folder with multiple PDF

# pdf_folder_path = "/content/data/" # if running in Google Colab/Colab Enterprise
pdf_folder_path = "data/"  # if running in Vertex AI Workbench.

# Specify the image description prompt. Change it
image_description_prompt = """Explain what is going on in the image.
If it's a table, extract all elements of the table.
If it's a graph, explain the findings in the graph.
Do not include any numbers that are not mentioned in the image.
"""

# Extract text and image metadata from the PDF document
text_metadata_df, image_metadata_df = get_document_metadata(
    multimodal_model,  # we are passing Gemini 1.5 Pro model
    pdf_folder_path,
    image_save_dir="images",
    image_description_prompt=image_description_prompt,
    embedding_size=1408,
    # add_sleep_after_page = True, # Uncomment this if you are running into API quota issues
    # sleep_time_after_page = 5,
    # generation_config = # see next cell
    # safety_settings =  # see next cell
)

print("\n\n --- Completed processing. ---")
```

    
    
     Processing the file: --------------------------------- data/google-10k-sample-part1.pdf 
    
    
    Processing page: 1
    Processing page: 2
    Extracting image from page: 2, saved as: images/google-10k-sample-part1.pdf_image_1_0_11.jpeg
    Processing page: 3
    Extracting image from page: 3, saved as: images/google-10k-sample-part1.pdf_image_2_0_15.jpeg
    Processing page: 4
    Extracting image from page: 4, saved as: images/google-10k-sample-part1.pdf_image_3_0_18.jpeg
    Processing page: 5
    Extracting image from page: 5, saved as: images/google-10k-sample-part1.pdf_image_4_0_21.jpeg
    Processing page: 6
    Processing page: 7
    
    
     Processing the file: --------------------------------- data/google-10k-sample-part2.pdf 
    
    
    Processing page: 1
    Extracting image from page: 1, saved as: images/google-10k-sample-part2.pdf_image_0_0_6.jpeg
    Extracting image from page: 1, saved as: images/google-10k-sample-part2.pdf_image_0_1_8.jpeg
    Processing page: 2
    Extracting image from page: 2, saved as: images/google-10k-sample-part2.pdf_image_1_0_13.jpeg
    Processing page: 3
    Processing page: 4
    Extracting image from page: 4, saved as: images/google-10k-sample-part2.pdf_image_3_0_19.jpeg
    Processing page: 5
    Extracting image from page: 5, saved as: images/google-10k-sample-part2.pdf_image_4_0_22.jpeg
    Extracting image from page: 5, saved as: images/google-10k-sample-part2.pdf_image_4_1_23.jpeg
    Processing page: 6
    Extracting image from page: 6, saved as: images/google-10k-sample-part2.pdf_image_5_0_26.jpeg
    Processing page: 7
    
    
     --- Completed processing. ---
    


```
# # Parameters for Gemini API call.
# # reference for parameters: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini

# generation_config=  GenerationConfig(temperature=0.2, max_output_tokens=2048)

# # Set the safety settings if Gemini is blocking your content or you are facing "ValueError("Content has no parts")" error or "Exception occurred" in your data.
# # ref for settings and thresholds: https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/configure-safety-attributes

# safety_settings = {
#                   HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#                   HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#                   HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#                   HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#                   }

# # You can also pass parameters and safety_setting to "get_gemini_response" function
```

#### Inspect the processed text metadata


The following cell will produce a metadata table which describes the different parts of text metadata, including:

- **text**: the original text from the page
- **text_embedding_page**: the embedding of the original text from the page
- **chunk_text**: the original text divided into smaller chunks
- **chunk_number**: the index of each text chunk
- **text_embedding_chunk**: the embedding of each text chunk


```
text_metadata_df.head()
```





  <div id="df-1e1b05bc-c376-4cbb-aed7-8af227b5e934" class="colab-df-container">
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
      <th>file_name</th>
      <th>page_num</th>
      <th>text</th>
      <th>text_embedding_page</th>
      <th>chunk_number</th>
      <th>chunk_text</th>
      <th>text_embedding_chunk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>google-10k-sample-part1.pdf</td>
      <td>1</td>
      <td>source: https://abc.xyz/assets/investor/static...</td>
      <td>[0.06604167073965073, 0.054615460336208344, -0...</td>
      <td>1</td>
      <td>source: https://abc.xyz/assets/investor/static...</td>
      <td>[0.05712887644767761, 0.06096643581986427, -0....</td>
    </tr>
    <tr>
      <th>1</th>
      <td>google-10k-sample-part1.pdf</td>
      <td>1</td>
      <td>source: https://abc.xyz/assets/investor/static...</td>
      <td>[0.06604167073965073, 0.054615460336208344, -0...</td>
      <td>2</td>
      <td>of Record\nAs of December 31, 2021, there wer...</td>
      <td>[0.030928857624530792, 0.02287178672850132, -0...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>google-10k-sample-part1.pdf</td>
      <td>2</td>
      <td>Issuer Purchases of Equity Securities\nThe fol...</td>
      <td>[0.035787057131528854, 0.008179700933396816, -...</td>
      <td>1</td>
      <td>Issuer Purchases of Equity Securities\nThe fol...</td>
      <td>[0.035787057131528854, 0.008179700933396816, -...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>google-10k-sample-part1.pdf</td>
      <td>3</td>
      <td>Stock Performance Graphs\nThe graph below matc...</td>
      <td>[0.04338429123163223, 0.024151558056473732, -0...</td>
      <td>1</td>
      <td>Stock Performance Graphs\nThe graph below matc...</td>
      <td>[0.04338429123163223, 0.024151558056473732, -0...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>google-10k-sample-part1.pdf</td>
      <td>4</td>
      <td>The graph below matches Alphabet Inc. Class A'...</td>
      <td>[0.05739395692944527, 0.02924434281885624, -0....</td>
      <td>1</td>
      <td>The graph below matches Alphabet Inc. Class A'...</td>
      <td>[0.05739395692944527, 0.02924434281885624, -0....</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-1e1b05bc-c376-4cbb-aed7-8af227b5e934')"
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
        document.querySelector('#df-1e1b05bc-c376-4cbb-aed7-8af227b5e934 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1e1b05bc-c376-4cbb-aed7-8af227b5e934');
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


<div id="df-6052b484-8083-49be-baf7-3912524e4367">
  <button class="colab-df-quickchart" onclick="quickchart('df-6052b484-8083-49be-baf7-3912524e4367')"
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
        document.querySelector('#df-6052b484-8083-49be-baf7-3912524e4367 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




#### Inspect the processed image metadata

The following cell will produce a metadata table which describes the different parts of image metadata, including:
* **img_desc**: Gemini-generated textual description of the image.
* **mm_embedding_from_text_desc_and_img**: Combined embedding of image and its description, capturing both visual and textual information.
* **mm_embedding_from_img_only**: Image embedding without description, for comparison with description-based analysis.
* **text_embedding_from_image_description**: Separate text embedding of the generated description, enabling textual analysis and comparison.


```
image_metadata_df.head()
```





  <div id="df-769ab2e5-8ea1-4d9e-b82b-6300d5e7022a" class="colab-df-container">
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
      <th>file_name</th>
      <th>page_num</th>
      <th>img_num</th>
      <th>img_path</th>
      <th>img_desc</th>
      <th>mm_embedding_from_img_only</th>
      <th>text_embedding_from_image_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>google-10k-sample-part1.pdf</td>
      <td>2</td>
      <td>1</td>
      <td>images/google-10k-sample-part1.pdf_image_1_0_1...</td>
      <td>The image is a table that shows the number of ...</td>
      <td>[0.0232503358, -0.000800505339, 0.0152765336, ...</td>
      <td>[0.022681448608636856, 0.016109690070152283, 0...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>google-10k-sample-part1.pdf</td>
      <td>3</td>
      <td>1</td>
      <td>images/google-10k-sample-part1.pdf_image_2_0_1...</td>
      <td>The image is a line graph comparing the cumula...</td>
      <td>[0.0068904995, 0.0237238202, -0.00902639609, 0...</td>
      <td>[0.012635421007871628, 0.029609661549329758, -...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>google-10k-sample-part1.pdf</td>
      <td>4</td>
      <td>1</td>
      <td>images/google-10k-sample-part1.pdf_image_3_0_1...</td>
      <td>The image is a line graph comparing the cumula...</td>
      <td>[0.0065851449, 0.010453077, -0.0087017715, 0.0...</td>
      <td>[0.031192757189273834, 0.04452929273247719, -0...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>google-10k-sample-part1.pdf</td>
      <td>5</td>
      <td>1</td>
      <td>images/google-10k-sample-part1.pdf_image_4_0_2...</td>
      <td>The image is a table showing financial data fo...</td>
      <td>[0.01665527, 0.019989036, -0.0204045977, -0.01...</td>
      <td>[0.03382618725299835, 0.04069436714053154, -0....</td>
    </tr>
    <tr>
      <th>4</th>
      <td>google-10k-sample-part2.pdf</td>
      <td>1</td>
      <td>1</td>
      <td>images/google-10k-sample-part2.pdf_image_0_0_6...</td>
      <td>The image is a table showing percentages for d...</td>
      <td>[0.0357290804, 0.0324405842, 0.0125655765, -0....</td>
      <td>[0.017558930441737175, 0.046305228024721146, -...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-769ab2e5-8ea1-4d9e-b82b-6300d5e7022a')"
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
        document.querySelector('#df-769ab2e5-8ea1-4d9e-b82b-6300d5e7022a button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-769ab2e5-8ea1-4d9e-b82b-6300d5e7022a');
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


<div id="df-34ef9a06-3129-476b-96aa-cf0a01af7b1e">
  <button class="colab-df-quickchart" onclick="quickchart('df-34ef9a06-3129-476b-96aa-cf0a01af7b1e')"
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
        document.querySelector('#df-34ef9a06-3129-476b-96aa-cf0a01af7b1e button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




### Import the helper functions to implement RAG

You will be importing the following functions which will be used in the remainder of this notebook to implement RAG:

* **get_similar_text_from_query():** Given a text query, finds text from the document which are relevant, using cosine similarity algorithm. It uses text embeddings from the metadata to compute and the results can be filtered by top score, page/chunk number, or embedding size.
* **print_text_to_text_citation():** Prints the source (citation) and details of the retrieved text from the `get_similar_text_from_query()` function.
* **get_similar_image_from_query():** Given an image path or an image, finds images from the document which are relevant. It uses image embeddings from the metadata.
* **print_text_to_image_citation():** Prints the source (citation) and the details of retrieved images from the `get_similar_image_from_query()` function.
* **get_gemini_response():** Interacts with a Gemini model to answer questions based on a combination of text and image inputs.
* **display_images():**  Displays a series of images provided as paths or PIL Image objects.


```
from intro_multimodal_rag_utils import (
    display_images,
    get_gemini_response,
    get_similar_image_from_query,
    get_similar_text_from_query,
    print_text_to_image_citation,
    print_text_to_text_citation,
)
```

Before implementing a multimodal RAG, let's take a step back and explore what you can achieve with just text or image embeddings alone. It will help to set the foundation for implementing a multimodal RAG, which you will be doing in the later part of the notebook. You can also use these essential elements together to build applications for multimodal use cases for extracting meaningful information from the document.

## Text Search

Let's start the search with a simple question and see if the simple text search using text embeddings can answer it. The expected answer is to show the value of basic and diluted net income per share of Google for different share types.


```
query = "I need details for basic and diluted net income per share of Class A, Class B, and Class C share for google?"
```

### Search similar text with text query


```
# Matching user text query with "chunk_embedding" to find relevant chunks.
matching_results_text = get_similar_text_from_query(
    query,
    text_metadata_df,
    column_name="text_embedding_chunk",
    top_n=3,
    chunk_text=True,
)

# Print the matched text citations
print_text_to_text_citation(matching_results_text, print_top=False, chunk_text=True)
```

    [91mCitation 1: Matched text: 
    [0m
    [94mscore: [0m 0.76
    [94mfile_name: [0m google-10k-sample-part2.pdf
    [94mpage_number: [0m 4
    [94mchunk_number: [0m 1
    [94mchunk_text: [0m liquidation and dividend rights are identical, the undistributed earnings are
    allocated on a proportionate basis.
    In the years ended December 31, 2019, 2020 and 2021, the net income per
    share amounts are the same for Class A, Class B, and Class C stock because
    the holders of each class are entitled to equal per share dividends or distributions
    in liquidation in accordance with the Amended and Restated Certificate of
    Incorporation of Alphabet Inc.
    The following tables set forth the computation of basic and diluted net income per
    share of Class A, Class B, and Class C stock (in millions, except share amounts
    which are reflected in thousands and per share amounts):
    
    [91mCitation 2: Matched text: 
    [0m
    [94mscore: [0m 0.7
    [94mfile_name: [0m google-10k-sample-part2.pdf
    [94mpage_number: [0m 3
    [94mchunk_number: [0m 1
    [94mchunk_text: [0m increases in content acquisition costs primarily for YouTube, data center and
    other operations costs, and hardware costs. The increase in data center and
    Table of Contents Alphabet Inc. 36 other operations costs was partially offset by
    a reduction in depreciation expense due to the change in the estimated useful life
    of our servers and certain network equipment beginning in the first quarter of
    2021.
    Net Income Per Share
    We compute net income per share of Class A, Class B, and Class C stock using
    the two-class method. Basic net income per share is computed using the
    weighted-average number of shares outstanding during the period. Diluted net
    income per share is computed using the weighted-average number of shares and
    the effect of potentially dilutive securities outstanding during the period.
    Potentially dilutive securities consist of restricted stock units and other
    contingently issuable shares. The dilutive effect of outstanding restricted stock
    units and other contingently issuable 
    [91mCitation 3: Matched text: 
    [0m
    [94mscore: [0m 0.67
    [94mfile_name: [0m google-10k-sample-part2.pdf
    [94mpage_number: [0m 3
    [94mchunk_number: [0m 2
    [94mchunk_text: [0m e shares. The dilutive effect of outstanding restricted stock
    units and other contingently issuable shares is reflected in diluted earnings per
    share by application of the treasury stock method. The computation of the diluted
    net income per share of Class A stock assumes the conversion of Class B stock,
    while the diluted net income per share of Class B stock does not assume the
    conversion of those shares.
    The rights, including the liquidation and dividend rights, of the holders of our
    Class A, Class B, and Class C stock are identical, except with respect to voting.
    Furthermore, there are a number of safeguards built into our certificate of
    incorporation, as well as Delaware law, which preclude our Board of Directors
    from declaring or paying unequal per share dividends on our Class A, Class B,
    and Class C stock. Specifically, Delaware law provides that amendments to our
    certificate of incorporation which would have the effect of adversely altering the
    rights, powers, or preferences of a
    

You can see that the first high score match does have what we are looking for, but upon closer inspection, it mentions that the information is available in the "following" table. The table data is available as an image rather than as text, and hence, the chances are you will miss the information unless you can find a way to process images and their data.

However, Let's feed the relevant text chunk across the data into the Gemini 1.0 Pro model and see if it can get your desired answer by considering all the chunks across the document. This is like basic text-based RAG implementation.


```
print("\n **** Result: ***** \n")

# All relevant text chunk found across documents based on user query
context = "\n".join(
    [value["chunk_text"] for key, value in matching_results_text.items()]
)

instruction = f"""Answer the question with the given context.
If the information is not available in the context, just return "not available in the context".
Question: {query}
Context: {context}
Answer:
"""

# Prepare the model input
model_input = instruction

# Generate Gemini response with streaming output
get_gemini_response(
    text_model,  # we are passing Gemini 1.0 Pro
    model_input=model_input,
    stream=True,
    generation_config=GenerationConfig(temperature=0.2),
)
```

    
     **** Result: ***** 
    
    




    'not available in the context \n'



You can see that it returned:

*"The provided context does not include the details for basic and diluted net income per share of Class A, Class B, and Class C share for google.
"*

This is expected as discussed previously. No other text chunk (total 3) had the information you sought.
This is because the information is only available in the images rather than in the text part of the document. Next, let's see if you can solve this problem by leveraging Gemini 1.0 Pro Vision and Multimodal Embeddings.

Note: We handcrafted examples in our document to simulate real-world cases where information is often embedded in charts, table, graphs, and other image-based elements and unavailable as plain text.  

### Search similar images with text query

Since plain text search didn't provide the desired answer and the information may be visually represented in a table or another image format, you will use multimodal capability of Gemini 1.0 Pro Vision model for the similar task. The goal here also is to find an image similar to the text query. You may also print the citations to verify.


```
query = "I need details for basic and diluted net income per share of Class A, Class B, and Class C share for google?"
```


```
matching_results_image = get_similar_image_from_query(
    text_metadata_df,
    image_metadata_df,
    query=query,
    column_name="text_embedding_from_image_description",  # Use image description text embedding
    image_emb=False,  # Use text embedding instead of image embedding
    top_n=3,
    embedding_size=1408,
)

# Markdown(print_text_to_image_citation(matching_results_image, print_top=True))
print("\n **** Result: ***** \n")

# Display the top matching image
display(matching_results_image[0]["image_object"])
```

    
     **** Result: ***** 
    
    


    
![png](output_54_1.png)
    


Bingo! It found exactly what you were looking for. You wanted the details on Google's Class A, B, and C shares' basic and diluted net income, and guess what? This image fits the bill perfectly thanks to its descriptive metadata using Gemini.

You can also send the image and its description to Gemini 1.0 Pro Vision and get the answer as JSON:


```
print("\n **** Result: ***** \n")

# All relevant text chunk found across documents based on user query
context = f"""Image: {matching_results_image[0]['image_object']}
Description: {matching_results_image[0]['image_description']}
"""

instruction = f"""Answer the question in JSON format with the given context of Image and its Description. Only include value.
Question: {query}
Context: {context}
Answer:
"""

# Prepare the model input
model_input = instruction

# Generate Gemini response with streaming output
Markdown(
    get_gemini_response(
        multimodal_model_flash,  # we are passing Gemini 1.5 Pro Flash
        model_input=model_input,
        stream=True,
        generation_config=GenerationConfig(temperature=1),
    )
)
```

    
     **** Result: ***** 
    
    




```json
{
"Class A": {
"Basic Net Income Per Share": 59.15,
"Diluted Net Income Per Share": 58.61
},
"Class B": {
"Basic Net Income Per Share": 59.15,
"Diluted Net Income Per Share": 58.61
},
"Class C": {
"Basic Net Income Per Share": 59.15,
"Diluted Net Income Per Share": 58.61
}
}
```




```
## you can check the citations to probe further.
## check the "image description:" which is a description extracted through Gemini which helped search our query.
Markdown(print_text_to_image_citation(matching_results_image, print_top=True))
```

    [91mCitation 1: Matched image path, page number and page text: 
    [0m
    [94mscore: [0m 0.73
    [94mfile_name: [0m google-10k-sample-part2.pdf
    [94mpath: [0m images/google-10k-sample-part2.pdf_image_4_0_22.jpeg
    [94mpage number: [0m 5
    [94mpage text: [0m Stock-Based Award Activities
    The weighted-average grant-date fair value of RSUs granted during the years
    ended December 31, 2019 and 2020 was $1,092.36 and $1,407.97, respectively.
    Total fair value of RSUs, as of their respective vesting dates, during the years
    ended December 31, 2019, 2020, and 2021 were $15.2 billion, $17.8 billion, and
    $28.8 billion, respectively. As of December 31, 2021, there was $25.8 billion of
    unrecognized compensation cost related to unvested employee RSUs. This
    amount is expected to be recognized over a weighted-average period of 2.5
    years. 401(k) Plans We have two 401(k) Savings Plans that qualify as deferred
    
    [94mimage description: [0m The image is a table showing the calculation of basic and diluted net income per share for different classes of shares (Class A, Class B, Class C) for the year ended December 31, 2020. 
    
    **Basic Net Income Per Share**
    
    * **Numerator:**
        * Allocation of undistributed earnings: Class A: $17,733, Class B: $2,732, Class C: $19,804
    * **Denominator:**
        * Number of shares used in per share computation: Class A: 299,815, Class B: 46,182, Class C: 334,819
    * **Basic net income per share:** Class A: $59.15, Class B: $59.15, Class C: $59.15
    
    **Diluted Net Income Per Share**
    
    * **Numerator:**
        * Allocation of undistributed earnings for basic computation: Class A: $17,733, Class B: $2,732, Class C: $19,804
        * Reallocation of undistributed earnings as a result of conversion of Class B to Class A shares: Class A: $2,732, Class B: $0, Class C: $0
        * Reallocation of undistributed earnings: Class A: $(180), Class B: $(25), Class C: $180
        * Allocation of undistributed earnings: Class A: $20,285, Class B: $2,707, Class C: $19,984
    * **Denominator:**
        * Number of shares used in basic computation: Class A: 299,815, Class B: 46,182, Class C: 334,819
        * Add:
            * Conversion of Class B to Class A shares outstanding: Class A: 46,182, Class B: 0, Class C: 0
            * Restricted stock units and other contingently issuable shares: Class A: 87, Class B: 0, Class C: 6,125
        * Number of shares used in per share computation: Class A: 346,084, Class B: 46,182, Class C: 340,944
    * **Diluted net income per share:** Class A: $58.61, Class B: $58.61, Class C: $58.61 
    
    




    <IPython.core.display.Markdown object>



## Image Search

### Search similar image with image query

Imagine searching for images, but instead of typing words, you use an actual image as the clue. You have a table with numbers about the cost of revenue for two years, and you want to find other images that look like it, from the same document or across multiple documents.

Think of it like searching with a mini-map instead of a written address. It's a different way to ask, "Show me more stuff like this". So, instead of typing "cost of revenue 2020 2021 table", you show a picture of that table and say, "Find me more like this"

For demonstration purposes, we will only be finding similar images that show the cost of revenue or similar values in a single document below. However, you can scale this design pattern to match (find relevant images) across multiple documents.


```
# You can find a similar image as per the images you have in the metadata.
# In this case, you have a table (picked from the same document source) and you would like to find similar tables in the document.
image_query_path = "tac_table_revenue.png"

# Print a message indicating the input image
print("***Input image from user:***")

# Display the input image
Image.load_from_file(image_query_path)
```

    ***Input image from user:***
    




    
![png](output_61_1.png)
    



You expect to find tables (as images) that are similar in terms of "Other/Total cost of revenues."


```
# Search for Similar Images Based on Input Image and Image Embedding

matching_results_image = get_similar_image_from_query(
    text_metadata_df,
    image_metadata_df,
    query=query,  # Use query text for additional filtering (optional)
    column_name="mm_embedding_from_img_only",  # Use image embedding for similarity calculation
    image_emb=True,
    image_query_path=image_query_path,  # Use input image for similarity calculation
    top_n=3,  # Retrieve top 3 matching images
    embedding_size=1408,  # Use embedding size of 1408
)

print("\n **** Result: ***** \n")

# Display the Top Matching Image
display(
    matching_results_image[0]["image_object"]
)  # Display the top matching image object (Pillow Image)
```

    
     **** Result: ***** 
    
    


    
![png](output_63_1.png)
    


It did find a similar-looking image (table), which gives more detail about different revenue, expenses, income, and a few more details based on the given image. More importantly, both tables show numbers related to the "cost of revenue."

You can also print the citation to see what it has matched.


```
# Display citation details for the top matching image
print_text_to_image_citation(
    matching_results_image, print_top=True
)  # Print citation details for the top matching image
```

    [91mCitation 1: Matched image path, page number and page text: 
    [0m
    [94mscore: [0m 0.77
    [94mfile_name: [0m google-10k-sample-part1.pdf
    [94mpath: [0m images/google-10k-sample-part1.pdf_image_4_0_21.jpeg
    [94mpage number: [0m 5
    [94mpage text: [0m Executive Overview
    The following table summarizes consolidated financial results for the years ended
    December 31, 2020 and 2021 unless otherwise specified (in millions, except for
    per share information and percentages):
    Revenues were $257.6 billion, an increase of 41%. The increase in
    revenues was primarily driven by Google Services and Google Cloud. The
    adverse effect of COVID-19 on 2020 advertising revenues also contributed
    to the year-over-year growth.
    Cost of revenues was $110.9 billion, an increase of 31%, primarily driven
    by increases in TAC and content acquisition costs.
    An overall increase in data centers and other operations costs was partially
    offset by a reduction in depreciation expense due to the change in the
    estimated useful life of our servers and certain network equipment. 
    Operating expenses were $68.0 billion, an increase of 20%, primarily
    driven by headcount growth, increases in advertising and promotional
    expenses and charges related to legal matters.
    
    [94mimage description: [0m The image is a table showing financial data for a company for the years 2020 and 2021. Here are the elements of the table:
    
    | Category | 2020 | 2021 | $ Change | % Change |
    |---|---|---|---|---|
    | Consolidated revenues | $182,527 | $257,637 | $75,110 | 41% |
    | Change in consolidated constant currency revenues |  |  |  | 39% |
    | Cost of revenues | $84,732 | $110,939 | $26,207 | 31% |
    | Operating expenses | $56,571 | $67,984 | $11,413 | 20% |
    | Operating income | $41,224 | $78,714 | $37,490 | 91% |
    | Operating margin | 23% | 31% |  | 8% |
    | Other income (expense), net | $6,858 | $12,020 | $5,162 | 75% |
    | Net Income | $40,269 | $76,033 | $35,764 | 89% |
    | Diluted EPS | $58.61 | $112.20 | $53.59 | 91% |
    | Number of Employees | 135,301 | 156,500 | 21,199 | 16% | 
    
    


```
# Check Other Matched Images (Optional)
# You can access the other two matched images using:

print("---------------Matched Images------------------\n")
display_images(
    [
        matching_results_image[0]["img_path"],
        matching_results_image[1]["img_path"],
    ],
    resize_ratio=0.5,
)
```

    ---------------Matched Images------------------
    
    


    
![png](output_66_1.png)
    


    
    
    


    
![png](output_66_3.png)
    


    
    
    

The ability to identify similar text and images based on user input, using Gemini and embeddings, forms a crucial foundation for development of multimodal RAG systems, which you explore in the next section.

### Comparative reasoning

Next, let's apply what you have done so far to doing comparative reasoning.

For this example:

Step 1: You will search all the images for a specific query

Step 2: Send those images to Gemini 1.0 Pro Vision to ask multiple questions, where it has to compare and provide you with answers.


```
matching_results_image_query_1 = get_similar_image_from_query(
    text_metadata_df,
    image_metadata_df,
    query="Show me all the graphs that shows Google Class A cumulative 5-year total return",
    column_name="text_embedding_from_image_description",  # Use image description text embedding # mm_embedding_from_img_only text_embedding_from_image_description
    image_emb=False,  # Use text embedding instead of image embedding
    top_n=3,
    embedding_size=1408,
)
```


```
# Check Matched Images
# You can access the other two matched images using:

print("---------------Matched Images------------------\n")
display_images(
    [
        matching_results_image_query_1[0]["img_path"],
        matching_results_image_query_1[1]["img_path"],
    ],
    resize_ratio=0.5,
)
```

    ---------------Matched Images------------------
    
    


    
![png](output_71_1.png)
    


    
    
    


    
![png](output_71_3.png)
    


    
    
    


```
prompt = f""" Instructions: Compare the images and the Gemini extracted text provided as Context: to answer Question:
Make sure to think thoroughly before answering the question and put the necessary steps to arrive at the answer in bullet points for easy explainability.

Context:
Image_1: {matching_results_image_query_1[0]["image_object"]}
gemini_extracted_text_1: {matching_results_image_query_1[0]['image_description']}
Image_2: {matching_results_image_query_1[1]["image_object"]}
gemini_extracted_text_2: {matching_results_image_query_1[2]['image_description']}

Question:
 - Key findings of Class A share?
 - What are the critical differences between the graphs for Class A Share?
 - What are the key findings of Class A shares concerning the S&P 500?
 - Which index best matches Class A share performance closely where Google is not already a part? Explain the reasoning.
 - Identify key chart patterns in both graphs.
 - Which index best matches Class A share performance closely where Google is not already a part? Explain the reasoning.
"""

# Generate Gemini response with streaming output
rich_Markdown(
    get_gemini_response(
        multimodal_model,  # we are passing Gemini 1.5 Pro
        model_input=[prompt],
        stream=True,
        generation_config=GenerationConfig(temperature=1),
    )
)
```




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Unfortunately, I cannot directly access or analyze the content of images or URLs, including the ones represented   
here as "&lt;vertexai.generative_models._generative_models.Image object...&gt;". My abilities are limited to processing  
and understanding the text provided.                                                                               

However, based on the text descriptions you've given:                                                              

<span style="font-weight: bold">Image 1 Analysis</span>                                                                                                   

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Key findings of Class A share?</span>                                                                                  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>To answer this, I need the actual performance trend from the line graph.  Was Alphabet Inc. Class A stock    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>generally up, down, or flat over the 5-year period? Did it experience any sharp rises or falls?              
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">What are the critical differences between the graphs for Class A Share?</span>                                         
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>This question seems to imply there are multiple graphs for Class A shares within Image 1. The description    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>mentions only one line representing Class A shares.  Please clarify.                                         
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">What are the key findings of Class A shares concerning the S&amp;P 500?</span>                                             
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>This requires comparing the Class A share performance line to the S&amp;P 500 line on the graph. Did Class A     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>shares outperform or underperform the S&amp;P 500? Was the relationship consistent, or did it change over the 5  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>years?                                                                                                       
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Which index best matches Class A share performance closely where Google is not already a part?</span>                  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>Again, I need the visual trend from the graph to make this comparison.  Ideally, you'd look for an index line
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>that closely follows the shape and direction of the Class A share line, indicating a similar performance     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>pattern.                                                                                                     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Identify key chart patterns in both graphs.</span>                                                                     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">For the line graph (Image 1):</span> Look for trends (upward, downward, flat), any significant spikes or dips, and  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>how the lines for different indices relate to each other.                                                    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">For the table (Image 2):</span>  Key patterns would involve analyzing the changes in unvested shares over time. For 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>example, was there a significant increase or decrease in granted shares or shares that were                  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>forfeited/canceled?                                                                                          

<span style="font-weight: bold">Image 2 Analysis</span>                                                                                                   

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span>This table focuses on Alphabet's employee stock options (RSUs), not directly on the performance of Class A      
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>shares in the market.  It wouldn't be appropriate to compare or draw direct conclusions about Class A share     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>performance based on this table.                                                                                

<span style="font-weight: bold">To get a complete analysis, please provide:</span>                                                                        

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Clarification on multiple Class A graphs:</span> If there's more than one representation of Class A shares in Image 1. 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Description of the line trends:</span> Summarize how the lines move on the graph (e.g., "Class A shares rose sharply in
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>2019, then plateaued...").                                                                                      

Let me know if you can provide more context, and I'll do my best to help!                                          
</pre>




<div class="alert alert-block alert-warning">
<b>⚠️ Disclaimer: This is not a real investment advise and should not be taken seriously!! ⚠️</b>
</div>

## Multimodal retrieval augmented generation (RAG)

Let's bring everything together to implement multimodal RAG. You will use all the elements that you've explored in previous sections to implement the multimodal RAG. These are the steps:

* **Step 1:** The user gives a query in text format where the expected information is available in the document and is embedded in images and text.
* **Step 2:** Find all text chunks from the pages in the documents using a method similar to the one you explored in `Text Search`.
* **Step 3:** Find all similar images from the pages based on the user query matched with `image_description` using a method identical to the one you explored in `Image Search`.
* **Step 4:** Combine all similar text and images found in steps 2 and 3 as `context_text` and `context_images`.
* **Step 5:** With the help of Gemini, we can pass the user query with text and image context found in steps 2 & 3. You can also add a specific instruction the model should remember while answering the user query.
* **Step 6:** Gemini produces the answer, and you can print the citations to check all relevant text and images used to address the query.

### Step 1: User query


```
# this time we are not passing any images, but just a simple text query.

query = """Questions:
 - What are the critical difference between various graphs for Class A Share?
 - Which index best matches Class A share performance closely where Google is not already a part? Explain the reasoning.
 - Identify key chart patterns for Google Class A shares.
 - What is cost of revenues, operating expenses and net income for 2020. Do mention the percentage change
 - What was the effect of Covid in the 2020 financial year?
 - What are the total revenues for APAC and USA for 2021?
 - What is deferred income taxes?
 - How do you compute net income per share?
 - What drove percentage change in the consolidated revenue and cost of revenue for the year 2021 and was there any effect of Covid?
 - What is the cause of 41% increase in revenue from 2020 to 2021 and how much is dollar change?
 """
```

### Step 2: Get all relevant text chunks


```
# Retrieve relevant chunks of text based on the query
matching_results_chunks_data = get_similar_text_from_query(
    query,
    text_metadata_df,
    column_name="text_embedding_chunk",
    top_n=10,
    chunk_text=True,
)
```

### Step 3: Get all relevant images


```
# Get all relevant images based on user query
matching_results_image_fromdescription_data = get_similar_image_from_query(
    text_metadata_df,
    image_metadata_df,
    query=query,
    column_name="text_embedding_from_image_description",
    image_emb=False,
    top_n=10,
    embedding_size=1408,
)
```

### Step 4: Create context_text and context_images


```
# combine all the selected relevant text chunks
context_text = []
for key, value in matching_results_chunks_data.items():
    context_text.append(value["chunk_text"])
final_context_text = "\n".join(context_text)

# combine all the relevant images and their description generated by Gemini
context_images = []
for key, value in matching_results_image_fromdescription_data.items():
    context_images.extend(
        ["Image: ", value["image_object"], "Caption: ", value["image_description"]]
    )
```

### Step 5: Pass context to Gemini


```
prompt = f""" Instructions: Compare the images and the text provided as Context: to answer multiple Question:
Make sure to think thoroughly before answering the question and put the necessary steps to arrive at the answer in bullet points for easy explainability.
If unsure, respond, "Not enough context to answer".

Context:
 - Text Context:
 {final_context_text}
 - Image Context:
 {context_images}

{query}

Answer:
"""

# Generate Gemini response with streaming output
rich_Markdown(
    get_gemini_response(
        multimodal_model,
        model_input=[prompt],
        stream=True,
        generation_config=GenerationConfig(temperature=1),
    )
)
```




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Here are the answers to your questions, broken down for clarity:                                                   

<span style="font-weight: bold">1. What are the critical difference between various graphs for Class A Share?</span>                                      

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Time Period:</span> The provided text includes two different 5-year cumulative return graphs for Alphabet Inc. Class A 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>stock:                                                                                                          
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>December 31, 2016 to December 31, 2021                                                                       
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>December 31, 2017 to December 31, 2022                                                                       
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Data Included:</span>                                                                                                  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>Both graphs compare Class A share performance to the S&amp;P 500, NASDAQ Composite, and RDG Internet Composite.  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>No other graphs specifically focusing on Class A shares are described in the text.                           

<span style="font-weight: bold">2. Which index best matches Class A share performance closely where Google is not already a part? Explain the </span>     
<span style="font-weight: bold">reasoning.</span>                                                                                                         

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Not enough context to answer.</span>                                                                                   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>The text doesn't specify the composition of the RDG Internet Composite index. It's impossible to know if     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>Google is already a component of this index.                                                                 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>To determine the closest matching index, we'd need to visually analyze the graphs and compare the slopes and 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>patterns of the lines representing each index against Alphabet Inc. Class A shares.                          

<span style="font-weight: bold">3. Identify key chart patterns for Google Class A shares.</span>                                                          

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Not enough context to answer.</span>  We need to visually inspect the graphs to identify patterns (e.g., uptrends,     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>downtrends, consolidation periods).                                                                             

<span style="font-weight: bold">4. What is cost of revenues, operating expenses, and net income for 2020. Do mention the percentage change</span>         

Here's the information from the provided text:                                                                     

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Cost of Revenues (2020):</span> $84,732 million                                                                        
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Operating Expenses (2020):</span> $56,571 million                                                                      
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Net Income (2020):</span> $40,269 million                                                                              

The text doesn't directly provide the percentage change for these items in 2020. It focuses on changes from 2020 to
2021.                                                                                                              

<span style="font-weight: bold">5. What was the effect of Covid in the 2020 financial year?</span>                                                        

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Reduced advertising revenue:</span>  The text states that starting in March 2020, advertising revenue declined despite 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>increased search activity. This was attributed to:                                                              
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>Users searching for less commercially focused topics.                                                        
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>Advertisers reducing spending.                                                                               
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Revenue decline in Q2 2020:</span>  The text mentions a continued decline in advertising revenues for the quarter ended
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>June 30, 2020.                                                                                                  

<span style="font-weight: bold">6. What are the total revenues for APAC and USA for 2021?</span>                                                          

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">APAC Revenues (2021):</span> $46,123 million                                                                           
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">USA Revenues (2021):</span> $117,854 million                                                                           

<span style="font-weight: bold">7. What is deferred income taxes?</span>                                                                                  

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Definition:</span> Deferred income taxes represent the difference between income tax expenses reported on the income   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>statement and taxes payable to tax authorities.  These differences arise due to temporary discrepancies between 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>accounting rules and tax regulations.                                                                           
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Example:</span>  A company might depreciate an asset faster for tax purposes than for financial reporting. This creates
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>a deferred tax liability because they'll eventually have to pay taxes on the income that was deferred.          

<span style="font-weight: bold">8. How do you compute net income per share?</span>                                                                        

<span style="font-weight: bold">Basic Net Income Per Share:</span>                                                                                        

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> 1 </span><span style="font-weight: bold">Numerator:</span> Net income available to common shareholders (after preferred dividends, if any)                      
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> 2 </span><span style="font-weight: bold">Denominator:</span> Weighted-average number of common shares outstanding during the period                             

<span style="font-weight: bold">Diluted Net Income Per Share:</span>                                                                                      

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> 1 </span><span style="font-weight: bold">Numerator:</span>  Same as basic net income, adjusted for the potential dilutive effect of stock options, convertible  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>securities, etc.                                                                                                
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> 2 </span><span style="font-weight: bold">Denominator:</span> Weighted-average number of common shares outstanding <span style="font-weight: bold">plus</span> the potential dilutive effect of stock   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>options, convertible securities, etc., if they were exercised/converted.                                        

<span style="font-weight: bold">9. What drove the percentage change in consolidated revenue and the cost of revenue for the year 2021, and was </span>    
<span style="font-weight: bold">there any effect of Covid?</span>                                                                                         

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Consolidated Revenue Increase (41%):</span>                                                                            
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>Primarily driven by growth in Google Services and Google Cloud.                                              
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>Rebound from the adverse effects of COVID-19 on 2020 advertising revenue also contributed.                   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Cost of Revenue Increase (31%):</span>                                                                                 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>Primarily driven by:                                                                                         
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">       • </span>Increase in TAC (Traffic Acquisition Costs) - payments to distribution partners.                          
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">       • </span>Increase in content acquisition costs (primarily for YouTube).                                            
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">       • </span>Increase in data center and other operational costs.                                                      
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>Partially offset by a decrease in depreciation expense due to changes in the estimated useful life of servers
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>and network equipment.                                                                                       
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">COVID-19 Effect:</span>  The significant revenue increase in 2021 was partly due to the recovery from the pandemic's   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>negative impact in 2020, suggesting a continued effect of COVID-19, though it's shifting from negative to       
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>positive.                                                                                                       

<span style="font-weight: bold">10. What is the cause of the 41% increase in revenue from 2020 to 2021 and how much is the dollar change?</span>          

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Cause:</span>                                                                                                          
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>Growth in Google Services and Google Cloud.                                                                  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>Recovery from the negative impact of COVID-19 on 2020 advertising revenues.                                  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Dollar Change:</span> $75,110 million (from $182,527 million in 2020 to $257,637 million in 2021).                     
</pre>




### Step 6: Print citations and references


```
print("---------------Matched Images------------------\n")
display_images(
    [
        matching_results_image_fromdescription_data[0]["img_path"],
        matching_results_image_fromdescription_data[1]["img_path"],
        matching_results_image_fromdescription_data[2]["img_path"],
        matching_results_image_fromdescription_data[3]["img_path"],
    ],
    resize_ratio=0.5,
)
```

    ---------------Matched Images------------------
    
    


    
![png](output_86_1.png)
    


    
    
    


    
![png](output_86_3.png)
    


    
    
    


    
![png](output_86_5.png)
    


    
    
    


    
![png](output_86_7.png)
    


    
    
    


```
# Image citations. You can check how Gemini generated metadata helped in grounding the answer.

print_text_to_image_citation(
    matching_results_image_fromdescription_data, print_top=False
)
```

    [91mCitation 1: Matched image path, page number and page text: 
    [0m
    [94mscore: [0m 0.66
    [94mfile_name: [0m google-10k-sample-part2.pdf
    [94mpath: [0m images/google-10k-sample-part2.pdf_image_4_0_22.jpeg
    [94mpage number: [0m 5
    [94mpage text: [0m Stock-Based Award Activities
    The weighted-average grant-date fair value of RSUs granted during the years
    ended December 31, 2019 and 2020 was $1,092.36 and $1,407.97, respectively.
    Total fair value of RSUs, as of their respective vesting dates, during the years
    ended December 31, 2019, 2020, and 2021 were $15.2 billion, $17.8 billion, and
    $28.8 billion, respectively. As of December 31, 2021, there was $25.8 billion of
    unrecognized compensation cost related to unvested employee RSUs. This
    amount is expected to be recognized over a weighted-average period of 2.5
    years. 401(k) Plans We have two 401(k) Savings Plans that qualify as deferred
    
    [94mimage description: [0m The image is a table showing the calculation of basic and diluted net income per share for different classes of shares (Class A, Class B, Class C) for the year ended December 31, 2020. 
    
    **Basic Net Income Per Share**
    
    * **Numerator:**
        * Allocation of undistributed earnings: Class A: $17,733, Class B: $2,732, Class C: $19,804
    * **Denominator:**
        * Number of shares used in per share computation: Class A: 299,815, Class B: 46,182, Class C: 334,819
    * **Basic net income per share:** Class A: $59.15, Class B: $59.15, Class C: $59.15
    
    **Diluted Net Income Per Share**
    
    * **Numerator:**
        * Allocation of undistributed earnings for basic computation: Class A: $17,733, Class B: $2,732, Class C: $19,804
        * Reallocation of undistributed earnings as a result of conversion of Class B to Class A shares: Class A: $2,732, Class B: $0, Class C: $0
        * Reallocation of undistributed earnings: Class A: $(180), Class B: $(25), Class C: $180
        * Allocation of undistributed earnings: Class A: $20,285, Class B: $2,707, Class C: $19,984
    * **Denominator:**
        * Number of shares used in basic computation: Class A: 299,815, Class B: 46,182, Class C: 334,819
        * Add:
            * Conversion of Class B to Class A shares outstanding: Class A: 46,182, Class B: 0, Class C: 0
            * Restricted stock units and other contingently issuable shares: Class A: 87, Class B: 0, Class C: 6,125
        * Number of shares used in per share computation: Class A: 346,084, Class B: 46,182, Class C: 340,944
    * **Diluted net income per share:** Class A: $58.61, Class B: $58.61, Class C: $58.61 
    
    [91mCitation 2: Matched image path, page number and page text: 
    [0m
    [94mscore: [0m 0.62
    [94mfile_name: [0m google-10k-sample-part1.pdf
    [94mpath: [0m images/google-10k-sample-part1.pdf_image_2_0_15.jpeg
    [94mpage number: [0m 3
    [94mpage text: [0m Stock Performance Graphs
    The graph below matches Alphabet Inc. Class A's cumulative 5-year total
    stockholder return on common stock with the cumulative total returns of the S&P
    500 index, the NASDAQ Composite index, and the RDG Internet Composite
    index. The graph tracks the performance of a $100 investment in our common
    stock and in each index (with the reinvestment of all dividends) from December
    31, 2016 to December 31, 2021. The returns shown are based on historical
    results and are not intended to suggest future performance.
    
    [94mimage description: [0m The image is a line graph comparing the cumulative 5-year total return of Alphabet Inc. Class A common stock to the S&P 500 Index, the NASDAQ Composite Index, and the RDG Internet Composite Index. The graph shows that Alphabet Inc. Class A common stock outperformed the other three indices over the 5-year period. 
    
    [91mCitation 3: Matched image path, page number and page text: 
    [0m
    [94mscore: [0m 0.62
    [94mfile_name: [0m google-10k-sample-part1.pdf
    [94mpath: [0m images/google-10k-sample-part1.pdf_image_3_0_18.jpeg
    [94mpage number: [0m 4
    [94mpage text: [0m The graph below matches Alphabet Inc. Class A's cumulative 5-year total
    stockholder return on common stock with the cumulative total returns of the S&P
    500 index, the NASDAQ Composite index, and the RDG Internet Composite
    index. The graph tracks the performance of a $100 investment in our common
    stock and in each index (with the reinvestment of all dividends) from December
    31, 2017 to December 31, 2022. The returns shown are based on historical
    results and are not intended to suggest future performance.
    
    [94mimage description: [0m The image is a line graph comparing the cumulative 5-year total return among Alphabet Inc. Class A common stock, the S&P 500 Index, the NASDAQ Composite Index, and the RDG Internet Composite Index. The graph shows the performance of each investment from December 31, 2017 to December 31, 2022. 
    
    [91mCitation 4: Matched image path, page number and page text: 
    [0m
    [94mscore: [0m 0.62
    [94mfile_name: [0m google-10k-sample-part1.pdf
    [94mpath: [0m images/google-10k-sample-part1.pdf_image_4_0_21.jpeg
    [94mpage number: [0m 5
    [94mpage text: [0m Executive Overview
    The following table summarizes consolidated financial results for the years ended
    December 31, 2020 and 2021 unless otherwise specified (in millions, except for
    per share information and percentages):
    Revenues were $257.6 billion, an increase of 41%. The increase in
    revenues was primarily driven by Google Services and Google Cloud. The
    adverse effect of COVID-19 on 2020 advertising revenues also contributed
    to the year-over-year growth.
    Cost of revenues was $110.9 billion, an increase of 31%, primarily driven
    by increases in TAC and content acquisition costs.
    An overall increase in data centers and other operations costs was partially
    offset by a reduction in depreciation expense due to the change in the
    estimated useful life of our servers and certain network equipment. 
    Operating expenses were $68.0 billion, an increase of 20%, primarily
    driven by headcount growth, increases in advertising and promotional
    expenses and charges related to legal matters.
    
    [94mimage description: [0m The image is a table showing financial data for a company for the years 2020 and 2021. Here are the elements of the table:
    
    | Category | 2020 | 2021 | $ Change | % Change |
    |---|---|---|---|---|
    | Consolidated revenues | $182,527 | $257,637 | $75,110 | 41% |
    | Change in consolidated constant currency revenues |  |  |  | 39% |
    | Cost of revenues | $84,732 | $110,939 | $26,207 | 31% |
    | Operating expenses | $56,571 | $67,984 | $11,413 | 20% |
    | Operating income | $41,224 | $78,714 | $37,490 | 91% |
    | Operating margin | 23% | 31% |  | 8% |
    | Other income (expense), net | $6,858 | $12,020 | $5,162 | 75% |
    | Net Income | $40,269 | $76,033 | $35,764 | 89% |
    | Diluted EPS | $58.61 | $112.20 | $53.59 | 91% |
    | Number of Employees | 135,301 | 156,500 | 21,199 | 16% | 
    
    [91mCitation 5: Matched image path, page number and page text: 
    [0m
    [94mscore: [0m 0.62
    [94mfile_name: [0m google-10k-sample-part2.pdf
    [94mpath: [0m images/google-10k-sample-part2.pdf_image_0_1_8.jpeg
    [94mpage number: [0m 1
    [94mpage text: [0m source: https://abc.xyz/assets/investor/static/pdf/20220202_alphabet_10K.pdf
    source: https://abc.xyz/assets/9a/bd/838c917c4b4ab21f94e84c3c2c65/goog-10-k-q4-2022.pdf
    Note: Tables and figures are converted to images for demonstration purposes.
    Revenues by Geography
    The following table presents revenues by geography as a percentage of
    revenues, determined based on the addresses of our customers:
    The following table presents the foreign exchange effect on international
    revenues and total revenues (in millions, except percentages):
    EMEA revenue growth from 2020 to 2021 was favorably affected by foreign
    currency exchange rates, primarily due to the U.S. dollar weakening relative to
    the Euro and British pound.
    
    [94mimage description: [0m The image is a table that shows the revenue of a company in different regions for the years 2020 and 2021. It also shows the percentage change in revenue from the prior year.
    
    | Revenue Type | 2020 | 2021 | % Change from Prior Year |
    |---|---|---|---|
    | EMEA revenues | $55,370 | $79,107 | 43% |
    | EMEA constant currency revenues |  | $76,321 | 38% |
    | APAC revenues | $32,550 | $46,123 | 42% |
    | APAC constant currency revenues |  | $45,666 | 40% |
    | Other Americas revenues | $9,417 | $14,404 | 53% |
    | Other Americas constant currency revenues |  | $14,317 | 52% |
    | United States revenues | $85,014 | $117,854 | 39% |
    | Hedging gains (losses) | $176 | $149 |  |
    | Total revenues | $182,527 | $257,637 | 41% |
    | Revenues, excluding hedging effect | $182,351 | $257,488 |  |
    | Exchange rate effect |  | $(3,330) |  |
    | Total constant currency revenues |  | $254,158 | 39% | 
    
    [91mCitation 6: Matched image path, page number and page text: 
    [0m
    [94mscore: [0m 0.6
    [94mfile_name: [0m google-10k-sample-part2.pdf
    [94mpath: [0m images/google-10k-sample-part2.pdf_image_1_0_13.jpeg
    [94mpage number: [0m 2
    [94mpage text: [0m APAC revenue growth from 2020 to 2021 was favorably affected by foreign
    currency exchange rates, primarily due to the U.S. dollar weakening relative to
    the Australian dollar, partially offset by the U.S. dollar strengthening relative to
    the Japanese yen.
    Other Americas growth change from 2020 to 2021 was favorably affected by
    changes in foreign currency exchange rates, primarily due to the U.S. dollar
    weakening relative to the Canadian dollar, partially offset by the U.S. dollar
    strengthening relative to the Argentine peso and the Brazilian real.
    Costs and Expenses
    Cost of Revenues
    The following tables present cost of revenues, including TAC (in millions, except
    percentages):
    Cost of revenues increased $26.2 billion from 2020 to 2021. The increase was
    due to an increase in other cost of revenues and TAC of $13.4 billion and $12.8
    billion, respectively.
    The increase in TAC from 2020 to 2021 was due to an increase in TAC paid to
    distribution partners and to Google Network partners, primarily driven by growth
    in revenues subject to TAC. The TAC rate decreased from 22.3% to 21.8% from
    2020 to 2021 primarily due to a revenue mix shift from Google Network
    properties to Google Search & other properties.
    The TAC rate on Google Search & other properties revenues and the TAC rate
    on Google Network revenues were both substantially consistent from 2020 to
    2021. The increase in other cost of revenues from 2020 to 2021 was driven by
    
    [94mimage description: [0m The image is a table showing the total cost of revenues for a company for the years ended December 31, 2020 and 2021.
    
    | Item | 2020 | 2021 |
    |---|---|---|
    | TAC | $32,778 | $45,566 |
    | Other cost of revenues | $51,954 | $65,373 |
    | **Total cost of revenues** | **$84,732** | **$110,939** |
    | Total cost of revenues as a percentage of revenues | 46.4% | 43.1% | 
    
    [91mCitation 7: Matched image path, page number and page text: 
    [0m
    [94mscore: [0m 0.6
    [94mfile_name: [0m google-10k-sample-part2.pdf
    [94mpath: [0m images/google-10k-sample-part2.pdf_image_3_0_19.jpeg
    [94mpage number: [0m 4
    [94mpage text: [0m liquidation and dividend rights are identical, the undistributed earnings are
    allocated on a proportionate basis.
    In the years ended December 31, 2019, 2020 and 2021, the net income per
    share amounts are the same for Class A, Class B, and Class C stock because
    the holders of each class are entitled to equal per share dividends or distributions
    in liquidation in accordance with the Amended and Restated Certificate of
    Incorporation of Alphabet Inc.
    The following tables set forth the computation of basic and diluted net income per
    share of Class A, Class B, and Class C stock (in millions, except share amounts
    which are reflected in thousands and per share amounts):
    
    [94mimage description: [0m The image is a table showing the calculation of basic and diluted net income per share for different classes of shares (Class A, Class B, Class C) for the year ended December 31, 2019. 
    
    **Basic Net Income Per Share**
    
    | Item | Class A | Class B | Class C |
    |---|---|---|---|
    | Allocation of undistributed earnings | $14,846 | $2,307 | $17,190 |
    | Number of shares used in per share computation | 299,402 | 46,527 | 346,667 |
    | **Basic net income per share** | **$49.59** | **$49.59** | **$49.59** |
    
    **Diluted Net Income Per Share**
    
    | Item | Class A | Class B | Class C |
    |---|---|---|---|
    | Allocation of undistributed earnings for basic computation | $14,846 | $2,307 | $17,190 |
    | Reallocation of undistributed earnings as a result of conversion of Class B to Class A shares | 2,307 | 0 | 0 |
    | Reallocation of undistributed earnings | (126) | (20) | 126 |
    | **Allocation of undistributed earnings** | **$17,027** | **$2,287** | **$17,316** |
    | Number of shares used in basic computation | 299,402 | 46,527 | 346,667 |
    | Conversion of Class B to Class A shares outstanding | 46,527 | 0 | 0 |
    | Restricted stock units and other contingently issuable shares | 413 | 0 | 5,547 |
    | **Number of shares used in per share computation** | **346,342** | **46,527** | **352,214** |
    | **Diluted net income per share** | **$49.16** | **$49.16** | **$49.16** | 
    
    [91mCitation 8: Matched image path, page number and page text: 
    [0m
    [94mscore: [0m 0.55
    [94mfile_name: [0m google-10k-sample-part2.pdf
    [94mpath: [0m images/google-10k-sample-part2.pdf_image_4_1_23.jpeg
    [94mpage number: [0m 5
    [94mpage text: [0m Stock-Based Award Activities
    The weighted-average grant-date fair value of RSUs granted during the years
    ended December 31, 2019 and 2020 was $1,092.36 and $1,407.97, respectively.
    Total fair value of RSUs, as of their respective vesting dates, during the years
    ended December 31, 2019, 2020, and 2021 were $15.2 billion, $17.8 billion, and
    $28.8 billion, respectively. As of December 31, 2021, there was $25.8 billion of
    unrecognized compensation cost related to unvested employee RSUs. This
    amount is expected to be recognized over a weighted-average period of 2.5
    years. 401(k) Plans We have two 401(k) Savings Plans that qualify as deferred
    
    [94mimage description: [0m The image is a table summarizing the activity of unvested Alphabet RSUs for the year ended December 31, 2021. Here are the elements of the table:
    
    | Activity | Number of Shares | Weighted-Average Grant-Date Fair Value |
    |---|---|---|
    | Unvested as of December 31, 2020 | 19,288,793 | $1,262.13 |
    | Granted | 10,582,700 | $1,949.16 |
    | Vested | (11,209,486) | $1,345.98 |
    | Forfeited/canceled | (1,767,294) | $1,425.48 |
    | Unvested as of December 31, 2021 | 16,894,713 | $1,626.13 | 
    
    [91mCitation 9: Matched image path, page number and page text: 
    [0m
    [94mscore: [0m 0.54
    [94mfile_name: [0m google-10k-sample-part2.pdf
    [94mpath: [0m images/google-10k-sample-part2.pdf_image_5_0_26.jpeg
    [94mpage number: [0m 6
    [94mpage text: [0m salary arrangements under Section 401(k) of the Internal Revenue Code. Under
    these 401(k) Plans, matching contributions are based upon the amount of the
    employees contributions subject to certain limitations. We recognized expense of
    approximately $724 million, $855 million, and $916 million for the years ended
    December 31, 2019, 2020, and 2021, respectively. Note 14. Income Taxes
    Income from continuing operations before income taxes consisted of the
    following (in millions):
    Deferred Income Taxes
    Deferred income taxes reflect the net effects of temporary differences between
    the carrying amounts of assets and liabilities for financial reporting purposes and
    the amounts used for income tax purposes. Significant components of our
    deferred tax assets and liabilities were as follows (in millions):
    As of December 31, 2021, our federal, state, and foreign net operating loss
    carryforwards for income tax purposes were approximately $5.6 billion, $4.6
    billion, and $1.7 billion respectively. If not utilized, the federal net operating loss
    carryforwards will begin to expire in 2023, foreign net operating loss
    carryforwards will begin to expire in 2025 and the state net operating loss
    carryforwards will begin to expire in 2028. It is more likely than not that certain
    
    [94mimage description: [0m The image is a table showing the deferred tax assets and liabilities as of December 31, 2020 and 2021. 
    
    Here are the elements of the table:
    
    | Item | 2020 | 2021 |
    |---|---|---|
    | **Deferred tax assets:** | | |
    | Accrued employee benefits | 580 | 549 |
    | Accruals and reserves not currently deductible | 1,049 | 1,816 |
    | Tax credits | 3,723 | 5,179 |
    | Net operating losses | 1,085 | 1,790 |
    | Operating leases | 2,620 | 2,503 |
    | Intangible assets | 1,525 | 2,034 |
    | Other | 981 | 925 |
    | Total deferred tax assets | 11,563 | 14,796 |
    | Valuation allowance | (4,823) | (7,129) |
    | Total deferred tax assets net of valuation allowance | 6,740 | 7,667 |
    | **Deferred tax liabilities:** | | |
    | Property and equipment, net | (3,382) | (5,237) |
    | Net investment gains | (1,901) | (3,229) |
    | Operating leases | (2,354) | (2,228) |
    | Other | (1,580) | (946) |
    | Total deferred tax liabilities | (9,217) | (11,640) |
    | **Net deferred tax assets (liabilities)** | **(2,477)** | **(3,973)** | 
    
    [91mCitation 10: Matched image path, page number and page text: 
    [0m
    [94mscore: [0m 0.52
    [94mfile_name: [0m google-10k-sample-part2.pdf
    [94mpath: [0m images/google-10k-sample-part2.pdf_image_0_0_6.jpeg
    [94mpage number: [0m 1
    [94mpage text: [0m source: https://abc.xyz/assets/investor/static/pdf/20220202_alphabet_10K.pdf
    source: https://abc.xyz/assets/9a/bd/838c917c4b4ab21f94e84c3c2c65/goog-10-k-q4-2022.pdf
    Note: Tables and figures are converted to images for demonstration purposes.
    Revenues by Geography
    The following table presents revenues by geography as a percentage of
    revenues, determined based on the addresses of our customers:
    The following table presents the foreign exchange effect on international
    revenues and total revenues (in millions, except percentages):
    EMEA revenue growth from 2020 to 2021 was favorably affected by foreign
    currency exchange rates, primarily due to the U.S. dollar weakening relative to
    the Euro and British pound.
    
    [94mimage description: [0m The image is a table showing percentages for different regions of the world in the years 2020 and 2021. 
    
    Here are the elements of the table:
    
    | Region | Year Ended December 31, 2020 | Year Ended December 31, 2021 |
    |---|---|---|
    | United States | 47% | 46% |
    | EMEA | 30% | 31% |
    | APAC | 18% | 18% |
    | Other Americas | 5% | 5% | 
    
    


```
# Text citations

print_text_to_text_citation(
    matching_results_chunks_data,
    print_top=False,
    chunk_text=True,
)
```

    [91mCitation 1: Matched text: 
    [0m
    [94mscore: [0m 0.77
    [94mfile_name: [0m google-10k-sample-part1.pdf
    [94mpage_number: [0m 5
    [94mchunk_number: [0m 1
    [94mchunk_text: [0m Executive Overview
    The following table summarizes consolidated financial results for the years ended
    December 31, 2020 and 2021 unless otherwise specified (in millions, except for
    per share information and percentages):
    Revenues were $257.6 billion, an increase of 41%. The increase in
    revenues was primarily driven by Google Services and Google Cloud. The
    adverse effect of COVID-19 on 2020 advertising revenues also contributed
    to the year-over-year growth.
    Cost of revenues was $110.9 billion, an increase of 31%, primarily driven
    by increases in TAC and content acquisition costs.
    An overall increase in data centers and other operations costs was partially
    offset by a reduction in depreciation expense due to the change in the
    estimated useful life of our servers and certain network equipment. 
    Operating expenses were $68.0 billion, an increase of 20%, primarily
    driven by headcount growth, increases in advertising and promotional
    expenses and charges related to legal matters.
    
    [91mCitation 2: Matched text: 
    [0m
    [94mscore: [0m 0.72
    [94mfile_name: [0m google-10k-sample-part2.pdf
    [94mpage_number: [0m 2
    [94mchunk_number: [0m 2
    [94mchunk_text: [0m 21 was due to an increase in TAC paid to
    distribution partners and to Google Network partners, primarily driven by growth
    in revenues subject to TAC. The TAC rate decreased from 22.3% to 21.8% from
    2020 to 2021 primarily due to a revenue mix shift from Google Network
    properties to Google Search & other properties.
    The TAC rate on Google Search & other properties revenues and the TAC rate
    on Google Network revenues were both substantially consistent from 2020 to
    2021. The increase in other cost of revenues from 2020 to 2021 was driven by
    
    [91mCitation 3: Matched text: 
    [0m
    [94mscore: [0m 0.71
    [94mfile_name: [0m google-10k-sample-part2.pdf
    [94mpage_number: [0m 2
    [94mchunk_number: [0m 1
    [94mchunk_text: [0m APAC revenue growth from 2020 to 2021 was favorably affected by foreign
    currency exchange rates, primarily due to the U.S. dollar weakening relative to
    the Australian dollar, partially offset by the U.S. dollar strengthening relative to
    the Japanese yen.
    Other Americas growth change from 2020 to 2021 was favorably affected by
    changes in foreign currency exchange rates, primarily due to the U.S. dollar
    weakening relative to the Canadian dollar, partially offset by the U.S. dollar
    strengthening relative to the Argentine peso and the Brazilian real.
    Costs and Expenses
    Cost of Revenues
    The following tables present cost of revenues, including TAC (in millions, except
    percentages):
    Cost of revenues increased $26.2 billion from 2020 to 2021. The increase was
    due to an increase in other cost of revenues and TAC of $13.4 billion and $12.8
    billion, respectively.
    The increase in TAC from 2020 to 2021 was due to an increase in TAC paid to
    distribution partners and to Google Network partners, prima
    [91mCitation 4: Matched text: 
    [0m
    [94mscore: [0m 0.71
    [94mfile_name: [0m google-10k-sample-part2.pdf
    [94mpage_number: [0m 3
    [94mchunk_number: [0m 1
    [94mchunk_text: [0m increases in content acquisition costs primarily for YouTube, data center and
    other operations costs, and hardware costs. The increase in data center and
    Table of Contents Alphabet Inc. 36 other operations costs was partially offset by
    a reduction in depreciation expense due to the change in the estimated useful life
    of our servers and certain network equipment beginning in the first quarter of
    2021.
    Net Income Per Share
    We compute net income per share of Class A, Class B, and Class C stock using
    the two-class method. Basic net income per share is computed using the
    weighted-average number of shares outstanding during the period. Diluted net
    income per share is computed using the weighted-average number of shares and
    the effect of potentially dilutive securities outstanding during the period.
    Potentially dilutive securities consist of restricted stock units and other
    contingently issuable shares. The dilutive effect of outstanding restricted stock
    units and other contingently issuable 
    [91mCitation 5: Matched text: 
    [0m
    [94mscore: [0m 0.69
    [94mfile_name: [0m google-10k-sample-part1.pdf
    [94mpage_number: [0m 6
    [94mchunk_number: [0m 1
    [94mchunk_text: [0m Other information:
    Operating cash flow was $91.7 billion, primarily driven by revenues
    generated from our advertising products.
    Share repurchases were $50.3 billion, an increase of 62%. See Note 11 of
    the Notes to Consolidated Financial Statements included in Item 8 of this
    Annual Report on Form 10-K for further information.
    Capital expenditures, which primarily reflected investments in technical
    infrastructure, were $24.6 billion.
    In January 2021, we updated the useful lives of certain of our servers and
    network equipment, resulting in a reduction in depreciation expense of
    $2.6 billion recorded primarily in cost of revenues and R&D. See Note 1 of
    the Notes to Consolidated Financial Statements included in Item 8 of this
    Annual Report on Form 10-K for further information.
    Our acquisition of Fitbit closed in early January 2021, and the related
    revenues are included in Google other. See Note 8 of the Notes to
    Consolidated Financial Statements included in Item 8 of this Annual Report
    on F
    [91mCitation 6: Matched text: 
    [0m
    [94mscore: [0m 0.68
    [94mfile_name: [0m google-10k-sample-part1.pdf
    [94mpage_number: [0m 6
    [94mchunk_number: [0m 2
    [94mchunk_text: [0m te 8 of the Notes to
    Consolidated Financial Statements included in Item 8 of this Annual Report
    on Form 10-K for further information.
    
    On February 1, 2022, the Company announced that the Board of Directors
    had approved and declared a 20- for-one stock split in the form of a
    one-time special stock dividend on each share of the Companys Class A,
    Class B, and Class C stock. See Note 11 of the Notes to Consolidated
    Financial Statements included in Item 8 of this Annual Report on Form
    10-K for additional information.
    The Effect of COVID-19 on our Financial Results
    We began to observe the effect of COVID-19 on our financial results in March
    2020 when, despite an increase in users' search activity, our advertising
    revenues declined compared to the prior year. This was due to a shift of user
    search activity to less commercial topics and reduced spending by our
    advertisers. For the quarter ended June 30, 2020 our advertising revenues
    declined due to the continued effects of COVID-19 and the rel
    [91mCitation 7: Matched text: 
    [0m
    [94mscore: [0m 0.68
    [94mfile_name: [0m google-10k-sample-part2.pdf
    [94mpage_number: [0m 4
    [94mchunk_number: [0m 1
    [94mchunk_text: [0m liquidation and dividend rights are identical, the undistributed earnings are
    allocated on a proportionate basis.
    In the years ended December 31, 2019, 2020 and 2021, the net income per
    share amounts are the same for Class A, Class B, and Class C stock because
    the holders of each class are entitled to equal per share dividends or distributions
    in liquidation in accordance with the Amended and Restated Certificate of
    Incorporation of Alphabet Inc.
    The following tables set forth the computation of basic and diluted net income per
    share of Class A, Class B, and Class C stock (in millions, except share amounts
    which are reflected in thousands and per share amounts):
    
    [91mCitation 8: Matched text: 
    [0m
    [94mscore: [0m 0.67
    [94mfile_name: [0m google-10k-sample-part1.pdf
    [94mpage_number: [0m 3
    [94mchunk_number: [0m 1
    [94mchunk_text: [0m Stock Performance Graphs
    The graph below matches Alphabet Inc. Class A's cumulative 5-year total
    stockholder return on common stock with the cumulative total returns of the S&P
    500 index, the NASDAQ Composite index, and the RDG Internet Composite
    index. The graph tracks the performance of a $100 investment in our common
    stock and in each index (with the reinvestment of all dividends) from December
    31, 2016 to December 31, 2021. The returns shown are based on historical
    results and are not intended to suggest future performance.
    
    [91mCitation 9: Matched text: 
    [0m
    [94mscore: [0m 0.67
    [94mfile_name: [0m google-10k-sample-part2.pdf
    [94mpage_number: [0m 1
    [94mchunk_number: [0m 1
    [94mchunk_text: [0m source: https://abc.xyz/assets/investor/static/pdf/20220202_alphabet_10K.pdf
    source: https://abc.xyz/assets/9a/bd/838c917c4b4ab21f94e84c3c2c65/goog-10-k-q4-2022.pdf
    Note: Tables and figures are converted to images for demonstration purposes.
    Revenues by Geography
    The following table presents revenues by geography as a percentage of
    revenues, determined based on the addresses of our customers:
    The following table presents the foreign exchange effect on international
    revenues and total revenues (in millions, except percentages):
    EMEA revenue growth from 2020 to 2021 was favorably affected by foreign
    currency exchange rates, primarily due to the U.S. dollar weakening relative to
    the Euro and British pound.
    
    [91mCitation 10: Matched text: 
    [0m
    [94mscore: [0m 0.64
    [94mfile_name: [0m google-10k-sample-part1.pdf
    [94mpage_number: [0m 4
    [94mchunk_number: [0m 1
    [94mchunk_text: [0m The graph below matches Alphabet Inc. Class A's cumulative 5-year total
    stockholder return on common stock with the cumulative total returns of the S&P
    500 index, the NASDAQ Composite index, and the RDG Internet Composite
    index. The graph tracks the performance of a $100 investment in our common
    stock and in each index (with the reinvestment of all dividends) from December
    31, 2017 to December 31, 2022. The returns shown are based on historical
    results and are not intended to suggest future performance.
    
    

## Conclusions

Congratulations on making it through this multimodal RAG notebook!

While multimodal RAG can be quite powerful, note that it can face some limitations:

* **Data dependency:** Needs high-quality paired text and visuals.
* **Computationally demanding:** Processing multimodal data is resource-intensive.
* **Domain specific:** Models trained on general data may not shine in specialized fields like medicine.
* **Black box:** Understanding how these models work can be tricky, hindering trust and adoption.


Despite these challenges, multimodal RAG represents a significant step towards search and retrieval systems that can handle diverse, multimodal data.
