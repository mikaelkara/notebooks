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

# Building a DIY Multimodal Question Answering System with Vertex AI (A Beginner's Guide - Multimodal RAG)

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/qa-ops/building_DIY_multimodal_qa_system_with_mRAG.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fqa-ops%2Fbuilding_DIY_multimodal_qa_system_with_mRAG.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/qa-ops/building_DIY_multimodal_qa_system_with_mRAG.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/qa-ops/building_DIY_multimodal_qa_system_with_mRAG.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>    
</table>

| | |
|-|-|
|Author(s) | [Lavi Nigam](https://github.com/lavinigam-gcp) |

<div class="alert alert-block alert-warning">
<b>⚠️ This is a new version of the old mRAG notebook with modifications and new data. You can refer to the old notebook here:  ⚠️</b>
</div>

[**intro_multimodal_rag.ipynb**](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/retrieval-augmented-generation/intro_multimodal_rag.ipynb)

## Overview


This guide is your hands-on introduction to creating a question answering system that understands both text and images. We'll build this system from the ground up using Google's Vertex AI, giving you a clear understanding of how it works without relying on complex third-party tools.


## Why Build It Yourself?

Large Language Models (LLMs) are powerful, but they can seem like a "black box". By building our own system, we'll break open that box and explore the core concepts. This will give you the knowledge to customize and optimize every aspect of your question answering system, whether you ultimately choose to code everything yourself or use external libraries.


## What We'll Do:

* **Focus on Fundamentals**: We'll start with the essential design pattern of "Retrieval Augmented Generation" (RAG) – a way to find and use relevant information to answer questions.

* **Work with Text and Images**: We'll expand RAG to handle both text and images found in PDF documents. Future guides in this series will explore even more types of data, like videos and audio.

* **Use Vertex AI**: We'll only use Google's Vertex AI Embeddings API and Gemini API, ensuring you have complete control and understanding of the building blocks.


By the end of this guide, you'll have a solid foundation in building multimodal question answering systems, empowering you to create smarter applications that can understand and respond to a wider range of information.


### Gemini

Gemini is a family of generative AI models developed by Google DeepMind that is designed for multimodal use cases. The Gemini API gives you access to the Gemini 1.0 Pro Vision, Gemini 1.0 Pro & Gemini 1.5 Pro models.

### Comparing text-based and multimodal RAG

Multimodal RAG offers several advantages over text-based RAG:

1. **Enhanced knowledge access:** Multimodal RAG can access and process both textual and visual information, providing a richer and more comprehensive knowledge base for the LLM.
2. **Improved reasoning capabilities:** By incorporating visual cues, multimodal RAG can make better informed inferences across different types of data modalities.

This notebook shows you how to implement DIY RAG with Gemini API in Vertex AI
 and Vertex AI Embeddings API; [text embeddings](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text-embeddings), and [multimodal embeddings](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/multimodal-embeddings), to build a document search engine.

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
%pip install --upgrade --user google-cloud-aiplatform pymupdf rich
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

If you are running this notebook on Google Colab, run the following cell to authenticate your environment. This step is not required if you are using [Vertex AI Workbench](https://cloud.google.com/vertex-ai-workbench).



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
from IPython.display import display
from rich import print as rich_print
from rich.markdown import Markdown as rich_Markdown
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Image,
)
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import MultiModalEmbeddingModel
```

### Load the Gemini 1.5 Pro, Gemini 1.5 Pro Flash, Gemini 1.0 Pro Vision and Gemini 1.0 Pro models


Learn more about each models and their differences: [here](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-multimodal-prompts)

Learn about the quotas: [here](https://cloud.google.com/vertex-ai/generative-ai/docs/quotas)


```
# Instantiate text model with appropriate name and version
text_model = GenerativeModel("gemini-1.0-pro")  # works with text, code

# Multimodal models: Choose based on your performance/cost needs
multimodal_model_15 = GenerativeModel(
    "gemini-1.5-pro"
)  # works with text, code, images, video(with or without audio) and audio(mp3) with 1M input context - complex reasoning

# Multimodal models: Choose based on your performance/cost needs
multimodal_model_15_flash = GenerativeModel(
    "gemini-1.5-flash"
)  # works with text, code, images, video(with or without audio) and audio(mp3) with 1M input context - faster inference

multimodal_model_10 = GenerativeModel(
    "gemini-1.0-pro-vision-001"
)  # works with text, code, video(without audio) and images with 16k input context

# Load text embedding model from pre-trained source
text_embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

# Load multimodal embedding model from pre-trained source
multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained(
    "multimodalembedding"
)  # works with image, image with caption(~32 words), video, video with caption(~32 words)
```

#### Get documents and images from GCS


```
# download documents and images used in this notebook - will take ~30 sec
!gsutil -m -q rsync -r gs://github-repo/rag/intro_multimodal_rag/intro_multimodal_rag_v2 .
print("Download completed")
```

## Building metadata of documents containing text and images

### The data

The source data that you will use in this notebook are:


* [Google Cloud TPU Scaling blog](https://storage.googleapis.com/github-repo/rag/intro_multimodal_rag/data/Google%20Cloud%20TPU%20blog.pdf)
* [Gemini 1.5 Technical Report](https://storage.googleapis.com/github-repo/rag/intro_multimodal_rag/data/gemini_v1_5_report_technical.pdf)
* [Google Gemma Technical Paper](https://storage.googleapis.com/github-repo/rag/intro_multimodal_rag/data/gemma_technical_paper.pdf)
* [Med-Gemini Technical Paper](https://storage.googleapis.com/github-repo/rag/intro_multimodal_rag/data/med_gemini.pdf)



You can also use your data, by first deleting the current files and then placing your files in the `data/` folder.

### Import helper functions to build metadata

Before building the Multimodal Question Answering System with Vertex AI, it's important to have metadata of all the text and images in the document. For references and citations purposes, the metadata should contain essential elements, including page number, file name, image counter, and so on. Hence, as a next step, you will generate embeddings from the metadata, which is required to perform similarity search when querying the data.


```
from multimodal_qa_with_rag_utils import get_document_metadata, set_global_variable

set_global_variable("text_embedding_model", text_embedding_model)
set_global_variable("multimodal_embedding_model", multimodal_embedding_model)
```

 You can also view the code (`multimodal_qa_with_rag_utils`) [directly](https://storage.googleapis.com/github-repo/rag/intro_multimodal_rag/utils/multimodal_qa_with_rag_utils.py).

### Extract and store metadata of text and images from a document

You just imported a function called `get_document_metadata()`. This function extracts text and image metadata from a document, and returns two dataframes, namely *text_metadata* and *image_metadata*, as outputs. If you want to find out more about how `get_document_metadata()` function is implemented using Gemini and the embedding models, you can take look at the [source code](https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/use-cases/retrieval-augmented-generation/utils/intro_multimodal_rag_utils.py) directly.

The reason for extraction and storing both text metadata and image metadata is that just by using either of the two alone is not sufficient to come out with a relevent answer. For example, the relevant answers could be in visual form within a document, but text-based RAG won't be able to take into consideration of the visual images. You will also be exploring this example later in this notebook.


At the next step, you will use the function to extract and store metadata of text and images from a document. Please note that the following cell may take a few minutes to complete:

**NOTE: Given that we are loading 4 files with roughly 200 pages and approximately 84 images, the cell below will take approximately 7 minutes to run. We recommend loading pre-computed metadata instead.**


```
%%time
# Specify the PDF folder with multiple PDF ~7m

print(
    "Removing pre-existing images folder, since you are running the logic from scratch"
)
! rm -rf images/

pdf_folder_path = "data/"  # if running in Vertex AI Workbench.

# Specify the image description prompt. Change it
# image_description_prompt = """Explain what is going on in the image.
# If it's a table, extract all elements of the table.
# If it's a graph, explain the findings in the graph.
# Do not include any numbers that are not mentioned in the image.
# """

image_description_prompt = """You are a technical image analysis expert. You will be provided with various types of images extracted from documents like research papers, technical blogs, and more.
Your task is to generate concise, accurate descriptions of the images without adding any information you are not confident about.
Focus on capturing the key details, trends, or relationships depicted in the image.

Important Guidelines:
* Prioritize accuracy:  If you are uncertain about any detail, state "Unknown" or "Not visible" instead of guessing.
* Avoid hallucinations: Do not add information that is not directly supported by the image.
* Be specific: Use precise language to describe shapes, colors, textures, and any interactions depicted.
* Consider context: If the image is a screenshot or contains text, incorporate that information into your description.
"""


# Extract text and image metadata from the PDF document
text_metadata_df, image_metadata_df = get_document_metadata(
    multimodal_model_15,  # we are passing Gemini 1.5 Pro
    pdf_folder_path,
    image_save_dir="images",
    image_description_prompt=image_description_prompt,
    embedding_size=1408,
    # add_sleep_after_page = True, # Uncomment this if you are running into API quota issues
    # sleep_time_after_page = 5,
    add_sleep_after_document=True,  # Uncomment this if you are running into API quota issues
    sleep_time_after_document=5,  # Increase the value in seconds, if you are still getting quota issues. It will slow down the processing.
    # generation_config = # see next cell
    # safety_settings =  # see next cell
)

print("\n\n --- Completed processing. ---")
```

    Removing pre-exsisting images folder, since you are running the logic from scratch
    
    
     Processing the file: --------------------------------- data/gemma_technical_paper.pdf 
    
    
    Processing page: 1
    Extracting image from page: 1, saved as: images/gemma_technical_paper.pdf_image_0_0_153.jpeg
    Processing page: 2
    Processing page: 3
    Processing page: 4
    Processing page: 5
    Processing page: 6
    Processing page: 7
    Processing page: 8
    Processing page: 9
    Processing page: 10
    Processing page: 11
    Processing page: 12
    Processing page: 13
    Processing page: 14
    Processing page: 15
    Processing page: 16
    Processing page: 17
    
     
     Sleeping for  5  sec before processing the next document to avoid quota issues. You can disable it: "add_sleep_after_document = False"  
    
    
     Processing the file: --------------------------------- data/gemini_v1_5_report_technical.pdf 
    
    
    Processing page: 1
    Extracting image from page: 1, saved as: images/gemini_v1_5_report_technical.pdf_image_0_0_1910.jpeg
    Processing page: 2
    Processing page: 3
    Processing page: 4
    Processing page: 5
    Extracting image from page: 5, saved as: images/gemini_v1_5_report_technical.pdf_image_4_0_142.jpeg
    Extracting image from page: 5, saved as: images/gemini_v1_5_report_technical.pdf_image_4_1_143.jpeg
    Processing page: 6
    Extracting image from page: 6, saved as: images/gemini_v1_5_report_technical.pdf_image_5_0_148.jpeg
    Extracting image from page: 6, saved as: images/gemini_v1_5_report_technical.pdf_image_5_1_149.jpeg
    Processing page: 7
    Processing page: 8
    Processing page: 9
    Processing page: 10
    Processing page: 11
    Processing page: 12
    Processing page: 13
    Processing page: 14
    Processing page: 15
    Processing page: 16
    Processing page: 17
    Processing page: 18
    Processing page: 19
    Processing page: 20
    Processing page: 21
    Processing page: 22
    Processing page: 23
    Processing page: 24
    Processing page: 25
    Extracting image from page: 25, saved as: images/gemini_v1_5_report_technical.pdf_image_24_0_488.jpeg
    Processing page: 26
    Processing page: 27
    Processing page: 28
    Processing page: 29
    Processing page: 30
    Processing page: 31
    Processing page: 32
    Processing page: 33
    Processing page: 34
    Processing page: 35
    Processing page: 36
    Processing page: 37
    Processing page: 38
    Processing page: 39
    Processing page: 40
    Processing page: 41
    Processing page: 42
    Processing page: 43
    Processing page: 44
    Processing page: 45
    Extracting image from page: 45, saved as: images/gemini_v1_5_report_technical.pdf_image_44_0_606.jpeg
    Processing page: 46
    Processing page: 47
    Processing page: 48
    Processing page: 49
    Processing page: 50
    Processing page: 51
    Processing page: 52
    Processing page: 53
    Processing page: 54
    Processing page: 55
    Processing page: 56
    Processing page: 57
    Processing page: 58
    
     
     Sleeping for  5  sec before processing the next document to avoid quota issues. You can disable it: "add_sleep_after_document = False"  
    
    
     Processing the file: --------------------------------- data/med_gemini.pdf 
    
    
    Processing page: 1
    Extracting image from page: 1, saved as: images/med_gemini.pdf_image_0_0_46.jpeg
    Extracting image from page: 1, saved as: images/med_gemini.pdf_image_0_1_48.jpeg
    Processing page: 2
    Extracting image from page: 2, saved as: images/med_gemini.pdf_image_1_0_73.jpeg
    Extracting image from page: 2, saved as: images/med_gemini.pdf_image_1_1_75.jpeg
    Extracting image from page: 2, saved as: images/med_gemini.pdf_image_1_2_77.jpeg
    Extracting image from page: 2, saved as: images/med_gemini.pdf_image_1_3_79.jpeg
    Extracting image from page: 2, saved as: images/med_gemini.pdf_image_1_4_81.jpeg
    Extracting image from page: 2, saved as: images/med_gemini.pdf_image_1_5_83.jpeg
    Extracting image from page: 2, saved as: images/med_gemini.pdf_image_1_6_84.jpeg
    Extracting image from page: 2, saved as: images/med_gemini.pdf_image_1_7_86.jpeg
    Processing page: 3
    Processing page: 4
    Processing page: 5
    Processing page: 6
    Processing page: 7
    Processing page: 8
    Processing page: 9
    Processing page: 10
    Processing page: 11
    Processing page: 12
    Processing page: 13
    Processing page: 14
    Processing page: 15
    Processing page: 16
    Extracting image from page: 16, saved as: images/med_gemini.pdf_image_15_0_464.jpeg
    Extracting image from page: 16, saved as: images/med_gemini.pdf_image_15_1_465.jpeg
    Processing page: 17
    Extracting image from page: 17, saved as: images/med_gemini.pdf_image_16_0_480.jpeg
    Extracting image from page: 17, saved as: images/med_gemini.pdf_image_16_1_481.jpeg
    Processing page: 18
    Processing page: 19
    Extracting image from page: 19, saved as: images/med_gemini.pdf_image_18_0_573.jpeg
    Extracting image from page: 19, saved as: images/med_gemini.pdf_image_18_1_575.jpeg
    Extracting image from page: 19, saved as: images/med_gemini.pdf_image_18_2_576.jpeg
    Extracting image from page: 19, saved as: images/med_gemini.pdf_image_18_3_578.jpeg
    Processing page: 20
    Extracting image from page: 20, saved as: images/med_gemini.pdf_image_19_0_613.jpeg
    Extracting image from page: 20, saved as: images/med_gemini.pdf_image_19_1_615.jpeg
    Extracting image from page: 20, saved as: images/med_gemini.pdf_image_19_2_617.jpeg
    Extracting image from page: 20, saved as: images/med_gemini.pdf_image_19_3_618.jpeg
    Processing page: 21
    Processing page: 22
    Extracting image from page: 22, saved as: images/med_gemini.pdf_image_21_0_681.jpeg
    Processing page: 23
    Processing page: 24
    Extracting image from page: 24, saved as: images/med_gemini.pdf_image_23_0_726.jpeg
    Extracting image from page: 24, saved as: images/med_gemini.pdf_image_23_1_727.jpeg
    Extracting image from page: 24, saved as: images/med_gemini.pdf_image_23_2_728.jpeg
    Processing page: 25
    Extracting image from page: 25, saved as: images/med_gemini.pdf_image_24_0_752.jpeg
    Extracting image from page: 25, saved as: images/med_gemini.pdf_image_24_1_754.jpeg
    Extracting image from page: 25, saved as: images/med_gemini.pdf_image_24_2_780.jpeg
    Extracting image from page: 25, saved as: images/med_gemini.pdf_image_24_3_782.jpeg
    Extracting image from page: 25, saved as: images/med_gemini.pdf_image_24_4_783.jpeg
    Processing page: 26
    Extracting image from page: 26, saved as: images/med_gemini.pdf_image_25_0_812.jpeg
    Extracting image from page: 26, saved as: images/med_gemini.pdf_image_25_1_813.jpeg
    Extracting image from page: 26, saved as: images/med_gemini.pdf_image_25_2_815.jpeg
    Extracting image from page: 26, saved as: images/med_gemini.pdf_image_25_3_817.jpeg
    Extracting image from page: 26, saved as: images/med_gemini.pdf_image_25_4_819.jpeg
    Extracting image from page: 26, saved as: images/med_gemini.pdf_image_25_5_820.jpeg
    Processing page: 27
    Extracting image from page: 27, saved as: images/med_gemini.pdf_image_26_0_849.jpeg
    Extracting image from page: 27, saved as: images/med_gemini.pdf_image_26_1_851.jpeg
    Extracting image from page: 27, saved as: images/med_gemini.pdf_image_26_2_853.jpeg
    Extracting image from page: 27, saved as: images/med_gemini.pdf_image_26_3_855.jpeg
    Processing page: 28
    Extracting image from page: 28, saved as: images/med_gemini.pdf_image_27_0_890.jpeg
    Extracting image from page: 28, saved as: images/med_gemini.pdf_image_27_1_891.jpeg
    Extracting image from page: 28, saved as: images/med_gemini.pdf_image_27_2_892.jpeg
    Extracting image from page: 28, saved as: images/med_gemini.pdf_image_27_3_893.jpeg
    Extracting image from page: 28, saved as: images/med_gemini.pdf_image_27_4_894.jpeg
    Extracting image from page: 28, saved as: images/med_gemini.pdf_image_27_5_895.jpeg
    Extracting image from page: 28, saved as: images/med_gemini.pdf_image_27_6_896.jpeg
    Extracting image from page: 28, saved as: images/med_gemini.pdf_image_27_7_897.jpeg
    Extracting image from page: 28, saved as: images/med_gemini.pdf_image_27_8_899.jpeg
    Extracting image from page: 28, saved as: images/med_gemini.pdf_image_27_9_901.jpeg
    Extracting image from page: 28, saved as: images/med_gemini.pdf_image_27_10_903.jpeg
    Extracting image from page: 28, saved as: images/med_gemini.pdf_image_27_11_905.jpeg
    Extracting image from page: 28, saved as: images/med_gemini.pdf_image_27_12_907.jpeg
    Extracting image from page: 28, saved as: images/med_gemini.pdf_image_27_13_909.jpeg
    Extracting image from page: 28, saved as: images/med_gemini.pdf_image_27_14_910.jpeg
    Extracting image from page: 28, saved as: images/med_gemini.pdf_image_27_15_911.jpeg
    Extracting image from page: 28, saved as: images/med_gemini.pdf_image_27_16_912.jpeg
    Extracting image from page: 28, saved as: images/med_gemini.pdf_image_27_17_913.jpeg
    Extracting image from page: 28, saved as: images/med_gemini.pdf_image_27_18_914.jpeg
    Processing page: 29
    Processing page: 30
    Processing page: 31
    Processing page: 32
    Processing page: 33
    Processing page: 34
    Processing page: 35
    Processing page: 36
    Processing page: 37
    Processing page: 38
    Processing page: 39
    Processing page: 40
    Processing page: 41
    Processing page: 42
    Processing page: 43
    Processing page: 44
    Processing page: 45
    Processing page: 46
    Processing page: 47
    Processing page: 48
    Processing page: 49
    Processing page: 50
    Processing page: 51
    Processing page: 52
    Processing page: 53
    Processing page: 54
    Processing page: 55
    Processing page: 56
    Extracting image from page: 56, saved as: images/med_gemini.pdf_image_55_0_1493.jpeg
    Extracting image from page: 56, saved as: images/med_gemini.pdf_image_55_1_1494.jpeg
    Extracting image from page: 56, saved as: images/med_gemini.pdf_image_55_2_1495.jpeg
    Extracting image from page: 56, saved as: images/med_gemini.pdf_image_55_3_1496.jpeg
    Extracting image from page: 56, saved as: images/med_gemini.pdf_image_55_4_1497.jpeg
    Extracting image from page: 56, saved as: images/med_gemini.pdf_image_55_5_1498.jpeg
    Extracting image from page: 56, saved as: images/med_gemini.pdf_image_55_6_1499.jpeg
    Processing page: 57
    Processing page: 58
    
     
     Sleeping for  5  sec before processing the next document to avoid quota issues. You can disable it: "add_sleep_after_document = False"  
    
    
     Processing the file: --------------------------------- data/Google Cloud TPU blog.pdf 
    
    
    Processing page: 1
    Processing page: 2
    Extracting image from page: 2, saved as: images/Google Cloud TPU blog.pdf_image_1_0_10.jpeg
    Processing page: 3
    Processing page: 4
    Extracting image from page: 4, saved as: images/Google Cloud TPU blog.pdf_image_3_0_18.jpeg
    Processing page: 5
    Processing page: 6
    Processing page: 7
    Extracting image from page: 7, saved as: images/Google Cloud TPU blog.pdf_image_6_0_35.jpeg
    Processing page: 8
    Processing page: 9
    Processing page: 10
    Processing page: 11
    Processing page: 12
    Extracting image from page: 12, saved as: images/Google Cloud TPU blog.pdf_image_11_0_62.jpeg
    Extracting image from page: 12, saved as: images/Google Cloud TPU blog.pdf_image_11_1_63.jpeg
    Processing page: 13
    Extracting image from page: 13, saved as: images/Google Cloud TPU blog.pdf_image_12_0_66.jpeg
    Processing page: 14
    Extracting image from page: 14, saved as: images/Google Cloud TPU blog.pdf_image_13_0_69.jpeg
    Processing page: 15
    Extracting image from page: 15, saved as: images/Google Cloud TPU blog.pdf_image_14_0_72.jpeg
    Processing page: 16
    Extracting image from page: 16, saved as: images/Google Cloud TPU blog.pdf_image_15_0_75.jpeg
    Processing page: 17
    Processing page: 18
    
     
     Sleeping for  5  sec before processing the next document to avoid quota issues. You can disable it: "add_sleep_after_document = False"  
    
    
     --- Completed processing. ---
    CPU times: user 20.1 s, sys: 1.22 s, total: 21.3 s
    Wall time: 5min 49s
    

If you would like to pass additional parameters to Gemini while building metadata, here are some options:


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

### Load pre-computed metadata of text and images from source document

**If you are facing constant issues with Quota or want to focus on the outputs, you should load pre-computed metadata.**


```
# import pickle

# # Load the pickle file
# with open("mrag_metadata.pkl", "rb") as f:
#     data = pickle.load(f)

# # Extract the DataFrames
# text_metadata_df = data["text_metadata"]
# image_metadata_df = data["image_metadata"]
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





  <div id="df-474994ad-a87b-4c92-8abe-d9671ba69913" class="colab-df-container">
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
      <td>gemma_technical_paper.pdf</td>
      <td>1</td>
      <td>2024-02-21\nGemma: Open Models Based on Gemini...</td>
      <td>[0.029665455222129822, 0.043536581099033356, -...</td>
      <td>1</td>
      <td>2024-02-21\nGemma: Open Models Based on Gemini...</td>
      <td>[0.035918254405260086, 0.039294395595788956, -...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gemma_technical_paper.pdf</td>
      <td>1</td>
      <td>2024-02-21\nGemma: Open Models Based on Gemini...</td>
      <td>[0.029665455222129822, 0.043536581099033356, -...</td>
      <td>2</td>
      <td>Gemma, a family of open models\nbased on Googl...</td>
      <td>[0.03788634389638901, 0.05161239951848984, -0....</td>
    </tr>
    <tr>
      <th>2</th>
      <td>gemma_technical_paper.pdf</td>
      <td>1</td>
      <td>2024-02-21\nGemma: Open Models Based on Gemini...</td>
      <td>[0.029665455222129822, 0.043536581099033356, -...</td>
      <td>3</td>
      <td>d for dialogue, instruction-following, help-\n...</td>
      <td>[0.045551132410764694, 0.04680870845913887, -0...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>gemma_technical_paper.pdf</td>
      <td>1</td>
      <td>2024-02-21\nGemma: Open Models Based on Gemini...</td>
      <td>[0.029665455222129822, 0.043536581099033356, -...</td>
      <td>4</td>
      <td>ce (Cobbe et al.,\n2021; Hendrycks et al., 202...</td>
      <td>[0.017123950645327568, 0.05736316367983818, -0...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>gemma_technical_paper.pdf</td>
      <td>1</td>
      <td>2024-02-21\nGemma: Open Models Based on Gemini...</td>
      <td>[0.029665455222129822, 0.043536581099033356, -...</td>
      <td>5</td>
      <td>rigorous evaluation and\nanalysis of current ...</td>
      <td>[0.03978141397237778, 0.029347488656640053, -0...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-474994ad-a87b-4c92-8abe-d9671ba69913')"
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
        document.querySelector('#df-474994ad-a87b-4c92-8abe-d9671ba69913 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-474994ad-a87b-4c92-8abe-d9671ba69913');
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


<div id="df-a6be63c4-404b-4067-a1c8-66c9f7243946">
  <button class="colab-df-quickchart" onclick="quickchart('df-a6be63c4-404b-4067-a1c8-66c9f7243946')"
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
        document.querySelector('#df-a6be63c4-404b-4067-a1c8-66c9f7243946 button');
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





  <div id="df-3a3991d7-48fb-4561-98f7-1988e37522e2" class="colab-df-container">
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
      <td>gemma_technical_paper.pdf</td>
      <td>1</td>
      <td>1</td>
      <td>images/gemma_technical_paper.pdf_image_0_0_153...</td>
      <td>The image shows the Google logo in its standar...</td>
      <td>[-0.0264111869, 0.0349279381, -0.0139624262, -...</td>
      <td>[0.025234034284949303, 0.040900733321905136, -...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gemini_v1_5_report_technical.pdf</td>
      <td>1</td>
      <td>1</td>
      <td>images/gemini_v1_5_report_technical.pdf_image_...</td>
      <td>The image shows the Google logo in its standar...</td>
      <td>[-0.0264111571, 0.0349279791, -0.0139624029, -...</td>
      <td>[0.025234034284949303, 0.040900733321905136, -...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>gemini_v1_5_report_technical.pdf</td>
      <td>5</td>
      <td>1</td>
      <td>images/gemini_v1_5_report_technical.pdf_image_...</td>
      <td>The image depicts a user prompt asking for the...</td>
      <td>[-0.0133379065, 0.0248864833, -0.0143319033, 0...</td>
      <td>[0.04414892569184303, 0.008230694569647312, -0...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>gemini_v1_5_report_technical.pdf</td>
      <td>5</td>
      <td>2</td>
      <td>images/gemini_v1_5_report_technical.pdf_image_...</td>
      <td>The image depicts a system designed to transla...</td>
      <td>[-0.0333386473, 0.0493280329, 0.00310201081, 0...</td>
      <td>[-0.0197904035449028, 0.011377139948308468, -0...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>gemini_v1_5_report_technical.pdf</td>
      <td>6</td>
      <td>1</td>
      <td>images/gemini_v1_5_report_technical.pdf_image_...</td>
      <td>The image depicts a user interface with three ...</td>
      <td>[-0.0436088592, 0.00902390946, -0.0243023187, ...</td>
      <td>[0.018790962174534798, 0.014835771173238754, -...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-3a3991d7-48fb-4561-98f7-1988e37522e2')"
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
        document.querySelector('#df-3a3991d7-48fb-4561-98f7-1988e37522e2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3a3991d7-48fb-4561-98f7-1988e37522e2');
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


<div id="df-d597476c-0c78-4ac2-81f5-6f68989c4da0">
  <button class="colab-df-quickchart" onclick="quickchart('df-d597476c-0c78-4ac2-81f5-6f68989c4da0')"
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
        document.querySelector('#df-d597476c-0c78-4ac2-81f5-6f68989c4da0 button');
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
from multimodal_qa_with_rag_utils import (
    display_images,
    get_answer_from_qa_system,
    get_gemini_response,
    get_similar_image_from_query,
    get_similar_text_from_query,
    print_text_to_image_citation,
    print_text_to_text_citation,
)
```

Before implementing a Multimodal Question Answering System with Vertex AI, let's explore what you can achieve with just text or image embeddings. This will set the foundation for implementing a multimodal Retrieval Augmented Generation (RAG) system, which you will do later in this notebook.

You can also use these essential elements together to build applications for multimodal use cases, extracting meaningful information from documents.

## Text Search

Let's start the search with a simple question and see if the simple text search using text embeddings can answer it. The expected answer is to show the value of basic and diluted net income per share of Google for different share types.



```
query = "What are various med-gemini medical benchmarks that shows its performance relative to other models?"
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
print_text_to_text_citation(
    matching_results_text, print_top=True, chunk_text=True
)  # print_top=False to see all text matches
```

    [91mCitation 1: Matched text: 
    [0m
    [94mscore: [0m 0.78
    [94mfile_name: [0m med_gemini.pdf
    [94mpage_number: [0m 29
    [94mchunk_number: [0m 1
    [94mchunk_text: [0m Capabilities of Gemini Models in Medicine
    5. Discussion
    Med-Gemini, built upon the Gemini models, demonstrates significant advancements in clinical
    reasoning, multimodal understanding, and long-context processing within the medical domain. This
    is evidenced by its strong performance across a diverse range of 25 tasks spanning 14 medical
    benchmarks, encompassing medical knowledge, clinical reasoning, genomics, waveforms, medical
    imaging, health records and videos.
    MedQA performance
    Notably, Med-Gemini-L 1.0 achieves a new SoTA on MedQA (USMLE), a
    popular benchmark for medical question answering with the use of self-training based fine-tuning
    and search integration. Our thorough relabeling of the MedQA test set (performed by attending
    clinicians) reveals important insights. While MedQA (USMLE) is a useful benchmark for assessing
    medical knowledge and reasoning, it is essential to acknowledge its limitations. We discover that
    approximately 4% of the questions contain missing information, 
    

### Get answer with text-RAG


```
# All relevant text chunk found across documents based on user query
context = "\n".join(
    [value["chunk_text"] for key, value in matching_results_text.items()]
)

prompt = f"""Answer the question with the given context. If the specific answer is not in the context, please answer "I don't know".
Question: {query}
Context: {context}
Answer:
"""
```


```
safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
}
```


```
%%time
# Generate response with Gemini 1.5 Pro
print("\n **** Result: ***** \n")

rich_Markdown(
    get_gemini_response(
        multimodal_model_15,
        model_input=prompt,
        stream=True,
        safety_settings=safety_settings,
        generation_config=GenerationConfig(temperature=1, max_output_tokens=8192),
    )
)
```

    
     **** Result: ***** 
    
    CPU times: user 24 ms, sys: 3.12 ms, total: 27.1 ms
    Wall time: 1.25 s
    




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">The passage mentions that Med-Gemini models are evaluated on 14 medical benchmarks, but it does not enumerate the  
specific benchmarks. Therefore, I cannot provide a list of the benchmarks.                                         
</pre>





```
%%time
# Generate response with Gemini 1.5 FLash
rich_Markdown(
    get_gemini_response(
        multimodal_model_15_flash,
        model_input=prompt,
        stream=True,
        safety_settings=safety_settings,
        generation_config=GenerationConfig(temperature=0.1),
    )
)
```

    CPU times: user 26.9 ms, sys: 3.69 ms, total: 30.6 ms
    Wall time: 964 ms
    




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">The context mentions that Med-Gemini was evaluated on 25 tasks across 14 medical benchmarks, but it doesn't list   
the specific benchmarks. Therefore, I don't know the various med-gemini medical benchmarks that show its           
performance relative to other models.                                                                              
</pre>




### Search similar images with text query

Since plain text search and RAG didn't provide the detailed answer and the information may be visually represented in a table or another image format, you can use multimodal capability of Gemini 1.0 Pro Vision or Gemini 1.5 Pro model for the similar task.

The goal here also is to find an image similar to the text query. You may also print the citations to verify.


```
query = "What are various med-gemini medical benchmarks that shows its performance relative to other models?"
```


```
matching_results_image = get_similar_image_from_query(
    text_metadata_df,
    image_metadata_df,
    query=query,
    column_name="text_embedding_from_image_description",  # Use image description text embedding
    image_emb=False,  # Use text embedding instead of image embedding
    top_n=5,
    embedding_size=1408,
)

# Markdown(print_text_to_image_citation(matching_results_image, print_top=True))
print("\n **** Result: ***** \n")

# Display the top matching image
display_images(
    [
        matching_results_image[0]["img_path"],
        matching_results_image[1]["img_path"],
        matching_results_image[2]["img_path"],
        matching_results_image[3]["img_path"],
    ],
    resize_ratio=0.3,
)
```

    
     **** Result: ***** 
    
    


    
![png](output_60_1.png)
    


    
    
    


    
![png](output_60_3.png)
    


    
    
    


    
![png](output_60_5.png)
    


    
    
    


    
![png](output_60_7.png)
    


    
    
    


```
%%time

print("\n **** Result: ***** \n")

instruction = f"""Answer the question and explain results with the given Image:
Question: {query}
Image:
"""

# Prepare the model input
model_input = [
    instruction,
    # passing all matched images to Gemini
    "Image:",
    matching_results_image[0]["image_object"],
    "Description:",
    matching_results_image[0]["image_description"],
    "Image:",
    matching_results_image[1]["image_object"],
    "Description:",
    matching_results_image[1]["image_description"],
    "Image:",
    matching_results_image[2]["image_object"],
    "Description:",
    matching_results_image[2]["image_description"],
    "Image:",
    matching_results_image[3]["image_object"],
    "Description:",
    matching_results_image[3]["image_description"],
]

# Generate Gemini response with streaming output
rich_Markdown(
    get_gemini_response(
        multimodal_model_15,
        model_input=model_input,
        stream=True,
        safety_settings=safety_settings,
        generation_config=GenerationConfig(temperature=1, max_output_tokens=8192),
    )
)
```

    
     **** Result: ***** 
    
    CPU times: user 136 ms, sys: 22.6 ms, total: 158 ms
    Wall time: 13.7 s
    




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">The images provide a comprehensive view of Med-Gemini's performance across various medical benchmarks relative to  
other models, primarily focusing on accuracy and expert preference.                                                

Here's a breakdown of the insights gleaned from each image and what they tell us about Med-Gemini's capabilities:  

<span style="font-weight: bold">Image 1: NEJM CPC Accuracy</span>                                                                                         

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Superior Performance:</span> Med-Gemini, both with and without search augmentation, consistently outperforms prior     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>state-of-the-art (SOTA) models and clinician baselines on the NEJM CPC (New England Journal of Medicine Clinical
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>Problem Solving Challenge) task.                                                                                
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Search Augmentation Benefits:</span>  Adding search capabilities boosts the accuracy for both Med-Gemini and clinician 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>performance, highlighting the value of information retrieval in medical reasoning tasks.                        

<span style="font-weight: bold">Image 2: Performance Across Tasks</span>                                                                                  

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Generally Strong Performance:</span> While specific tasks are unnamed, Med-Gemini demonstrates superior performance    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>compared to the previous SOTA in most categories.                                                               
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">GPT-4 Comparison:</span> Interestingly, the best GPT-4 method lags behind both Med-Gemini and the prior SOTA in most   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>tasks. This suggests a specialized model like Med-Gemini, fine-tuned on medical data, can outperform more       
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>general-purpose language models in the medical domain.                                                          

<span style="font-weight: bold">Image 3: Expert Preference</span>                                                                                         

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">High Subjective Preference:</span>  Med-Gemini receives a high percentage of "preferred" responses from experts in     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>tasks like Doctor Referral Generation and Medical Simplification. This suggests the model's outputs align well  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>with expert judgment in these areas.                                                                            
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Room for Improvement:</span>  In Medical Summarization, while Med-Gemini is still favored, there's a notable proportion
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>of "Expert Preferred" responses. This highlights an area where further refinement could enhance alignment with  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>expert preferences.                                                                                             

<span style="font-weight: bold">Image 4: Accuracy in Gene &amp; DNA Tasks</span>                                                                              

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Strengths in Complex Tasks:</span> Med-Gemini showcases strong performance in tasks like "Protein-coding genes" and    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>"Human genome TF regulation," demonstrating proficiency in handling intricate biological information.           
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Areas for Attention:</span>  Tasks like "Gene name conversion" and "Multi-species DNA alignment" show a higher         
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>"Abstain" rate for Med-Gemini, indicating potential areas where the model may be less confident or the tasks are
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>particularly challenging.                                                                                       

<span style="font-weight: bold">Overall, the images paint a picture of Med-Gemini as a high-performing medical AI model.</span> It demonstrates:          

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Superior Accuracy:</span> Outperforming previous SOTA models and clinician baselines in challenging medical tasks.     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Effective Search Augmentation:</span>  Benefitting from incorporating search to improve accuracy further.              
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Expert Alignment:</span>  Generating responses often preferred by medical experts, particularly in referral generation 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>and medical simplification.                                                                                     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Specialization Advantage:</span>  Outperforming even powerful general-purpose language models like GPT-4 in specific   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>medical domains.                                                                                                

However, the analysis also reveals areas for potential improvement:                                                

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Enhancing Summarization:</span> Refining the model to better align with expert preferences in summarization tasks.     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Addressing Abstentions:</span>  Investigating and potentially improving performance in tasks where the model frequently
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>abstains, suggesting lower confidence or higher task complexity.                                                

In conclusion, Med-Gemini shows significant promise as a valuable tool in various medical applications. Continuous 
evaluation and refinement in areas needing improvement will further solidify its role in assisting healthcare      
professionals and researchers.                                                                                     
</pre>





```
## you can check the citations to probe further.
## check the "image description:" which is a description extracted through Gemini which helped search our query.
rich_print(print_text_to_image_citation(matching_results_image, print_top=True))
```

    [91mCitation 1: Matched image path, page number and page text: 
    [0m
    [94mscore: [0m 0.66
    [94mfile_name: [0m med_gemini.pdf
    [94mpath: [0m images/med_gemini.pdf_image_15_0_464.jpeg
    [94mpage number: [0m 16
    [94mpage text: [0m Capabilities of Gemini Models in Medicine
    (a) NEJM CPC
    (b) GeneTuring
    Figure 3 | Generalization of Med-Gemini-L 1.0 with web search to two additional text-based benchmarks. (a):
    Comparison of Med-Gemini-L 1.0s top-k accuracy on the NEJM CPC benchmark with prior SoTA LLMs and clinicians, with
    and without search. (b): Comparison between Med-Gemini-L 1.0 and SoTA models on the GeneTuring dataset modules.
    The bars represent the proportion of correct, incorrect, and abstention responses for each model.
    Revisiting MedQA (USMLE) labels
    MedQA (USMLE) is a popular benchmark for assessing the
    capabilities of LLMs in the medical domain. However, some MedQA test questions have missing
    information such as figures or lab results, and potentially outdated ground-truth answers. To address
    these concerns, we conduct a complete relabeling of the MedQA (USMLE) test set. Specifically, we
    recruit at least three US physicians to re-annotate each question, asking them to answer the question
    and evaluate the provided ground-truth answer. We also ask them to identify if there was any missing
    information in the questions. Following Stutz et al. (2023), we characterize the questions to exclude
    due to missing information or label errors by bootstrapping votes from committees of three raters per
    question. We additionally identify ambiguous questions as those allowing multiple correct answers
    (more details can be found in Appendix C.2).
    Figure 4b shows that, on average across bootstrapped committees, 3.8% of questions include
    missing information, following the unanimous vote of bootstrapped committees. Additionally, 2.9%
    likely include label errors. Another 0.7% are ambiguous. Excluding these questions is supported
    by high inter-rater agreement of 94%, 87.6%, and 94.6%, respectively. Importantly, Med-Gemini-L
    1.0s mistakes can be attributed disproportionately to these questions; our entropy-based uncertainty
    score also tends to be higher on these question (t-test, -value=0.033). Filtering both types improves
    accuracy from 91.1% to 91.8%  0.2%. Using majority instead of unanimous votes further improves
    accuracy to 92.9%  0.38% by discarding up to 20.9% of the uncertain questions.
    4.1.1. Performance on long-form medical text generation
    Med-Gemini-M 1.0 demonstrates the ability to generate long-form text for three challenging real-world
    use cases - after-visit clinical summaries, doctor referral letter generation and medical simplification.
    In side-by-side comparisons, Med-Gemini-M 1.0s responses are considered as good or better than
    expert responses more than half the time by clinician raters across the three tasks (Figure 5). For more
    task details, see Appendix C.4. Notably for the referral letter generation task, the model generated
    letters are preferred or tied with experts across all the samples evaluated.
    16
    
    [94mimage description: [0m The line chart illustrates the accuracy of different methods on the NEJM CPC task across varying Top-k values (1 to 10). 
    
    Here's a breakdown:
    
    * **Y-axis:** Represents the NEJM CPC Accuracy, ranging from 10 to 70.
    * **X-axis:** Shows the "Top-k" values, ranging from 1 to 10.
    
    Five distinct methods are compared:
    
    * **Med-Gemini + Search (Blue triangles with line):** Shows the highest accuracy across all Top-k values, with a steady upward trend. It's surrounded by a light blue shaded area indicating a confidence interval.
    * **Med-Gemini (Blue circles with line):**  Performs slightly worse than "Med-Gemini + Search," also showing an upward trend.
    * **Prior SOTA (Red circles with line):**  Has a lower accuracy compared to the "Med-Gemini" methods, with a gradual upward trend. It also has a light red shaded area around the line indicating a confidence interval.
    * **Clinician + Search (Orange triangles with line):** Performs better than "Clinician" alone, showing a slow upward trend. It has a light orange shaded area around the line.
    * **Clinician (Orange circles with line):**  Demonstrates the lowest accuracy among all methods, with a very gradual upward trend. It also has a light orange shaded area around the line.
    
    Overall, the chart suggests that both "Med-Gemini" methods outperform the other approaches in terms of accuracy on the NEJM CPC task. The addition of "Search" seems to improve the accuracy for both "Med-Gemini" and "Clinician" methods. 
    
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>
</pre>



## Image Search

### Search similar image with image input [using multimodal image embeddings]

Imagine searching for images, but instead of typing words, you use an actual image as the clue.

Think of it like searching with a mini-map instead of a written address.
It's a different way to ask, "Show me more stuff like this".

So, instead of typing "various example of Gemini 1.5 long context", you show a picture of that image and say, "Find me more like this"

For demonstration purposes, we will only be finding similar images that show the various features of Gemini in a single document below. However, you can scale this design pattern to match (find relevant images) across multiple documents.


```
# You can find a similar image as per the images you have in the metadata.

image_query_path = "images/gemini_v1_5_report_technical.pdf_image_5_0_148.jpeg"

# Print a message indicating the input image
print("***Input image from user:***")

# Display the input image
Image.load_from_file(image_query_path)
```

    ***Input image from user:***
    




    
![png](output_66_1.png)
    



You expect to find images that are similar in terms of "long context prompts for Gemini 1.5 Pro"


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
    
    


    
![png](output_68_1.png)
    


You can also print the citation to see what it has matched.


```
# Display citation details for the top matching image
print_text_to_image_citation(
    matching_results_image, print_top=True
)  # Print citation details for the top matching image
```

    [91mCitation 1: Matched image path, page number and page text: 
    [0m
    [94mscore: [0m 0.84
    [94mfile_name: [0m gemini_v1_5_report_technical.pdf
    [94mpath: [0m images/gemini_v1_5_report_technical.pdf_image_4_1_143.jpeg
    [94mpage number: [0m 5
    [94mpage text: [0m Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context
    4.1. Qualitative Examples of Multimodal Long-Context Capabilities
    The ability to process multiple millions of tokens unlocks practical applications that were not possible
    before. In this section we demonstrate some surprising interactions we observed with Gemini 1.5 Pro
    across code, text and video.
    As shown in the Figure 2, Gemini 1.5 Pro is able to ingest entire large codebases such as JAX
    (746,152 tokens), and answer very specific queries about them. in Figure 3 we show Gemini 1.5 Pros
    ability to learn a new language based only on reference materials given in its input (see Section 4.2.1.7
    for quantitative metrics for this use case). Additionally, we test Gemini 1.5 Pros ability to answer
    an image query given the entire text of Les Misrables and observe that being natively multimodal
    allows it to locate a famous scene from a hand-drawn sketch, as shown in Figure 4. Lastly, we ask
    Gemini 1.5 Pro questions about an entire movie of 45 minutes in Figure 5 which the model answers
    seamlessly while retrieving moments and timestamps down to a second. 6
    Figure 2 | Given the entire 746,152 token JAX codebase in context, Gemini 1.5 Pro can identify the
    specific location of a core automatic differentiation method.
    Figure 3 | Given a reference grammar book and a bilingual wordlist (dictionary), Gemini 1.5 Pro is
    able to translate from English to Kalamang with similar quality to a human who learned from the
    same materials.
    6For additional short videos of demonstrations of the long context abilities of Gemini 1.5 Pro across video, text, and
    code see https://deepmind.google/technologies/gemini/.
    5
    
    [94mimage description: [0m The image depicts a system designed to translate English to Kalamang. The system uses a grammar book, a dictionary, and a long context as reference materials. 
    
    The user prompt is: "Given the reference materials as context, translate the following sentence from English to Kalamang: I'm getting pandanus, I want to make a mat."
    
    The model output is: "An padanual repte, irar paruotkin." 
    
    The image also indicates that the combined grammar book and dictionary contain 250k tokens. 
    
    


```
# Check Other Matched Images (Optional)
# You can access the other two matched images using:

print("---------------Matched Images------------------\n")
display_images(
    [
        matching_results_image[0]["img_path"],
        matching_results_image[1]["img_path"],
        matching_results_image[2]["img_path"],
    ],
    resize_ratio=0.2,
)
```

    ---------------Matched Images------------------
    
    


    
![png](output_71_1.png)
    


    
    
    


    
![png](output_71_3.png)
    


    
    
    


    
![png](output_71_5.png)
    


    
    
    

The ability to identify similar text and images based on user input, using Gemini and embeddings, forms a crucial foundation for development of Multimodal Question Answering System with multimodal RAG design pattern, which you will explore in the coming sections.

### Comparative reasoning

Next, let's apply what you have done so far in doing comparative reasoning.

For this example:

* **Step 1:** You will search all the images for a specific query

* **Step 2:** Send those images to Gemini 1.5 Pro to ask multiple questions, where it has to compare among those images and provide you with answers.


```
matching_results_image_query_1 = get_similar_image_from_query(
    text_metadata_df,
    image_metadata_df,
    query="Show me all the images that can describe LLMs and TPU v5e scaling",
    column_name="text_embedding_from_image_description",  # Use image description text embedding # mm_embedding_from_img_only text_embedding_from_image_description
    image_emb=False,  # Use text embedding instead of image embedding
    top_n=5,
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
        matching_results_image_query_1[2]["img_path"],
        matching_results_image_query_1[3]["img_path"],
        matching_results_image_query_1[4]["img_path"],
    ],
    resize_ratio=0.2,
)
```

    ---------------Matched Images------------------
    
    


    
![png](output_76_1.png)
    


    
    
    


    
![png](output_76_3.png)
    


    
    
    


    
![png](output_76_5.png)
    


    
    
    


    
![png](output_76_7.png)
    


    
    
    


    
![png](output_76_9.png)
    


    
    
    


```
prompt = f"""Task: Answer the following questions in detail, providing clear reasoning and evidence from the images in bullet points.
Instructions:
1. Analyze the provided images focusing on the relationship between TPU v5e scaling efficiency, LLM model size growth, performance metrics, and quantization effects.
2. Answer the following questions in detail, providing clear reasoning and evidence from the images in bullet points
3. Cite the image sources to support your explanations. Mention the file name.

Additional Considerations:
* Clearly define any technical terms (e.g., EMFU, TFLOP/chip/s) within your answers for better understanding.
* Use specific examples and data points from the images to support your explanations.
* Feel free to request additional information or clarification if the images are unclear or ambiguous.

Question:
 - How does the scaling efficiency of TPU v5e compare to the overall growth in LLM model size over time?
 - How does the model size impact the observed Per-chip performance and EMFU for a fixed number of TPU v5e chips (e.g., 256)?
 - For the INT8 Quant training with 32B parameters, how does its high EMFU relate to the observed TFLOP/chip/s?
 - how does the "per device batch (seq)" for a 16B model compare to a 128B model, and how does this affect the "Total observed Perf"?
 - how might the MFU be impacted by increasing LLM model size?
"""
```


```
%%time
# Generate response with Gemini 1.5 Pro
print("\n **** Result: ***** \n")
rich_Markdown(
    get_gemini_response(
        multimodal_model_15,
        model_input=[
            prompt,
            "Images:",
            matching_results_image_query_1[0]["image_object"],
            matching_results_image_query_1[1]["image_object"],
            matching_results_image_query_1[2]["image_object"],
            matching_results_image_query_1[3]["image_object"],
            matching_results_image_query_1[4]["image_object"],
        ],
        stream=True,
        safety_settings=safety_settings,
        generation_config=GenerationConfig(temperature=1, max_output_tokens=8192),
    )
)
```

    
     **** Result: ***** 
    
    CPU times: user 195 ms, sys: 39.2 ms, total: 235 ms
    Wall time: 21.6 s
    




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
                              <span style="font-weight: bold; text-decoration: underline">Analysis of TPU v5e Scaling Efficiency and LLM Training</span>                              

Let's break down the provided information to understand the relationship between TPU v5e, LLM model size, and      
performance.                                                                                                       

<span style="font-weight: bold">Terminology:</span>                                                                                                       

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">EMFU:</span> Exa Multiply-accumulate operations per second Full Utilization. This metric represents the theoretical    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>peak performance of the TPU system when fully utilized.                                                         
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">TFLOP/chip/s:</span> Tera Floating Point Operations per second per chip. This metric indicates the actual computational
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>throughput of each TPU chip.                                                                                    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">INT8 Quant:</span> INT8 Quantization. This technique reduces the precision of model weights and activations from higher
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>precision formats like BF16 (brain float 16) to INT8 (8-bit integer), reducing memory footprint and             
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>computational cost.                                                                                             

<span style="font-weight: bold">Answers:</span>                                                                                                           

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> 1 </span><span style="font-weight: bold">Scaling Efficiency vs. LLM Model Size Growth:</span>                                                                   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Observation:</span> "TPU v5e Efficient Scaling with 32B LLM" (image 1) shows the scaling efficiency of TPU v5e      
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>decreases as the number of chips increases.                                                                  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Reasoning:</span> While larger TPU systems provide more computational power, the communication overhead and data    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>movement between chips become bottlenecks, reducing overall efficiency. This trend is common in large-scale  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>distributed computing.                                                                                       
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">LLM Growth:</span> "LLM model size growth" (image 2) illustrates the exponential growth of LLM model sizes over     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>time.                                                                                                        
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Comparison:</span> The declining scaling efficiency of TPUs highlights the challenge of efficiently training        
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>increasingly larger LLMs. Even with hardware advancements, maximizing performance requires careful           
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>optimization and potentially novel architectural approaches.                                                 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> 2 </span><span style="font-weight: bold">Model Size Impact on Performance (256 Chips):</span>                                                                   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Per-chip Performance:</span> "MaxText LLM Training Results" (image 3) shows a decrease in "Per-chip" performance    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>with increasing model size for a fixed 256 TPU v5e chips. For instance, the Per-chip performance drops from  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>132 TFLOP/s (32B model) to 110 TFLOP/s (128B model).                                                         
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Reasoning:</span> Larger models require more data movement and communication, exceeding the memory bandwidth and    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>inter-chip communication capabilities of the system. This bottleneck leads to reduced per-chip utilization   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>and lower performance.                                                                                       
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">EMFU:</span> The EMFU generally decreases with increasing model size. For example, EMFU drops from 66.86% (32B      
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>model) to 56.06% (128B model) on 256 chips.                                                                  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Reasoning:</span> This trend reinforces the challenge of keeping all TPU cores fully utilized with larger models.   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>The increased communication overhead and memory bottlenecks prevent achieving maximum theoretical            
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>performance.                                                                                                 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> 3 </span><span style="font-weight: bold">INT8 Quant, EMFU, and TFLOP/chip/s (32B Model):</span>                                                                 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">High EMFU:</span> Image 3 shows that INT8 quantization with a 32B model on 199 TPU v5e pods (50944 chips) achieves a
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>high EMFU of 52.99%.                                                                                         
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">TFLOP/chip/s:</span> This configuration also achieves 104.4 TOP/s per chip.                                         
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Relationship:</span> Quantization reduces the computational complexity and memory footprint of the model. This      
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>reduction allows for better utilization of the TPU cores, leading to higher EMFU and TFLOP/chip/s compared to
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>BF16 training with the same number of chips.                                                                 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> 4 </span><span style="font-weight: bold">Per-Device Batch Size and Total Performance:</span>                                                                    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Batch Size Comparison:</span> "MaxText Model Configurations" (image 5) shows that the "per device batch (seq)"      
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>decreases with increasing model size. A 16B model uses a batch size of 6, while a 128B model uses only 1.    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Reasoning:</span> Larger models require more memory. To accommodate these models within the limited memory capacity 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>of each TPU, the batch size needs to be reduced.                                                             
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Impact on Total Performance:</span> Smaller batch sizes generally lead to slower training convergence and can       
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>potentially reduce the "Total observed Perf" despite the higher computational power of larger TPU systems.   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> 5 </span><span style="font-weight: bold">MFU and Increasing LLM Model Size:</span>                                                                              
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">MFU (Matrix Multiply Unit):</span>  MFUs are specialized hardware units within TPUs optimized for performing matrix 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>multiplications, which are core operations in LLMs.                                                          
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Impact:</span> As LLM model sizes grow, the computational demands on the MFUs increase significantly. This can lead 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>to MFU utilization becoming a bottleneck, especially if the memory bandwidth and inter-chip communication    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>cannot keep up with the required data supply rate.                                                           
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Potential Mitigation:</span> Addressing this bottleneck might involve architectural changes to the TPU, such as     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>increasing on-chip memory, improving memory bandwidth, or exploring new interconnect technologies for faster 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>data movement between chips.                                                                                 

<span style="font-weight: bold">Conclusion:</span>                                                                                                        

The insights derived from the provided images highlight the complex interplay between hardware capabilities, model 
characteristics, and software optimizations in LLM training. While TPU v5e demonstrates impressive performance,    
efficiently scaling training for increasingly larger LLM models presents ongoing challenges that necessitate       
continuous innovation in hardware and software solutions.                                                          
</pre>




## Building Multimodal QA System with retrieval augmented generation (mRAG)

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

query = """- How does the scaling efficiency of TPU v5e compare to the overall growth in LLM model size over time?
 - How does the model size impact the observed Per-chip performance and EMFU for a fixed number of TPU v5e chips (e.g., 256)?
 - For the INT8 Quant training with 32B parameters, how does its high EMFU relate to the observed TFLOP/chip/s?
 - how does the "per device batch (seq)" for a 16B model compare to a 128B model, and how does this affect the "Total observed Perf"?
 - how might the MFU be impacted by increasing LLM model size?
 """
```

### Step 2: Get all relevant text chunks


```
# Retrieve relevant chunks of text based on the query
matching_results_chunks_data = get_similar_text_from_query(
    query,
    text_metadata_df,
    column_name="text_embedding_chunk",
    top_n=20,
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
instruction = """Task: Answer the following questions in detail, providing clear reasoning and evidence from the images and text in bullet points.
Instructions:

1. **Analyze:** Carefully examine the provided images and text context.
2. **Synthesize:** Integrate information from both the visual and textual elements.
3. **Reason:**  Deduce logical connections and inferences to address the question.
4. **Respond:** Provide a concise, accurate answer in the following format:

   * **Question:** [Question]
   * **Answer:** [Direct response to the question]
   * **Explanation:** [Bullet-point reasoning steps if applicable]
   * **Source** [name of the file, page, image from where the information is citied]

5. **Ambiguity:** If the context is insufficient to answer, respond "Not enough context to answer."

"""

# combine all the selected relevant text chunks
context_text = ["Text Context: "]
for key, value in matching_results_chunks_data.items():
    context_text.extend(
        [
            "Text Source: ",
            f"""file_name: "{value["file_name"]}" Page: "{value["page_num"]}""",
            "Text",
            value["chunk_text"],
        ]
    )

# combine all the selected relevant images
gemini_content = [
    instruction,
    "Questions: ",
    query,
    "Image Context: ",
]
for key, value in matching_results_image_fromdescription_data.items():
    gemini_content.extend(
        [
            "Image Path: ",
            value["img_path"],
            "Image Description: ",
            value["image_description"],
            "Image:",
            value["image_object"],
        ]
    )
gemini_content.extend(context_text)
```

### Step 5: Pass context to Gemini


```
# Generate Gemini response with streaming output
rich_Markdown(
    get_gemini_response(
        multimodal_model_15,
        model_input=gemini_content,
        stream=True,
        safety_settings=safety_settings,
        generation_config=GenerationConfig(temperature=1, max_output_tokens=8192),
    )
)
```




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Question:</span> How does the scaling efficiency of TPU v5e compare to the overall growth in LLM model size over time? 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Answer:</span> Not enough context to answer.                                                                           
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Explanation:</span>                                                                                                    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>While the text mentions the scaling efficiency of TPU v5e and provides data on its performance with different
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>MaxText LLM sizes, it doesn't offer a direct comparison to the overall growth in LLM model size over time.   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>The "LLM model size growth" chart shows the trend of increasing model size, but we need more specific data on
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>TPU v5e's scaling efficiency at each point in time to establish a correlation.                               
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Question:</span> How does the model size impact the observed Per-chip performance and EMFU for a fixed number of TPU   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>v5e chips (e.g., 256)?                                                                                          
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Answer:</span> For a fixed number of TPU v5e chips (256), increasing the model size generally leads to a decrease in   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>"Observed Per-chip performance" but doesn't show a clear trend for EMFU.                                        
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Explanation:</span>                                                                                                    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>Observing the "MaxText LLM Training Results" table, specifically the rows with 256 TPU v5e chips, as model   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>size increases from 16B to 128B parameters, the "Observed Perchip Perf" decreases (120 TFLOP/s to 110        
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>TFLOP/s).                                                                                                    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>However, EMFU fluctuates without a clear trend. It starts at 61.10% for 16B, peaks at 66.86% for 32B, then   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>decreases to 56.06% for 128B.                                                                                
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Source:</span> images/Google Cloud TPU blog.pdf_image_13_0_69.jpeg                                                     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Question:</span> For the INT8 Quant training with 32B parameters, how does its high EMFU relate to the observed        
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>TFLOP/chip/s?                                                                                                   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Answer:</span> Not enough context to answer.                                                                           
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Explanation:</span>                                                                                                    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>The table provides the EMFU (52.99%) for the INT8 Quant training with 32B parameters.                        
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>However, it doesn't list the observed TOP/chip/s, which is necessary to calculate EMFU based on the formula: 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>EMFU = (observed TOP/chip/s) / (peak hardware TFLOP/chip/s).                                                 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>Without the observed TOP/chip/s, we can't establish a direct relationship between EMFU and the observed      
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>TFLOP/chip/s.                                                                                                
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Source:</span> images/Google Cloud TPU blog.pdf_image_13_0_69.jpeg and images/Google Cloud TPU                         
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>blog.pdf_image_12_0_66.jpeg                                                                                     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Question:</span> how does the "per device batch (seq)" for a 16B model compare to a 128B model, and how does this      
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>affect the "Total observed Perf"?                                                                               
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Answer:</span>  The "per device batch (seq)" for a 16B model (6) is six times larger than a 128B model (1). This       
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>difference in batch size likely contributes to the 16B model having a lower "Total observed Perf" compared to   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>the 128B model, despite having higher "Observed Perchip Perf".                                                  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Explanation:</span>                                                                                                    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>The "MaxText Model Configurations" table shows that the "per device batch (seq)" decreases as the model size 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>increases.                                                                                                   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>A smaller batch size for larger models is expected, as they require more memory per instance.                
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>While the 16B model has a higher "Observed Perchip Perf" (120 TFLOP/s vs. 110 TFLOP/s) with 256 chips, its   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>smaller total batch size (6 vs. 1) likely limits its overall throughput, resulting in a lower "Total observed
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>Perf" (0.03 exa-FLOPs vs. 0.03 exa-FLOPs).                                                                   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Source:</span> images/Google Cloud TPU blog.pdf_image_13_0_69.jpeg and images/Google Cloud TPU                         
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>blog.pdf_image_6_0_35.jpeg                                                                                      
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Question:</span> How might the MFU be impacted by increasing LLM model size?                                           
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Answer:</span> Based on the provided information, we can infer that increasing LLM model size might lead to a decrease 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>in MFU.                                                                                                         
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Explanation:</span>                                                                                                    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>Larger models often require more complex communication patterns and memory management, potentially leading to
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>increased overhead and reduced computational efficiency.                                                     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span>The "MaxText LLM Training Results" table shows that for a fixed number of chips (256), "Observed Perchip     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>Perf" generally decreases as the model size increases. Since MFU is calculated using "observed TFLOP/chip/s",
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>a decrease in "Observed Perchip Perf" generally indicates a potential decrease in MFU.                       
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Source:</span> images/Google Cloud TPU blog.pdf_image_13_0_69.jpeg and images/Google Cloud TPU                         
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>blog.pdf_image_11_1_63.jpeg                                                                                     
</pre>




### Step 6: Print citations and references [Optional]

**Optional:** Uncomment to see the detailed citations.


```
# print("---------------Matched Images------------------\n")
# display_images(
#     [
#         matching_results_image_fromdescription_data[0]["img_path"],
#         matching_results_image_fromdescription_data[1]["img_path"],
#     ],
#     resize_ratio=0.2,
# )
```


```
# # Image citations. You can check how Gemini generated metadata helped in grounding the answer.

# print_text_to_image_citation(
#     matching_results_image_fromdescription_data, print_top=True
# )
```


```
# # Text citations

# print_text_to_text_citation(
#     matching_results_chunks_data,
#     print_top=True,
#     chunk_text=True,
# )
```

### Multimodal RAG

### More questions with Multimodal QA System


```
# Some questions to try
# this time we are not passing any images, but just a simple text query.
query = """Question 1: Imagine a patient presents with new onset prurigo nodularis.
Could Med-Gemini-M 1.5 be used to analyze dermatological images of the patient's lesions in conjunction with a comprehensive history taken
from an EHR dialogue to help a clinician reach a diagnosis and develop a treatment plan?
What are the limitations and potential ethical considerations of using the model in this way?

Question 2: The paper focuses on uncertainty-guided search for text-based reasoning tasks.
How could this approach be extended to multimodal tasks?
For instance, if Med-Gemini-M 1.5 encounters uncertainty when analyzing a dermatology image, could it generate queries to
search for relevant visual examples or supplemental clinical information to refine its interpretation?

Question 3:  Considering the potential benefits and risks highlighted in the paper, what specific steps should be taken during the development,
validation, and deployment of Med-Gemini models to ensure they are used safely, fairly, and effectively in real-world clinical settings?
How can these steps be informed by ongoing collaboration between researchers, clinicians, regulators, and patient communities?
 """

(
    response,
    matching_results_chunks_data,
    matching_results_image_fromdescription_data,
) = get_answer_from_qa_system(
    query,
    text_metadata_df,
    image_metadata_df,
    top_n_text=10,
    top_n_image=5,
    model=multimodal_model_15,
    safety_settings=safety_settings,
    generation_config=GenerationConfig(temperature=1, max_output_tokens=8192),
)

rich_Markdown(response)
```




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
                                               <span style="font-weight: bold; text-decoration: underline">Med-Gemini Analysis:</span>                                                

<span style="font-weight: bold">Question 1:</span> Imagine a patient presents with new-onset prurigo nodularis. Could Med-Gemini-M 1.5 be used to analyze 
dermatological images of the patient's lesions in conjunction with a comprehensive history taken from an EHR       
dialogue to help a clinician reach a diagnosis and develop a treatment plan? What are the limitations and potential
ethical considerations of using the model in this way?                                                             

<span style="font-weight: bold">Answer:</span> While Med-Gemini-M 1.5 shows promise for multimodal diagnosis, it's not ready for real-world deployment in 
its current form.                                                                                                  

<span style="font-weight: bold">Explanation:</span>                                                                                                       

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Potential:</span>                                                                                                      
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Multimodal Integration:</span> Med-Gemini-M 1.5 demonstrates the ability to analyze both text (EHR dialogues) and   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>images (dermatological photos), potentially aiding in prurigo nodularis diagnosis.                           
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Diagnosis &amp; Treatment Suggestions:</span>  Figure 6a showcases a hypothetical dialogue where Med-Gemini-M 1.5       
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>accurately diagnoses prurigo nodularis from a patient description and image, and provides treatment options. 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span><span style="font-weight: bold">Source: med_gemini.pdf, Page 18, 19, Figure 6a</span>                                                               
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Limitations:</span>                                                                                                    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Limited Data:</span> Figure 6b highlights that the example relies on "limited data" of one photo and brief          
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>description. More comprehensive data is needed for robust diagnosis. <span style="font-weight: bold">Source: med_gemini.pdf, Page 19, Figure </span>
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span><span style="font-weight: bold">6b</span>                                                                                                           
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Lack of Real-world Validation:</span> The paper acknowledges the need for "considerable further research and        
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>development" before real-world use. <span style="font-weight: bold">Source: med_gemini.pdf, Page 19</span>                                          
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Ethical Considerations:</span>                                                                                         
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Bias:</span>  The model's training data could contain biases, leading to inaccurate or unfair diagnoses for certain 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>demographics.                                                                                                
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Over-reliance:</span> Clinicians might over-rely on the model's output, potentially missing crucial nuances in      
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>patient history or symptoms.                                                                                 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Patient Privacy:</span>  Integrating patient data raises privacy concerns, requiring robust data security and       
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>informed consent protocols.                                                                                  

<span style="font-weight: bold">Source:</span> med_gemini.pdf, Pages 18-19, Figure 6                                                                      

<span style="font-weight: bold">Question 2:</span> The paper focuses on uncertainty-guided search for text-based reasoning tasks. How could this approach 
be extended to multimodal tasks? For instance, if Med-Gemini-M 1.5 encounters uncertainty when analyzing a         
dermatology image, could it generate queries to search for relevant visual examples or supplemental clinical       
information to refine its interpretation?                                                                          

<span style="font-weight: bold">Answer:</span> Yes, uncertainty-guided search can be extended to multimodal tasks.  Med-Gemini-M 1.5 could generate       
queries for both visual and textual information to reduce uncertainty.                                             

<span style="font-weight: bold">Explanation:</span>                                                                                                       

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Current Approach:</span>  The paper describes using uncertainty-guided search for text-based reasoning. When the model 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>is uncertain, it generates search queries to gather more information. <span style="font-weight: bold">Source: med_gemini.pdf, Page 9</span>            
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Multimodal Extension:</span> This approach can be adapted for multimodal uncertainty:                                  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Visual Queries:</span>  If uncertain about a dermatology image, Med-Gemini-M 1.5 could generate queries like        
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>"prurigo nodularis different skin tones" or "dermatofibroma vs. prurigo nodularis images" to retrieve        
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>relevant visual examples.                                                                                    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    • </span><span style="font-weight: bold">Supplemental Information Queries:</span> It could also generate queries for text-based information, such as "prurigo
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>nodularis histopathology findings" or "differential diagnosis of prurigo nodularis."                         
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Refined Interpretation:</span> By accessing and integrating both visual and textual search results, the model could    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>improve its understanding and refine its interpretation of the image.                                           

<span style="font-weight: bold">Source:</span> med_gemini.pdf, Page 9                                                                                     

<span style="font-weight: bold">Question 3:</span> Considering the potential benefits and risks highlighted in the paper, what specific steps should be   
taken during the development, validation, and deployment of Med-Gemini models to ensure they are used safely,      
fairly, and effectively in real-world clinical settings? How can these steps be informed by ongoing collaboration  
between researchers, clinicians, regulators, and patient communities?                                              

<span style="font-weight: bold">Answer:</span> Ensuring safe, fair, and effective use of Med-Gemini models requires a multi-faceted approach throughout   
their lifecycle:                                                                                                   

<span style="font-weight: bold">Development:</span>                                                                                                       

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Diverse &amp; Representative Data:</span>  Use training data that reflects diverse patient demographics and clinical       
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>presentations to mitigate bias. <span style="font-weight: bold">Collaboration:</span> Engage with patient communities to identify potential data gaps  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>and biases.                                                                                                     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Transparency &amp; Explainability:</span> Develop methods to make the model's reasoning transparent and understandable to  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>clinicians. <span style="font-weight: bold">Collaboration:</span> Involve clinicians to determine the level of explainability needed for trust and     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>effective use.                                                                                                  

<span style="font-weight: bold">Validation:</span>                                                                                                        

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Rigorous Testing:</span> Conduct extensive testing on diverse datasets and in real-world clinical settings to assess   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>accuracy, safety, and fairness. <span style="font-weight: bold">Collaboration:</span> Work with regulators to design appropriate validation studies    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>that meet regulatory standards.                                                                                 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Bias Audits:</span>  Regularly audit the model for biases and implement mitigation strategies. <span style="font-weight: bold">Collaboration:</span>          
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>Collaborate with ethicists and social scientists to develop comprehensive bias detection methods.               

<span style="font-weight: bold">Deployment:</span>                                                                                                        

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Human Oversight:</span> Implement systems that ensure human clinicians make final decisions and maintain control over  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>patient care. <span style="font-weight: bold">Collaboration:</span> Work with clinicians to define clear guidelines for human oversight and            
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>intervention.                                                                                                   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Continuous Monitoring:</span> Monitor the model's performance in real-world settings and make adjustments as needed.   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span><span style="font-weight: bold">Collaboration:</span> Establish feedback mechanisms for clinicians and patients to report issues and contribute to     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>model improvement.                                                                                              

<span style="font-weight: bold">Ongoing Collaboration:</span>  Regular communication and collaboration between researchers, clinicians, regulators, and   
patient communities are crucial throughout the entire process to ensure responsible and beneficial use of          
Med-Gemini models.                                                                                                 

<span style="font-weight: bold">Source:</span>  med_gemini.pdf (general principles throughout the paper)                                                  
</pre>





```
# Some questions to try

query = """Question 1: How does the mixture-of-experts architecture in Gemini 1.5 Pro contribute to its ability to handle long
context while maintaining performance on core capabilities? Discuss the potential trade-offs involved.

Question 2: Gemini 1.5 Pro incorporates various safety mitigations, including supervised fine-tuning and reinforcement learning.
Discuss the effectiveness of these mitigations in addressing content safety and representational harms in both text-to-text and
image-to-text modalities. How can these evaluations be improved?

Question 3: Gemini 1.5 Pro demonstrates surprising in-context language learning capabilities for Kalamang,
a low-resource language. What are the implications of this finding for language preservation and revitalization?
What challenges need to be addressed for broader applicability of this approach?
"""
(
    response,
    matching_results_chunks_data,
    matching_results_image_fromdescription_data,
) = get_answer_from_qa_system(
    query,
    text_metadata_df,
    image_metadata_df,
    top_n_text=10,
    top_n_image=5,
    model=multimodal_model_15,
    safety_settings=safety_settings,
    generation_config=GenerationConfig(temperature=1, max_output_tokens=8192),
)

rich_Markdown(response)
```




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
                                                    <span style="font-weight: bold; text-decoration: underline">Question 1:</span>                                                    

<span style="font-weight: bold">Question:</span> How does the mixture-of-experts architecture in Gemini 1.5 Pro contribute to its ability to handle long  
context while maintaining performance on core capabilities? Discuss the potential trade-offs involved.             

<span style="font-weight: bold">Answer:</span> The mixture-of-experts (MoE) architecture enables Gemini 1.5 Pro to handle long contexts by selectively    
activating specific expert networks for different parts of the input. This sparsity allows for processing longer   
sequences without a proportional increase in computational cost, leading to improved long-context performance while
maintaining core capabilities.                                                                                     

<span style="font-weight: bold">Explanation:</span>                                                                                                       

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">MoE for Long Context:</span> Traditional transformer models struggle with long sequences due to quadratic complexity.  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>MoE addresses this by using multiple expert networks, each specializing in different aspects of the data. For   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>any given input, only a subset of these experts are activated, enabling efficient processing of longer          
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>sequences. ("gemini_v1_5_report_technical.pdf", Page 2)                                                         
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Maintaining Core Capabilities:</span> The text states that the improved long-context performance "does not come at the 
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>expense of multi-modal core capabilities." This suggests that the model's ability to excel in core tasks, likely
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>those requiring shorter context lengths, is maintained. ("gemini_v1_5_report_technical.pdf", Page 30)           

<span style="font-weight: bold">Potential Trade-offs:</span>                                                                                              

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Complexity:</span> MoE models are inherently more complex to train and require careful routing mechanisms to select    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>appropriate experts, potentially increasing training and inference costs.                                       
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Interpretability:</span>  The selective activation of experts can make it harder to interpret the model's              
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>decision-making process compared to standard transformer architectures.                                         


                                                    <span style="font-weight: bold; text-decoration: underline">Question 2:</span>                                                    

<span style="font-weight: bold">Question:</span> Gemini 1.5 Pro incorporates various safety mitigations, including supervised fine-tuning and             
reinforcement learning. Discuss the effectiveness of these mitigations in addressing content safety and            
representational harms in both text-to-text and image-to-text modalities. How can these evaluations be improved?   

<span style="font-weight: bold">Answer:</span> Not enough context to answer.                                                                              

<span style="font-weight: bold">Explanation:</span> While the text mentions the use of safety mitigations, it doesn't provide specific details about their
effectiveness in addressing content safety and representational harms.  The provided text focuses on the model's   
architecture and performance on various tasks but lacks information on safety evaluations and mitigation           
strategies.                                                                                                        


                                                    <span style="font-weight: bold; text-decoration: underline">Question 3:</span>                                                    

<span style="font-weight: bold">Question:</span> Gemini 1.5 Pro demonstrates surprising in-context language learning capabilities for Kalamang, a         
low-resource language. What are the implications of this finding for language preservation and revitalization? What
challenges need to be addressed for broader applicability of this approach?                                        

<span style="font-weight: bold">Answer:</span> Gemini 1.5 Pro's ability to learn Kalamang translation from limited resources has significant implications 
for language preservation and revitalization. It suggests large language models can leverage existing linguistic   
documentation to bridge the digital divide for under-resourced languages.                                          

<span style="font-weight: bold">Explanation:</span>                                                                                                       

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Language Preservation:</span> By learning from grammar books, dictionaries, and parallel texts, Gemini 1.5 Pro can     
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>potentially contribute to documenting and preserving endangered languages like Kalamang                         
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>("gemini_v1_5_report_technical.pdf", Page 13, Image:                                                            
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>"images/gemini_v1_5_report_technical.pdf_image_4_1_143.jpeg")                                                   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Revitalization:</span> The model's translation capabilities could facilitate communication and learning materials for  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>speakers of endangered languages, potentially aiding revitalization efforts.                                    

<span style="font-weight: bold">Challenges for Broader Applicability:</span>                                                                              

<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Resource Availability:</span>  The approach relies on the existence of structured linguistic resources like grammar    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>books and dictionaries, which might not be readily available for all low-resource languages.                    
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Scalability:</span>  Manually creating and curating these resources for every language is time-consuming and           
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>resource-intensive, making it challenging to scale to thousands of languages.                                   
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> • </span><span style="font-weight: bold">Quality and Bias:</span>  The quality of the model's output is dependent on the accuracy and comprehensiveness of the  
<span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>provided resources. Biased or incomplete documentation could propagate into the model's translations.           
</pre>




## Conclusions

Congratulations on making it through this multimodal RAG notebook!

While multimodal RAG can be quite powerful, note that it can face some limitations:

* **Data dependency:** Needs high-quality paired text and visuals.
* **Computationally demanding:** Processing multimodal data is resource-intensive.
* **Domain specific:** Models trained on general data may not shine in specialized fields like medicine.
* **Black box:** Understanding how these models work can be tricky, hindering trust and adoption.


Despite these challenges, multimodal RAG represents a significant step towards search and retrieval systems that can handle diverse, multimodal data.
