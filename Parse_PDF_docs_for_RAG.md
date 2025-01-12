# Parsing PDF documents for RAG applications

This notebook shows how to leverage GPT-4V to turn rich PDF documents such as slide decks or exports from web pages into usable content for your RAG application.

This technique can be used if you have a lot of unstructured data containing valuable information that you want to be able to retrieve as part of your RAG pipeline.

For example, you could build a Knowledge Assistant that could answer user queries about your company or product based on information contained in PDF documents. 

The example documents used in this notebook are located at [data/example_pdfs](data/example_pdfs). They are related to OpenAI's APIs and various techniques that can be used as part of LLM projects.

## Data preparation

In this section, we will process our input data to prepare it for retrieval.

We will do this in 2 ways:

1. Extracting text with pdfminer
2. Converting the PDF pages to images to analyze them with GPT-4V

You can skip the 1st method if you want to only use the content inferred from the image analysis.

### Setup

We need to install a few libraries to convert the PDF to images and extract the text (optional).

**Note: You need to install `poppler` on your machine for the `pdf2image` library to work. You can follow the instructions to install it [here](https://pypi.org/project/pdf2image/).**


```python
%pip install pdf2image
%pip install pdfminer
%pip install openai
%pip install scikit-learn
%pip install rich
%pip install tqdm
%pip install concurrent
```


```python
# Imports
from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from pdfminer.high_level import extract_text
import base64
from io import BytesIO
import os
import concurrent
from tqdm import tqdm
from openai import OpenAI
import re
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
from rich import print
from ast import literal_eval
```

### File processing


```python
def convert_doc_to_images(path):
    images = convert_from_path(path)
    return images

def extract_text_from_doc(path):
    text = extract_text(path)
    page_text = []
    return text
```

#### Testing with an example


```python
file_path = "data/example_pdfs/fine-tuning-deck.pdf"

images = convert_doc_to_images(file_path)
```


```python
text = extract_text_from_doc(file_path)
```


```python
for img in images:
    display(img)
```


    
![png](output_10_0.png)
    



    
![png](output_10_1.png)
    



    
![png](output_10_2.png)
    



    
![png](output_10_3.png)
    



    
![png](output_10_4.png)
    



    
![png](output_10_5.png)
    



    
![png](output_10_6.png)
    


### Image analysis with GPT-4V

After converting a PDF file to multiple images, we'll use GPT-4V to analyze the content based on the images.


```python
# Initializing OpenAI client - see https://platform.openai.com/docs/quickstart?context=python
client = OpenAI()
```


```python
# Converting images to base64 encoded images in a data URI format to use with the ChatCompletions API
def get_img_uri(img):
    buffer = BytesIO()
    img.save(buffer, format="jpeg")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{base64_image}"
    return data_uri
```


```python
system_prompt = '''
You will be provided with an image of a pdf page or a slide. Your goal is to talk about the content that you see, in technical terms, as if you were delivering a presentation.

If there are diagrams, describe the diagrams and explain their meaning.
For example: if there is a diagram describing a process flow, say something like "the process flow starts with X then we have Y and Z..."

If there are tables, describe logically the content in the tables
For example: if there is a table listing items and prices, say something like "the prices are the following: A for X, B for Y..."

DO NOT include terms referring to the content format
DO NOT mention the content type - DO focus on the content itself
For example: if there is a diagram/chart and text on the image, talk about both without mentioning that one is a chart and the other is text.
Simply describe what you see in the diagram and what you understand from the text.

You should keep it concise, but keep in mind your audience cannot see the image so be exhaustive in describing the content.

Exclude elements that are not relevant to the content:
DO NOT mention page numbers or the position of the elements on the image.

------

If there is an identifiable title, identify the title to give the output in the following format:

{TITLE}

{Content description}

If there is no clear title, simply return the content description.

'''

def analyze_image(img_url):
    response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    temperature=0,
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": img_url,
                },
            ],
        }
    ],
        max_tokens=300,
        top_p=0.1
    )

    return response.choices[0].message.content
```

#### Testing with an example


```python
img = images[2]
data_uri = get_img_uri(img)
```


```python
res = analyze_image(data_uri)
print(res)
```

    What is Fine-tuning
    
    Fine-tuning a model consists of training the model to follow a set of given input/output examples. This will teach the model to behave in a certain way when confronted with a similar input in the future.
    
    We recommend using 50-100 examples even if the minimum is 10.
    
    The process involves starting with a public model, using training data to train the model, and resulting in a fine-tuned model.
    

#### Processing all documents


```python
files_path = "data/example_pdfs"

all_items = os.listdir(files_path)
files = [item for item in all_items if os.path.isfile(os.path.join(files_path, item))]
```


```python
def analyze_doc_image(img):
    img_uri = get_img_uri(img)
    data = analyze_image(img_uri)
    return data
```

We will list all files in the example folder and process them by 
1. Extracting the text
2. Converting the docs to images
3. Analyzing pages with GPT-4V

Note: This takes about ~2 mins to run. Feel free to skip and load directly the result file (see below).


```python
docs = []

for f in files:
    
    path = f"{files_path}/{f}"
    doc = {
        "filename": f
    }
    text = extract_text_from_doc(path)
    doc['text'] = text
    imgs = convert_doc_to_images(path)
    pages_description = []
    
    print(f"Analyzing pages for doc {f}")
    
    # Concurrent execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        
        # Removing 1st slide as it's usually just an intro
        futures = [
            executor.submit(analyze_doc_image, img)
            for img in imgs[1:]
        ]
        
        with tqdm(total=len(imgs)-1) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
        
        for f in futures:
            res = f.result()
            pages_description.append(res)
        
    doc['pages_description'] = pages_description
    docs.append(doc)
```

    Analyzing pages for doc rag-deck.pdf
    

    100%|██████████████████████████████████████████████████████████████████| 19/19 [00:32<00:00,  1.72s/it]
    

    Analyzing pages for doc models-page.pdf
    

    100%|████████████████████████████████████████████████████████████████████| 9/9 [00:25<00:00,  2.80s/it]
    

    Analyzing pages for doc evals-decks.pdf
    

    100%|██████████████████████████████████████████████████████████████████| 12/12 [00:29<00:00,  2.44s/it]
    

    Analyzing pages for doc fine-tuning-deck.pdf
    

    100%|████████████████████████████████████████████████████████████████████| 6/6 [00:19<00:00,  3.32s/it]
    


```python
# Saving result to file for later
json_path = "data/parsed_pdf_docs.json"

with open(json_path, 'w') as f:
    json.dump(docs, f)
```


```python
# Optional: load content from the saved file
with open(json_path, 'r') as f:
    docs = json.load(f)
```

### Embedding content
Before embedding the content, we will chunk it logically by page.
For real-world scenarios, you could explore more advanced ways to chunk the content:
- Cutting it into smaller pieces
- Adding data - such as the slide title, deck title and/or the doc description - at the beginning of each piece of content. That way, each independent chunk can be in context

For the sake of brevity, we will use a very simple chunking strategy and rely on separators to split the text by page.


```python
# Chunking content by page and merging together slides text & description if applicable
content = []
for doc in docs:
    # Removing first slide as well
    text = doc['text'].split('\f')[1:]
    description = doc['pages_description']
    description_indexes = []
    for i in range(len(text)):
        slide_content = text[i] + '\n'
        # Trying to find matching slide description
        slide_title = text[i].split('\n')[0]
        for j in range(len(description)):
            description_title = description[j].split('\n')[0]
            if slide_title.lower() == description_title.lower():
                slide_content += description[j].replace(description_title, '')
                # Keeping track of the descriptions added
                description_indexes.append(j)
        # Adding the slide content + matching slide description to the content pieces
        content.append(slide_content) 
    # Adding the slides descriptions that weren't used
    for j in range(len(description)):
        if j not in description_indexes:
            content.append(description[j])
```


```python
for c in content:
    print(c)
    print("\n\n-------------------------------\n\n")
```


```python
# Cleaning up content
# Removing trailing spaces, additional line breaks, page numbers and references to the content being a slide
clean_content = []
for c in content:
    text = c.replace(' \n', '').replace('\n\n', '\n').replace('\n\n\n', '\n').strip()
    text = re.sub(r"(?<=\n)\d{1,2}", "", text)
    text = re.sub(r"\b(?:the|this)\s*slide\s*\w+\b", "", text, flags=re.IGNORECASE)
    clean_content.append(text)
```


```python
for c in clean_content:
    print(c)
    print("\n\n-------------------------------\n\n")
```


```python
# Creating the embeddings
# We'll save to a csv file here for testing purposes but this is where you should load content in your vectorDB.
df = pd.DataFrame(clean_content, columns=['content'])
print(df.shape)
df.head()
```

    (64, 1)
    




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
      <th>content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Overview\nRetrieval-Augmented Generationenhanc...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What is RAG\nRetrieve information to Augment t...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>When to use RAG\nGood for  ✅\nNot good for  ❌\...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Technical patterns\nData preparation\nInput pr...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Technical patterns\nData preparation\nchunk do...</td>
    </tr>
  </tbody>
</table>
</div>




```python
embeddings_model = "text-embedding-3-large"

def get_embeddings(text):
    embeddings = client.embeddings.create(
      model="text-embedding-3-small",
      input=text,
      encoding_format="float"
    )
    return embeddings.data[0].embedding
```


```python
df['embeddings'] = df['content'].apply(lambda x: get_embeddings(x))
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
      <th>content</th>
      <th>embeddings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Overview\nRetrieval-Augmented Generationenhanc...</td>
      <td>[-0.014744381, 0.03017278, 0.06353764, 0.02110...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What is RAG\nRetrieve information to Augment t...</td>
      <td>[-0.024337867, 0.022921458, -0.00971687, 0.010...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>When to use RAG\nGood for  ✅\nNot good for  ❌\...</td>
      <td>[-0.011084231, 0.021158217, -0.00430421, 0.017...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Technical patterns\nData preparation\nInput pr...</td>
      <td>[-0.0058343858, 0.0408407, 0.054318383, 0.0190...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Technical patterns\nData preparation\nchunk do...</td>
      <td>[-0.010359385, 0.03736894, 0.052995477, 0.0180...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Saving locally for later
data_path = "data/parsed_pdf_docs_with_embeddings.csv"
df.to_csv(data_path, index=False)
```


```python
# Optional: load data from saved file
df = pd.read_csv(data_path)
df["embeddings"] = df.embeddings.apply(literal_eval).apply(np.array)
```

## Retrieval-augmented generation

The last step of the process is to generate outputs in response to input queries, after retrieving content as context to reply.


```python
system_prompt = '''
    You will be provided with an input prompt and content as context that can be used to reply to the prompt.
    
    You will do 2 things:
    
    1. First, you will internally assess whether the content provided is relevant to reply to the input prompt. 
    
    2a. If that is the case, answer directly using this content. If the content is relevant, use elements found in the content to craft a reply to the input prompt.

    2b. If the content is not relevant, use your own knowledge to reply or say that you don't know how to respond if your knowledge is not sufficient to answer.
    
    Stay concise with your answer, replying specifically to the input prompt without mentioning additional information provided in the context content.
'''

model="gpt-4-turbo-preview"

def search_content(df, input_text, top_k):
    embedded_value = get_embeddings(input_text)
    df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(np.array(x).reshape(1,-1), np.array(embedded_value).reshape(1, -1)))
    res = df.sort_values('similarity', ascending=False).head(top_k)
    return res

def get_similarity(row):
    similarity_score = row['similarity']
    if isinstance(similarity_score, np.ndarray):
        similarity_score = similarity_score[0][0]
    return similarity_score

def generate_output(input_prompt, similar_content, threshold = 0.5):
    
    content = similar_content.iloc[0]['content']
    
    # Adding more matching content if the similarity is above threshold
    if len(similar_content) > 1:
        for i, row in similar_content.iterrows():
            similarity_score = get_similarity(row)
            if similarity_score > threshold:
                content += f"\n\n{row['content']}"
            
    prompt = f"INPUT PROMPT:\n{input_prompt}\n-------\nCONTENT:\n{content}"
    
    completion = client.chat.completions.create(
        model=model,
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return completion.choices[0].message.content
```


```python
# Example user queries related to the content
example_inputs = [
    'What are the main models you offer?',
    'Do you have a speech recognition model?',
    'Which embedding model should I use for non-English use cases?',
    'Can I introduce new knowledge in my LLM app using RAG?',
    'How many examples do I need to fine-tune a model?',
    'Which metric can I use to evaluate a summarization task?',
    'Give me a detailed example for an evaluation process where we are looking for a clear answer to compare to a ground truth.',
]
```


```python
# Running the RAG pipeline on each example
for ex in example_inputs:
    print(f"[deep_pink4][bold]QUERY:[/bold] {ex}[/deep_pink4]\n\n")
    matching_content = search_content(df, ex, 3)
    print(f"[grey37][b]Matching content:[/b][/grey37]\n")
    for i, match in matching_content.iterrows():
        print(f"[grey37][i]Similarity: {get_similarity(match):.2f}[/i][/grey37]")
        print(f"[grey37]{match['content'][:100]}{'...' if len(match['content']) > 100 else ''}[/[grey37]]\n\n")
    reply = generate_output(ex, matching_content)
    print(f"[turquoise4][b]REPLY:[/b][/turquoise4]\n\n[spring_green4]{reply}[/spring_green4]\n\n--------------\n\n")
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #af005f; text-decoration-color: #af005f; font-weight: bold">QUERY:</span><span style="color: #af005f; text-decoration-color: #af005f"> What are the main models you offer?</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">Matching content:</span>

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.43</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Models - OpenAI API</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">The content lists various API endpoints and their corresponding latest models:</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">-...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.39</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">26</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">02</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">2024</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">, </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">17:58</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Models - OpenAI API</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">The Moderation models are designed to check whether content co...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.39</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">The content describes various models provided by OpenAI, focusing on moderation models and GPT base ...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008787; text-decoration-color: #008787; font-weight: bold">REPLY:</span>

<span style="color: #00875f; text-decoration-color: #00875f">The main models we offer include:</span>
<span style="color: #00875f; text-decoration-color: #00875f">- For completions: gpt-</span><span style="color: #00875f; text-decoration-color: #00875f; font-weight: bold">3.5</span><span style="color: #00875f; text-decoration-color: #00875f">-turbo-instruct, babbage-</span><span style="color: #00875f; text-decoration-color: #00875f; font-weight: bold">002</span><span style="color: #00875f; text-decoration-color: #00875f">, and davinci-</span><span style="color: #00875f; text-decoration-color: #00875f; font-weight: bold">002</span><span style="color: #00875f; text-decoration-color: #00875f">.</span>
<span style="color: #00875f; text-decoration-color: #00875f">- For embeddings: text-embedding-</span><span style="color: #00875f; text-decoration-color: #00875f; font-weight: bold">3</span><span style="color: #00875f; text-decoration-color: #00875f">-small, text-embedding-</span><span style="color: #00875f; text-decoration-color: #00875f; font-weight: bold">3</span><span style="color: #00875f; text-decoration-color: #00875f">-large, and text-embedding-ada-</span><span style="color: #00875f; text-decoration-color: #00875f; font-weight: bold">002</span><span style="color: #00875f; text-decoration-color: #00875f">.</span>
<span style="color: #00875f; text-decoration-color: #00875f">- For fine-tuning jobs: gpt-</span><span style="color: #00875f; text-decoration-color: #00875f; font-weight: bold">3.5</span><span style="color: #00875f; text-decoration-color: #00875f">-turbo, babbage-</span><span style="color: #00875f; text-decoration-color: #00875f; font-weight: bold">002</span><span style="color: #00875f; text-decoration-color: #00875f">, and davinci-</span><span style="color: #00875f; text-decoration-color: #00875f; font-weight: bold">002</span><span style="color: #00875f; text-decoration-color: #00875f">.</span>
<span style="color: #00875f; text-decoration-color: #00875f">- For moderations: text-moderation-stable and text-moderation.</span>
<span style="color: #00875f; text-decoration-color: #00875f">Additionally, we have the latest models like gpt-</span><span style="color: #00875f; text-decoration-color: #00875f; font-weight: bold">3.5</span><span style="color: #00875f; text-decoration-color: #00875f">-turbo-16k and fine-tuned versions of gpt-</span><span style="color: #00875f; text-decoration-color: #00875f; font-weight: bold">3.5</span><span style="color: #00875f; text-decoration-color: #00875f">-turbo.</span>

--------------


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #af005f; text-decoration-color: #af005f; font-weight: bold">QUERY:</span><span style="color: #af005f; text-decoration-color: #af005f"> Do you have a speech recognition model?</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">Matching content:</span>

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.53</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">The content describes various models related to text-to-speech, speech recognition, embeddings, and ...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.50</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">26</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">02</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">2024</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">, </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">17:58</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Models - OpenAI API</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">MODEL</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">DE S CRIPTION</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">tts-</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">1</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">New  Text-to-speech </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">1</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">The latest tex...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.44</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">26</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">02</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">2024</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">, </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">17:58</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Models - OpenAI API</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">ENDP OINT</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">DATA USED</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">FOR TRAINING</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">DEFAULT</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">RETENTION</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">ELIGIBLE FO...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008787; text-decoration-color: #008787; font-weight: bold">REPLY:</span>

<span style="color: #00875f; text-decoration-color: #00875f">Yes, the Whisper model is a general-purpose speech recognition model mentioned in the content, capable of </span>
<span style="color: #00875f; text-decoration-color: #00875f">multilingual speech recognition, speech translation, and language identification. The v2-large model, referred to </span>
<span style="color: #00875f; text-decoration-color: #00875f">as </span><span style="color: #00875f; text-decoration-color: #00875f">"whisper-1"</span><span style="color: #00875f; text-decoration-color: #00875f">, is available through an API and is optimized for faster performance.</span>

--------------


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #af005f; text-decoration-color: #af005f; font-weight: bold">QUERY:</span><span style="color: #af005f; text-decoration-color: #af005f"> Which embedding model should I use for non-English use cases?</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">Matching content:</span>

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.57</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">The content describes various models related to text-to-speech, speech recognition, embeddings, and ...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.46</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">26</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">02</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">2024</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">, </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">17:58</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Models - OpenAI API</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">MODEL</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">DE S CRIPTION</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">tts-</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">1</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">New  Text-to-speech </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">1</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">The latest tex...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.40</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">26</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">02</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">2024</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">, </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">17:58</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Models - OpenAI API</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Multilingual capabilities</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">GPT-</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">4</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f"> outperforms both previous larg...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008787; text-decoration-color: #008787; font-weight: bold">REPLY:</span>

<span style="color: #00875f; text-decoration-color: #00875f">For non-English use cases, you should use the </span><span style="color: #00875f; text-decoration-color: #00875f">"V3 large"</span><span style="color: #00875f; text-decoration-color: #00875f"> embedding model, as it is described as the most capable </span>
<span style="color: #00875f; text-decoration-color: #00875f">for both English and non-English tasks, with an output dimension of </span><span style="color: #00875f; text-decoration-color: #00875f; font-weight: bold">3</span><span style="color: #00875f; text-decoration-color: #00875f">,</span><span style="color: #00875f; text-decoration-color: #00875f; font-weight: bold">072</span><span style="color: #00875f; text-decoration-color: #00875f">.</span>

--------------


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #af005f; text-decoration-color: #af005f; font-weight: bold">QUERY:</span><span style="color: #af005f; text-decoration-color: #af005f"> Can I introduce new knowledge in my LLM app using RAG?</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">Matching content:</span>

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.50</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">What is RAG</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Retrieve information to Augment the model’s knowledge and Generate the output</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">“What is y...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.49</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">When to use RAG</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Good for  ✅</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Not good for  ❌</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">●</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">●</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Introducing new information to the model</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">●</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Teaching ...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.43</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Technical patterns</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Data preparation: augmenting content</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">What does “Augmentingcontent” mean?</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Augmenti...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008787; text-decoration-color: #008787; font-weight: bold">REPLY:</span>

<span style="color: #00875f; text-decoration-color: #00875f">Yes, you can introduce new knowledge in your LLM app using RAG by retrieving information from a knowledge base or </span>
<span style="color: #00875f; text-decoration-color: #00875f">external sources to augment the model's knowledge and generate outputs relevant to the queries posed.</span>

--------------


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #af005f; text-decoration-color: #af005f; font-weight: bold">QUERY:</span><span style="color: #af005f; text-decoration-color: #af005f"> How many examples do I need to fine-tune a model?</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">Matching content:</span>

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.68</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">What is Fine-tuning</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Public Model</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Training data</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Training</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Fine-tunedmodel</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Fine-tuning a model consists...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.62</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">When to fine-tune</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Fine-tuning is good for:</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">- Following a given format or tone for the output</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">- Proce...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.57</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Overview</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Fine-tuning involves adjusting theparameters of pre-trained models on aspeciﬁc dataset or t...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008787; text-decoration-color: #008787; font-weight: bold">REPLY:</span>

<span style="color: #00875f; text-decoration-color: #00875f">We recommend using </span><span style="color: #00875f; text-decoration-color: #00875f; font-weight: bold">50</span><span style="color: #00875f; text-decoration-color: #00875f">-</span><span style="color: #00875f; text-decoration-color: #00875f; font-weight: bold">100</span><span style="color: #00875f; text-decoration-color: #00875f"> examples for fine-tuning a model, even though the minimum is </span><span style="color: #00875f; text-decoration-color: #00875f; font-weight: bold">10</span><span style="color: #00875f; text-decoration-color: #00875f">.</span>

--------------


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #af005f; text-decoration-color: #af005f; font-weight: bold">QUERY:</span><span style="color: #af005f; text-decoration-color: #af005f"> Which metric can I use to evaluate a summarization task?</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">Matching content:</span>

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.53</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Technical patterns</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Metric-based evaluations</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">ROUGE is a common metric for evaluating machine summariz...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.49</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Technical patterns</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Metric-based evaluations</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Component evaluations</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Subjective evaluations</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">●</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">●</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Compari...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.48</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Technical patterns</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Metric-based evaluations</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">BLEU score is another standard metric, this time focusin...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008787; text-decoration-color: #008787; font-weight: bold">REPLY:</span>

<span style="color: #00875f; text-decoration-color: #00875f">ROUGE is a common metric you can use to evaluate a summarization task.</span>

--------------


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #af005f; text-decoration-color: #af005f; font-weight: bold">QUERY:</span><span style="color: #af005f; text-decoration-color: #af005f"> Give me a detailed example for an evaluation process where we are looking for a clear answer to compare to a</span>
<span style="color: #af005f; text-decoration-color: #af005f">ground truth.</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">Matching content:</span>

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.60</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">What are evals</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Example</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Our ground truth matches the predicted answer, so the evaluation passes!</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Eval...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.59</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">What are evals</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Example</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">An evaluation contains a question and a correct answer. We call this the grou...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-style: italic">Similarity: </span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold; font-style: italic">0.50</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Technical patterns</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">Metric-based evaluations</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">What they’re good for</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">What to be aware of</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">●</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">●</span>
<span style="color: #5f5f5f; text-decoration-color: #5f5f5f">A good sta...</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">[</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f">/</span><span style="color: #5f5f5f; text-decoration-color: #5f5f5f; font-weight: bold">]</span>


</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008787; text-decoration-color: #008787; font-weight: bold">REPLY:</span>

<span style="color: #00875f; text-decoration-color: #00875f">The content provided is relevant and offers a detailed example for an evaluation process comparing to a ground </span>
<span style="color: #00875f; text-decoration-color: #00875f">truth. Here's a concise explanation based on the content:</span>

<span style="color: #00875f; text-decoration-color: #00875f">In the given example, the evaluation process involves a question-and-answer scenario to verify the accuracy of </span>
<span style="color: #00875f; text-decoration-color: #00875f">information retrieved by a tool or system in response to a query. The question posed is, </span><span style="color: #00875f; text-decoration-color: #00875f">"What is the population of</span>
<span style="color: #00875f; text-decoration-color: #00875f">Canada?"</span><span style="color: #00875f; text-decoration-color: #00875f"> The ground truth, or the correct answer, is established as </span><span style="color: #00875f; text-decoration-color: #00875f">"The population of Canada in 2023 is 39,566,248</span>
<span style="color: #00875f; text-decoration-color: #00875f">people."</span><span style="color: #00875f; text-decoration-color: #00875f"> A tool labeled </span><span style="color: #00875f; text-decoration-color: #00875f">"LLM"</span><span style="color: #00875f; text-decoration-color: #00875f"> is then used to search for the answer, which predicts </span><span style="color: #00875f; text-decoration-color: #00875f">"The current population of </span>
<span style="color: #00875f; text-decoration-color: #00875f">Canada is 39,566,248 as of Tuesday, May 23, 2023."</span><span style="color: #00875f; text-decoration-color: #00875f"> This predicted answer matches the ground truth exactly, </span>
<span style="color: #00875f; text-decoration-color: #00875f">indicating that the evaluation passes. This process demonstrates how an evaluation can be used to verify the </span>
<span style="color: #00875f; text-decoration-color: #00875f">accuracy of information retrieved by a tool, comparing the predicted answer to the ground truth to ensure </span>
<span style="color: #00875f; text-decoration-color: #00875f">correctness.</span>

--------------


</pre>



## Wrapping up

In this notebook, we have learned how to develop a basic RAG pipeline based on PDF documents. This includes:

- How to parse pdf documents, taking slide decks and an export from an HTML page as examples, using a python library as well as GPT-4V to interpret the visuals
- How to process the extracted content, clean it and chunk it into several pieces
- How to embed the processed content using OpenAI embeddings
- How to retrieve content that is relevant to an input query
- How to use GPT-4-turbo to generate an answer using the retrieved content as context

If you want to explore further, consider these optimisations:

- Playing around with the prompts provided as examples
- Chunking the content further and adding metadata as context to each chunk
- Adding rule-based filtering on the retrieval results or re-ranking results to surface to most relevant content

You can apply the techniques covered in this notebook to multiple use cases, such as assistants that can access your proprietary data, customer service or FAQ bots that can read from your internal policies, or anything that requires leveraging rich documents that would be better understood as images.
