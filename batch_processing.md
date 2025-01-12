# Batch processing with the Batch API

The new Batch API allows to **create async batch jobs for a lower price and with higher rate limits**.

Batches will be completed within 24h, but may be processed sooner depending on global usage. 

Ideal use cases for the Batch API include:

- Tagging, captioning, or enriching content on a marketplace or blog
- Categorizing and suggesting answers for support tickets
- Performing sentiment analysis on large datasets of customer feedback
- Generating summaries or translations for collections of documents or articles

and much more!

This cookbook will walk you through how to use the Batch API with a couple of practical examples.

We will start with an example to categorize movies using `gpt-4o-mini`, and then cover how we can use the vision capabilities of this model to caption images.

Please note that multiple models are available through the Batch API, and that you can use the same parameters in your Batch API calls as with the Chat Completions endpoint.

## Setup


```python
# Make sure you have the latest version of the SDK available to use the Batch API
%pip install openai --upgrade
```


```python
import json
from openai import OpenAI
import pandas as pd
from IPython.display import Image, display
```


```python
# Initializing OpenAI client - see https://platform.openai.com/docs/quickstart?context=python
client = OpenAI()
```

## First example: Categorizing movies

In this example, we will use `gpt-4o-mini` to extract movie categories from a description of the movie. We will also extract a 1-sentence summary from this description. 

We will use [JSON mode](https://platform.openai.com/docs/guides/text-generation/json-mode) to extract categories as an array of strings and the 1-sentence summary in a structured format. 

For each movie, we want to get a result that looks like this:

```
{
    categories: ['category1', 'category2', 'category3'],
    summary: '1-sentence summary'
}
```

### Loading data

We will use the IMDB top 1000 movies dataset for this example. 


```python
dataset_path = "data/imdb_top_1000.csv"

df = pd.read_csv(dataset_path)
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
      <th>Poster_Link</th>
      <th>Series_Title</th>
      <th>Released_Year</th>
      <th>Certificate</th>
      <th>Runtime</th>
      <th>Genre</th>
      <th>IMDB_Rating</th>
      <th>Overview</th>
      <th>Meta_score</th>
      <th>Director</th>
      <th>Star1</th>
      <th>Star2</th>
      <th>Star3</th>
      <th>Star4</th>
      <th>No_of_Votes</th>
      <th>Gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://m.media-amazon.com/images/M/MV5BMDFkYT...</td>
      <td>The Shawshank Redemption</td>
      <td>1994</td>
      <td>A</td>
      <td>142 min</td>
      <td>Drama</td>
      <td>9.3</td>
      <td>Two imprisoned men bond over a number of years...</td>
      <td>80.0</td>
      <td>Frank Darabont</td>
      <td>Tim Robbins</td>
      <td>Morgan Freeman</td>
      <td>Bob Gunton</td>
      <td>William Sadler</td>
      <td>2343110</td>
      <td>28,341,469</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://m.media-amazon.com/images/M/MV5BM2MyNj...</td>
      <td>The Godfather</td>
      <td>1972</td>
      <td>A</td>
      <td>175 min</td>
      <td>Crime, Drama</td>
      <td>9.2</td>
      <td>An organized crime dynasty's aging patriarch t...</td>
      <td>100.0</td>
      <td>Francis Ford Coppola</td>
      <td>Marlon Brando</td>
      <td>Al Pacino</td>
      <td>James Caan</td>
      <td>Diane Keaton</td>
      <td>1620367</td>
      <td>134,966,411</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://m.media-amazon.com/images/M/MV5BMTMxNT...</td>
      <td>The Dark Knight</td>
      <td>2008</td>
      <td>UA</td>
      <td>152 min</td>
      <td>Action, Crime, Drama</td>
      <td>9.0</td>
      <td>When the menace known as the Joker wreaks havo...</td>
      <td>84.0</td>
      <td>Christopher Nolan</td>
      <td>Christian Bale</td>
      <td>Heath Ledger</td>
      <td>Aaron Eckhart</td>
      <td>Michael Caine</td>
      <td>2303232</td>
      <td>534,858,444</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://m.media-amazon.com/images/M/MV5BMWMwMG...</td>
      <td>The Godfather: Part II</td>
      <td>1974</td>
      <td>A</td>
      <td>202 min</td>
      <td>Crime, Drama</td>
      <td>9.0</td>
      <td>The early life and career of Vito Corleone in ...</td>
      <td>90.0</td>
      <td>Francis Ford Coppola</td>
      <td>Al Pacino</td>
      <td>Robert De Niro</td>
      <td>Robert Duvall</td>
      <td>Diane Keaton</td>
      <td>1129952</td>
      <td>57,300,000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://m.media-amazon.com/images/M/MV5BMWU4N2...</td>
      <td>12 Angry Men</td>
      <td>1957</td>
      <td>U</td>
      <td>96 min</td>
      <td>Crime, Drama</td>
      <td>9.0</td>
      <td>A jury holdout attempts to prevent a miscarria...</td>
      <td>96.0</td>
      <td>Sidney Lumet</td>
      <td>Henry Fonda</td>
      <td>Lee J. Cobb</td>
      <td>Martin Balsam</td>
      <td>John Fiedler</td>
      <td>689845</td>
      <td>4,360,000</td>
    </tr>
  </tbody>
</table>
</div>



### Processing step 

Here, we will prepare our requests by first trying them out with the Chat Completions endpoint.

Once we're happy with the results, we can move on to creating the batch file.


```python
categorize_system_prompt = '''
Your goal is to extract movie categories from movie descriptions, as well as a 1-sentence summary for these movies.
You will be provided with a movie description, and you will output a json object containing the following information:

{
    categories: string[] // Array of categories based on the movie description,
    summary: string // 1-sentence summary of the movie based on the movie description
}

Categories refer to the genre or type of the movie, like "action", "romance", "comedy", etc. Keep category names simple and use only lower case letters.
Movies can have several categories, but try to keep it under 3-4. Only mention the categories that are the most obvious based on the description.
'''

def get_categories(description):
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.1,
    # This is to enable JSON mode, making sure responses are valid json objects
    response_format={ 
        "type": "json_object"
    },
    messages=[
        {
            "role": "system",
            "content": categorize_system_prompt
        },
        {
            "role": "user",
            "content": description
        }
    ],
    )

    return response.choices[0].message.content
```


```python
# Testing on a few examples
for _, row in df[:5].iterrows():
    description = row['Overview']
    title = row['Series_Title']
    result = get_categories(description)
    print(f"TITLE: {title}\nOVERVIEW: {description}\n\nRESULT: {result}")
    print("\n\n----------------------------\n\n")
```

    TITLE: The Shawshank Redemption
    OVERVIEW: Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.
    
    RESULT: {
        "categories": ["drama"],
        "summary": "Two imprisoned men develop a deep bond over the years, ultimately finding redemption through their shared acts of kindness."
    }
    
    
    ----------------------------
    
    
    TITLE: The Godfather
    OVERVIEW: An organized crime dynasty's aging patriarch transfers control of his clandestine empire to his reluctant son.
    
    RESULT: {
        "categories": ["crime", "drama"],
        "summary": "An aging crime lord hands over his empire to his hesitant son."
    }
    
    
    ----------------------------
    
    
    TITLE: The Dark Knight
    OVERVIEW: When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.
    
    RESULT: {
        "categories": ["action", "thriller", "superhero"],
        "summary": "Batman faces a formidable challenge as the Joker unleashes chaos on Gotham City."
    }
    
    
    ----------------------------
    
    
    TITLE: The Godfather: Part II
    OVERVIEW: The early life and career of Vito Corleone in 1920s New York City is portrayed, while his son, Michael, expands and tightens his grip on the family crime syndicate.
    
    RESULT: {
        "categories": ["crime", "drama"],
        "summary": "The film depicts the early life of Vito Corleone and the rise of his son Michael within the family crime syndicate in 1920s New York City."
    }
    
    
    ----------------------------
    
    
    TITLE: 12 Angry Men
    OVERVIEW: A jury holdout attempts to prevent a miscarriage of justice by forcing his colleagues to reconsider the evidence.
    
    RESULT: {
        "categories": ["drama", "thriller"],
        "summary": "A jury holdout fights to ensure justice is served by challenging his fellow jurors to reevaluate the evidence."
    }
    
    
    ----------------------------
    
    
    

### Creating the batch file

The batch file, in the `jsonl` format, should contain one line (json object) per request.
Each request is defined as such:

```
{
    "custom_id": <REQUEST_ID>,
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": <MODEL>,
        "messages": <MESSAGES>,
        // other parameters
    }
}
```

Note: the request ID should be unique per batch. This is what you can use to match results to the initial input files, as requests will not be returned in the same order.


```python
# Creating an array of json tasks

tasks = []

for index, row in df.iterrows():
    
    description = row['Overview']
    
    task = {
        "custom_id": f"task-{index}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            # This is what you would have in your Chat Completions API call
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "response_format": { 
                "type": "json_object"
            },
            "messages": [
                {
                    "role": "system",
                    "content": categorize_system_prompt
                },
                {
                    "role": "user",
                    "content": description
                }
            ],
        }
    }
    
    tasks.append(task)
```


```python
# Creating the file

file_name = "data/batch_tasks_movies.jsonl"

with open(file_name, 'w') as file:
    for obj in tasks:
        file.write(json.dumps(obj) + '\n')
```

### Uploading the file


```python
batch_file = client.files.create(
  file=open(file_name, "rb"),
  purpose="batch"
)
```


```python
print(batch_file)
```

    FileObject(id='file-lx16f1KyIxQ2UHVvkG3HLfNR', bytes=1127310, created_at=1721144107, filename='batch_tasks_movies.jsonl', object='file', purpose='batch', status='processed', status_details=None)
    

### Creating the batch job


```python
batch_job = client.batches.create(
  input_file_id=batch_file.id,
  endpoint="/v1/chat/completions",
  completion_window="24h"
)
```

### Checking batch status

Note: this can take up to 24h, but it will usually be completed faster.

You can continue checking until the status is 'completed'.


```python
batch_job = client.batches.retrieve(batch_job.id)
print(batch_job)
```

### Retrieving results


```python
result_file_id = batch_job.output_file_id
result = client.files.content(result_file_id).content
```


```python
result_file_name = "data/batch_job_results_movies.jsonl"

with open(result_file_name, 'wb') as file:
    file.write(result)
```


```python
# Loading data from saved file
results = []
with open(result_file_name, 'r') as file:
    for line in file:
        # Parsing the JSON string into a dict and appending to the list of results
        json_object = json.loads(line.strip())
        results.append(json_object)
```

### Reading results
Reminder: the results are not in the same order as in the input file.
Make sure to check the custom_id to match the results against the input requests


```python
# Reading only the first results
for res in results[:5]:
    task_id = res['custom_id']
    # Getting index from task id
    index = task_id.split('-')[-1]
    result = res['response']['body']['choices'][0]['message']['content']
    movie = df.iloc[int(index)]
    description = movie['Overview']
    title = movie['Series_Title']
    print(f"TITLE: {title}\nOVERVIEW: {description}\n\nRESULT: {result}")
    print("\n\n----------------------------\n\n")
```

    TITLE: American Psycho
    OVERVIEW: A wealthy New York City investment banking executive, Patrick Bateman, hides his alternate psychopathic ego from his co-workers and friends as he delves deeper into his violent, hedonistic fantasies.
    
    RESULT: {
        "categories": ["thriller", "psychological", "drama"],
        "summary": "A wealthy investment banker in New York City conceals his psychopathic alter ego while indulging in violent and hedonistic fantasies."
    }
    
    
    ----------------------------
    
    
    TITLE: Lethal Weapon
    OVERVIEW: Two newly paired cops who are complete opposites must put aside their differences in order to catch a gang of drug smugglers.
    
    RESULT: {
        "categories": ["action", "comedy", "crime"],
        "summary": "An action-packed comedy about two mismatched cops teaming up to take down a drug smuggling gang."
    }
    
    
    ----------------------------
    
    
    TITLE: A Star Is Born
    OVERVIEW: A musician helps a young singer find fame as age and alcoholism send his own career into a downward spiral.
    
    RESULT: {
        "categories": ["drama", "music"],
        "summary": "A musician's career spirals downward as he helps a young singer find fame amidst struggles with age and alcoholism."
    }
    
    
    ----------------------------
    
    
    TITLE: From Here to Eternity
    OVERVIEW: In Hawaii in 1941, a private is cruelly punished for not boxing on his unit's team, while his captain's wife and second-in-command are falling in love.
    
    RESULT: {
        "categories": ["drama", "romance", "war"],
        "summary": "A drama set in Hawaii in 1941, where a private faces punishment for not boxing on his unit's team, amidst a forbidden love affair between his captain's wife and second-in-command."
    }
    
    
    ----------------------------
    
    
    TITLE: The Jungle Book
    OVERVIEW: Bagheera the Panther and Baloo the Bear have a difficult time trying to convince a boy to leave the jungle for human civilization.
    
    RESULT: {
        "categories": ["adventure", "animation", "family"],
        "summary": "An adventure-filled animated movie about a panther and a bear trying to persuade a boy to leave the jungle for human civilization."
    }
    
    
    ----------------------------
    
    
    

## Second example: Captioning images

In this example, we will use `gpt-4-turbo` to caption images of furniture items. 

We will use the vision capabilities of the model to analyze the images and generate the captions.

### Loading data

We will use the Amazon furniture dataset for this example.


```python
dataset_path = "data/amazon_furniture_dataset.csv"
df = pd.read_csv(dataset_path)
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
      <th>asin</th>
      <th>url</th>
      <th>title</th>
      <th>brand</th>
      <th>price</th>
      <th>availability</th>
      <th>categories</th>
      <th>primary_image</th>
      <th>images</th>
      <th>upc</th>
      <th>...</th>
      <th>color</th>
      <th>material</th>
      <th>style</th>
      <th>important_information</th>
      <th>product_overview</th>
      <th>about_item</th>
      <th>description</th>
      <th>specifications</th>
      <th>uniq_id</th>
      <th>scraped_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B0CJHKVG6P</td>
      <td>https://www.amazon.com/dp/B0CJHKVG6P</td>
      <td>GOYMFK 1pc Free Standing Shoe Rack, Multi-laye...</td>
      <td>GOYMFK</td>
      <td>$24.99</td>
      <td>Only 13 left in stock - order soon.</td>
      <td>['Home &amp; Kitchen', 'Storage &amp; Organization', '...</td>
      <td>https://m.media-amazon.com/images/I/416WaLx10j...</td>
      <td>['https://m.media-amazon.com/images/I/416WaLx1...</td>
      <td>NaN</td>
      <td>...</td>
      <td>White</td>
      <td>Metal</td>
      <td>Modern</td>
      <td>[]</td>
      <td>[{'Brand': ' GOYMFK '}, {'Color': ' White '}, ...</td>
      <td>['Multiple layers: Provides ample storage spac...</td>
      <td>multiple shoes, coats, hats, and other items E...</td>
      <td>['Brand: GOYMFK', 'Color: White', 'Material: M...</td>
      <td>02593e81-5c09-5069-8516-b0b29f439ded</td>
      <td>2024-02-02 15:15:08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B0B66QHB23</td>
      <td>https://www.amazon.com/dp/B0B66QHB23</td>
      <td>subrtex Leather ding Room, Dining Chairs Set o...</td>
      <td>subrtex</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>['Home &amp; Kitchen', 'Furniture', 'Dining Room F...</td>
      <td>https://m.media-amazon.com/images/I/31SejUEWY7...</td>
      <td>['https://m.media-amazon.com/images/I/31SejUEW...</td>
      <td>NaN</td>
      <td>...</td>
      <td>Black</td>
      <td>Sponge</td>
      <td>Black Rubber Wood</td>
      <td>[]</td>
      <td>NaN</td>
      <td>['【Easy Assembly】: Set of 2 dining room chairs...</td>
      <td>subrtex Dining chairs Set of 2</td>
      <td>['Brand: subrtex', 'Color: Black', 'Product Di...</td>
      <td>5938d217-b8c5-5d3e-b1cf-e28e340f292e</td>
      <td>2024-02-02 15:15:09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B0BXRTWLYK</td>
      <td>https://www.amazon.com/dp/B0BXRTWLYK</td>
      <td>Plant Repotting Mat MUYETOL Waterproof Transpl...</td>
      <td>MUYETOL</td>
      <td>$5.98</td>
      <td>In Stock</td>
      <td>['Patio, Lawn &amp; Garden', 'Outdoor Décor', 'Doo...</td>
      <td>https://m.media-amazon.com/images/I/41RgefVq70...</td>
      <td>['https://m.media-amazon.com/images/I/41RgefVq...</td>
      <td>NaN</td>
      <td>...</td>
      <td>Green</td>
      <td>Polyethylene</td>
      <td>Modern</td>
      <td>[]</td>
      <td>[{'Brand': ' MUYETOL '}, {'Size': ' 26.8*26.8 ...</td>
      <td>['PLANT REPOTTING MAT SIZE: 26.8" x 26.8", squ...</td>
      <td>NaN</td>
      <td>['Brand: MUYETOL', 'Size: 26.8*26.8', 'Item We...</td>
      <td>b2ede786-3f51-5a45-9a5b-bcf856958cd8</td>
      <td>2024-02-02 15:15:09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B0C1MRB2M8</td>
      <td>https://www.amazon.com/dp/B0C1MRB2M8</td>
      <td>Pickleball Doormat, Welcome Doormat Absorbent ...</td>
      <td>VEWETOL</td>
      <td>$13.99</td>
      <td>Only 10 left in stock - order soon.</td>
      <td>['Patio, Lawn &amp; Garden', 'Outdoor Décor', 'Doo...</td>
      <td>https://m.media-amazon.com/images/I/61vz1Igler...</td>
      <td>['https://m.media-amazon.com/images/I/61vz1Igl...</td>
      <td>NaN</td>
      <td>...</td>
      <td>A5589</td>
      <td>Rubber</td>
      <td>Modern</td>
      <td>[]</td>
      <td>[{'Brand': ' VEWETOL '}, {'Size': ' 16*24INCH ...</td>
      <td>['Specifications: 16x24 Inch ', " High-Quality...</td>
      <td>The decorative doormat features a subtle textu...</td>
      <td>['Brand: VEWETOL', 'Size: 16*24INCH', 'Materia...</td>
      <td>8fd9377b-cfa6-5f10-835c-6b8eca2816b5</td>
      <td>2024-02-02 15:15:10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B0CG1N9QRC</td>
      <td>https://www.amazon.com/dp/B0CG1N9QRC</td>
      <td>JOIN IRON Foldable TV Trays for Eating Set of ...</td>
      <td>JOIN IRON Store</td>
      <td>$89.99</td>
      <td>Usually ships within 5 to 6 weeks</td>
      <td>['Home &amp; Kitchen', 'Furniture', 'Game &amp; Recrea...</td>
      <td>https://m.media-amazon.com/images/I/41p4d4VJnN...</td>
      <td>['https://m.media-amazon.com/images/I/41p4d4VJ...</td>
      <td>NaN</td>
      <td>...</td>
      <td>Grey Set of 4</td>
      <td>Iron</td>
      <td>X Classic Style</td>
      <td>[]</td>
      <td>NaN</td>
      <td>['Includes 4 Folding Tv Tray Tables And one Co...</td>
      <td>Set of Four Folding Trays With Matching Storag...</td>
      <td>['Brand: JOIN IRON', 'Shape: Rectangular', 'In...</td>
      <td>bdc9aa30-9439-50dc-8e89-213ea211d66a</td>
      <td>2024-02-02 15:15:11</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



### Processing step 

Again, we will first prepare our requests with the Chat Completions endpoint, and create the batch file afterwards.


```python
caption_system_prompt = '''
Your goal is to generate short, descriptive captions for images of items.
You will be provided with an item image and the name of that item and you will output a caption that captures the most important information about the item.
If there are multiple items depicted, refer to the name provided to understand which item you should describe.
Your generated caption should be short (1 sentence), and include only the most important information about the item.
The most important information could be: the type of item, the style (if mentioned), the material or color if especially relevant and/or any distinctive features.
Keep it short and to the point.
'''

def get_caption(img_url, title):
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=300,
    messages=[
        {
            "role": "system",
            "content": caption_system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": title
                },
                # The content type should be "image_url" to use gpt-4-turbo's vision capabilities
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url
                    }
                },
            ],
        }
    ]
    )

    return response.choices[0].message.content
```


```python
# Testing on a few images
for _, row in df[:5].iterrows():
    img_url = row['primary_image']
    caption = get_caption(img_url, row['title'])
    img = Image(url=img_url)
    display(img)
    print(f"CAPTION: {caption}\n\n")
```


<img src="https://m.media-amazon.com/images/I/416WaLx10jL._SS522_.jpg"/>


    CAPTION: A stylish white free-standing shoe rack featuring multiple layers and eight double hooks, perfect for organizing shoes and accessories in living rooms, bathrooms, or hallways.
    
    
    


<img src="https://m.media-amazon.com/images/I/31SejUEWY7L._SS522_.jpg"/>


    CAPTION: Set of 2 black leather dining chairs featuring a sleek design with vertical stitching and sturdy wooden legs.
    
    
    


<img src="https://m.media-amazon.com/images/I/41RgefVq70L._SS522_.jpg"/>


    CAPTION: The MUYETOL Plant Repotting Mat is a waterproof, portable, and foldable gardening work mat measuring 26.8" x 26.8", designed for easy soil changing and indoor transplanting.
    
    
    


<img src="https://m.media-amazon.com/images/I/61vz1IglerL._SS522_.jpg"/>


    CAPTION: Absorbent non-slip doormat featuring the phrase "It's a good day to play PICKLEBALL" with paddle graphics, measuring 16x24 inches.
    
    
    


<img src="https://m.media-amazon.com/images/I/41p4d4VJnNL._SS522_.jpg"/>


    CAPTION: Set of 4 foldable TV trays in grey, featuring a compact design with a stand for easy storage, perfect for small spaces.
    
    
    

### Creating the batch job

As with the first example, we will create an array of json tasks to generate a `jsonl` file and use it to create the batch job.


```python
# Creating an array of json tasks

tasks = []

for index, row in df.iterrows():
    
    title = row['title']
    img_url = row['primary_image']
    
    task = {
        "custom_id": f"task-{index}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            # This is what you would have in your Chat Completions API call
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": 300,
            "messages": [
                {
                    "role": "system",
                    "content": caption_system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": title
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_url
                            }
                        },
                    ],
                }
            ]            
        }
    }
    
    tasks.append(task)
```


```python
# Creating the file

file_name = "data/batch_tasks_furniture.jsonl"

with open(file_name, 'w') as file:
    for obj in tasks:
        file.write(json.dumps(obj) + '\n')
```


```python
# Uploading the file 

batch_file = client.files.create(
  file=open(file_name, "rb"),
  purpose="batch"
)
```


```python
# Creating the job

batch_job = client.batches.create(
  input_file_id=batch_file.id,
  endpoint="/v1/chat/completions",
  completion_window="24h"
)
```


```python
batch_job = client.batches.retrieve(batch_job.id)
print(batch_job)
```

### Getting results

As with the first example, we can retrieve results once the batch job is done.

Reminder: the results are not in the same order as in the input file.
Make sure to check the custom_id to match the results against the input requests


```python
# Retrieving result file

result_file_id = batch_job.output_file_id
result = client.files.content(result_file_id).content
```


```python
result_file_name = "data/batch_job_results_furniture.jsonl"

with open(result_file_name, 'wb') as file:
    file.write(result)
```


```python
# Loading data from saved file

results = []
with open(result_file_name, 'r') as file:
    for line in file:
        # Parsing the JSON string into a dict and appending to the list of results
        json_object = json.loads(line.strip())
        results.append(json_object)
```


```python
# Reading only the first results
for res in results[:5]:
    task_id = res['custom_id']
    # Getting index from task id
    index = task_id.split('-')[-1]
    result = res['response']['body']['choices'][0]['message']['content']
    item = df.iloc[int(index)]
    img_url = item['primary_image']
    img = Image(url=img_url)
    display(img)
    print(f"CAPTION: {result}\n\n")
```


<img src="https://m.media-amazon.com/images/I/31FOa-k+EtL._SS522_.jpg"/>


    CAPTION: Brushed brass pedestal towel rack with a sleek, modern design, featuring multiple bars for hanging towels, measuring 25.75 x 14.44 x 32 inches.
    
    
    


<img src="https://m.media-amazon.com/images/I/41z8YktAkGL._SS522_.jpg"/>


    CAPTION: Black round end table featuring a tempered glass top and a metal frame, with a lower shelf for additional storage.
    
    
    


<img src="https://m.media-amazon.com/images/I/511N0PuE9EL._SS522_.jpg"/>


    CAPTION: Black collapsible and height-adjustable telescoping stool, portable and designed for makeup artists and hairstylists, shown in various stages of folding for easy transport.
    
    
    


<img src="https://m.media-amazon.com/images/I/31dCSKQ14YL._SS522_.jpg"/>


    CAPTION: Ergonomic pink gaming chair featuring breathable fabric, adjustable height, lumbar support, a footrest, and a swivel recliner function.
    
    
    


<img src="https://m.media-amazon.com/images/I/51OPfpn9ovL._SS522_.jpg"/>


    CAPTION: A set of two Glitzhome adjustable bar stools featuring a mid-century modern design with swivel seats, PU leather upholstery, and wooden backrests.
    
    
    

## Wrapping up

In this cookbook, we have seen two examples of how to use the new Batch API, but keep in mind that the Batch API works the same way as the Chat Completions endpoint, supporting the same parameters and most of the recent models (gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo...).

By using this API, you can significantly reduce costs, so we recommend switching every workload that can happen async to a batch job with this new API.
