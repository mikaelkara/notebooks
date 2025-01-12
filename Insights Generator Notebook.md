```python
%load_ext autoreload
%autoreload 2
```

# Product Reviews Insights Generator

Takes in product reviews and generates high level and statistical insights.

First, some setup.


```python
from typing import List, Dict
from aiohttp import web
import json



from insights_generator import util
import insights_generator.core.utils as core_utils
import insights_generator.core.nl_query as nl_query
import insights_generator.core.extract_sentiment_aspects as extract_sentiment_aspects
import insights_generator.core.extract_summary as extract_summary
import insights_generator.core.extract_top_tags as extract_top_tags
import insights_generator.core.extract_statistical_summary as extract_statistical_summary

from datetime import datetime, timezone
from pathlib import Path
import logging
import pdb
import asyncio
import dateutil.parser

import os

def json_load_from_filename(filename):
    print('Reading json from file: ' + filename)
    file = open(filename, encoding = 'utf-8')
    results = json.load(file)
    return(results)

def json_dump_to_filename(data, filename):
    print('Writing json to file: ' + filename)
    file = open(filename, 'w', encoding = 'utf-8')
    results = json.dump(data, file, indent = 4, sort_keys = True)
    file.close()
    return

# Azure OpenAI configurations. This code was run on text-davinci-003
os.environ["AOAI_ENDPOINT"] = "YOUR_AZURE_OPENAI_ENDPOINT_URL"
os.environ["AOAI_KEY"] = "YOUR_KEY"

store_summary_locally = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

```


```python
# Create the project
project_object = {
    "name" : "headphones-1",
    "productCategory" : "headphones",
    "productName" : "Contoso headphones"
    }
print("Project metadata: ")
print(project_object)

    
# Read in reviews data
reviews_filename = "sample_reviews/contoso_headphones.json"
reviews = json_load_from_filename(reviews_filename)
```

    Project metadata: 
    {'name': 'headphones-1', 'productCategory': 'headphones', 'productName': 'Contoso headphones'}
    Reading json from file: sample_reviews/contoso_headphones.json
    

# Identify topics being discussed in the reviews.

First get aspects (i.e. topics) for each batch of reviews. Batching is done to standardize the aspect names across reviews.


```python
async def get_per_review_aspects(project_object, reviews):
    """Get aspects from each batch of reviews
    
    Reviews are batched.
    Sentiment aspects are extracted from each batch as key value pairs
    and stored with the key sentiment_aspects_key_value
    The aspects thus extracted will be later used to get the top topics across all reviews.
    The batching helps with standardizing the wording of the aspects.
    
    :param project_object: metadata of the project
    :param reviews: reviews data
    :type project_object: dict
    :type reviews: list

    :returns: list of batched reviews, with new key sentiment_aspects_key_value added".
    :rtype: list
    """

    # Extract sentiment aspects from the reviews
    product_category = project_object["productCategory"]
    product_name = project_object["productName"]
    batch_size = 6
    #batch_size = 1
    reviews_with_sentiment_aspects = extract_sentiment_aspects.extract_sentiment_aspects(reviews, product_name, batch_size)

    return(reviews_with_sentiment_aspects)


# Get per review aspects
reviews_with_sentiment_aspects = await get_per_review_aspects(project_object, reviews)
print("Sentiment Aspects extracted for reviews.")
print("Sample review with sentiment aspects:")
print(json.dumps(reviews_with_sentiment_aspects[0], indent = 4))
json_dump_to_filename(reviews_with_sentiment_aspects, "reviews_with_sentiment_aspects.json")
```

    Extracting sentiment aspects for 16 reviews
    Starting 4 parallel extractors
    Extracted sentiment aspects for 16 reviews in 3.282043933868408 seconds
    Done.
    Sentiment Aspects extracted for reviews.
    Sample review with sentiment aspects:
    {
        "review_text": "review: I recently purchased the Contoso 100 headphones and I'm really impressed with the sound quality. The noise cancellation is great and I can block out all outside noise. My only complaint is the battery life, which isn't as long as I'd like it to be.\nreview: I've been using the Contoso 100 headphones for a few weeks now and I'm really pleased with them. The sound quality is excellent and the noise cancellation works great. The only downside is the battery life, which seems to run out too quickly.\nreview: I recently bought the Contoso 100 headphones and I'm very happy with them. The sound quality is great and the noise cancellation is top notch. The only issue I have is with the battery life, which doesn't seem to last as long as I'd like.\nreview: I've been using the Contoso 100 headphones for a few weeks now and I'm really impressed. The sound quality is excellent and the noise cancellation works really well. My only complaint is with the battery life, which doesn't seem to last very long.\nreview: I recently bought the Contoso 100 headphones and I'm really happy with them. The sound quality is great and the noise cancellation works really well. My only issue is with the battery life, which seems to run out too quickly.\nreview: I've been using the Contoso 100 headphones for a few weeks now and I'm really pleased with them. The sound quality is excellent and the noise cancellation works great. My only complaint is with the battery life, which doesn't seem to last very long.",
        "sentiment_aspects": [
            [
                "sound quality",
                "positive"
            ],
            [
                "noise cancellation",
                "positive"
            ],
            [
                "battery life",
                "negative"
            ]
        ]
    }
    Writing json to file: reviews_with_sentiment_aspects.json
    

Next get the most frequent aspects, by counting occurrences across batches.


```python
async def get_top_aspects(project_object, reviews_with_sentiment_aspects):
    """Get the top aspects
    
    Get the aspects that occur the most across all reviews.
    
    :param project_object: metadata of the project
    :param reviews_with_sentiment_aspects: list of batched reviews with aspects information
    :type project_object: dict
    :type reviews_with_sentiment_aspects: list
    :returns: list of top aspects and their occurrence count in review batches.
    :rtype: list

    
    :Example:
    
    example of top aspects
    ['product quality', 'sound quality', 'ease of use', 'battery life', 'noise cancellation', 'comfort']
    
    """

    # Extract top aspects
    top_aspects = extract_top_tags.extract_top_tags(reviews_with_sentiment_aspects)

    # Store top aspects in project metadata, used later.
    project_object["top_aspects"] = top_aspects

    return(project_object, top_aspects)


# Get the top aspects
project_object, top_aspects = await get_top_aspects(project_object, reviews_with_sentiment_aspects)
print("Top aspects and scores:")
print(json.dumps(list(zip(*top_aspects)), indent = 4))
top_aspects = top_aspects[0]


```

    Top aspects and scores:
    [
        [
            "sound quality",
            3
        ],
        [
            "noise cancellation",
            3
        ],
        [
            "battery life",
            3
        ]
    ]
    

# Summarize the Top Topics (Aspects)


An aspect (i.e. topic) is summarized both in NL (natural language) and statistically.

1. Extracting sentiment + keyphrases for each review, for that aspect
2. Statistical summary of sentiments is done
3. Keyphrases are summarized as a NL (natural language) summary.

This aspect level summary is a lot more accurate and scalable than summarizing all aspects across all reviews.


```python
async def extract_statistics(project_object, reviews, top_aspects):
    
    """Extract statistics for each top aspect
    
    An aspect can have three type of sentiments in a review: "positive", "negative" or "unmentioned".
    Other sentiments besides positive and negative may be extracted by GPTx, but we summarize these as "unmentioned".
    For each review, identify the sentiment for top aspects. Add this to the review object as "top_aspect_sentiments".
    For each top aspect, get the sentiment statistics across reviews.
    
    For each aspect, we check if there are a significant number of mentions (at least 5 "positive" or "negative").
    For each aspect, we check if there is a dominant sentiment. Either "positive" should have > 75% mentions.
    Or "negative" should have > 75% mentions.  
    Currently, we do not process aspects that have insignificant mentions or are mixed in overall sentiment.
    If an aspect satisfies these conditions, we also calculate the following additional fields:
    "overall_sentiment", "percentage_positive", "keyphrases" (per review), "aspect_summary" and "aspect_action_items".
    
    The "keyphrases" are ...
    The "aspect_summary" is got by summarizing the "keyphrases" for this aspect, across all reviews.
    Similarly for "aspect_action_items".
    
    Returns the list of reviews with "top_aspect_sentiments" added, and the statistics for top aspects.
    
    :param project_object: metadata of the project
    :param reviews: list of reviews with aspects information
    :param top_aspects: list of top aspects
    :type project_object: dict
    :type reviews: list
    :type top_aspects: list
    :returns: list of reviews with "top_aspect_sentiments" added, dict of statistics + summaries for each top aspect
    :rtype: list, dict
    """


    product_category = project_object["productCategory"]
    # Perform statistical summary
    reviews_with_statistics, statistical_summary = extract_statistical_summary.extract_statistical_summary(product_category, reviews, top_aspects)

    return(reviews_with_statistics, statistical_summary)

# Extract statistical summaries
reviews_with_statistics, statistical_summary = await extract_statistics(project_object, reviews, top_aspects)
```

    Extracting keyphrases for sound quality
    Done.
    Extracting keyphrases for noise cancellation
    Done.
    Extracting keyphrases for battery life
    Done.
    Extracted all statistics
    


```python
print("Sample review with top aspect sentiments:")
print(json.dumps(reviews_with_statistics[0], indent = 4))
```

    Sample review with top aspect sentiments:
    {
        "location": "US",
        "profile_name": "James Smith",
        "review_text": "I recently purchased the Contoso 100 headphones and I'm really impressed with the sound quality. The noise cancellation is great and I can block out all outside noise. My only complaint is the battery life, which isn't as long as I'd like it to be.",
        "top_aspect_sentiments": {
            "sound quality": "positive",
            "noise cancellation": "positive",
            "battery life": "negative"
        }
    }
    


```python
print("Aspect level summary, example for " + top_aspects[0] + ":")
print(json.dumps(statistical_summary[top_aspects[0]], indent = 4))
```

    Aspect level summary, example for sound quality:
    {
        "positive": 13,
        "unmentioned": 2,
        "negative": 1,
        "overall_sentiment": "positive",
        "percentage_positive": 92.85714285714286,
        "keyphrases": [
            {
                "keyphrases": [
                    " Great sound quality",
                    " Excellent noise cancellation",
                    " Battery life not as long as desired"
                ],
                "profile_name": "James Smith",
                "location": "US"
            },
            {
                "keyphrases": [
                    " Excellent sound quality",
                    " Great noise cancellation"
                ],
                "profile_name": "Jessica Johnson",
                "location": "US"
            },
            {
                "keyphrases": [
                    " Great sound quality",
                    " Top notch noise cancellation"
                ],
                "profile_name": "Adam Williams",
                "location": "US"
            },
            {
                "keyphrases": [
                    " Excellent sound quality",
                    " Effective noise cancellation"
                ],
                "profile_name": "Samantha Jones",
                "location": "US"
            },
            {
                "keyphrases": [
                    " Great sound quality",
                    " Effective noise cancellation"
                ],
                "profile_name": "Matthew Brown",
                "location": "US"
            },
            {
                "keyphrases": [
                    " Excellent sound quality",
                    " Great noise cancellation"
                ],
                "profile_name": "Emily Davis",
                "location": "US"
            },
            {
                "keyphrases": [
                    " Great sound quality",
                    " Excellent noise cancellation",
                    " Battery life not as long as desired"
                ],
                "profile_name": "John Taylor",
                "location": "US"
            },
            {
                "keyphrases": [
                    " Excellent sound quality",
                    " Great noise cancellation"
                ],
                "profile_name": "Olivia White",
                "location": "US"
            },
            {
                "keyphrases": [
                    " Great sound quality",
                    " Top notch noise cancellation"
                ],
                "profile_name": "Daniel Moore",
                "location": "US"
            },
            {
                "keyphrases": [
                    " Excellent sound quality",
                    " Effective noise cancellation"
                ],
                "profile_name": "Charlotte Anderson",
                "location": "US"
            },
            {
                "keyphrases": [
                    " Great sound quality",
                    " Effective noise cancellation"
                ],
                "profile_name": "William Thomas",
                "location": "US"
            },
            {
                "keyphrases": [
                    " Impressive sound quality",
                    " Effective noise cancellation capabilities"
                ],
                "profile_name": "Joshua Jackson",
                "location": "US"
            },
            {
                "keyphrases": [
                    " Excellent sound quality",
                    " Effective noise cancellation"
                ],
                "profile_name": "Amelia Harris",
                "location": "US"
            }
        ],
        "aspect_summary": "The comments indicate that the sound quality is excellent with great noise cancellation.",
        "aspect_action_items": null
    }
    

# Overall NL (Natural Language) Summaries

The aspect level summaries from previous step are clubbed into high level NL (natural language) summaries.


```python
async def get_summary(project_object, statistical_summary):
    
    """Create high level summaries of the per aspect summaries.
    
    Extract various types of summaries: overall, highlights, lowlights and action items from the per aspect summaries.
    We only summarize aspects that have significant number of mentions and a dominant sentiment.
        
    :param project_object: metadata of the project
    :param statistical_summary: dict of statistics and summaries for top aspects
    :type project_object: dict
    :type statistical_summary: dict
    :returns: dict of various highlevel summaries
    :rtype: dict
    """

    product_category = project_object["productCategory"]
    
    # Extract summary from the reviews
    summary = extract_summary.extract_summary(product_category, statistical_summary)
    
    return(summary)

# Extract summary
summary = await get_summary(project_object, statistical_summary)
print(json.dumps(summary, indent = 4))

```

    Extracted NL version of statistical summary.
    {
        "overall": "The comments indicate that the sound quality and noise cancellation of the headphones are good, but the battery life is poor.",
        "highlights": "The comments about the headphones indicate that they have excellent sound quality and great noise cancellation, with the latter being particularly impressive, as it is able to block out all outside noise.",
        "lowlights": "The comments about the headphones indicate that the battery life is not satisfactory, as it runs out quickly and does not last as long as desired.",
        "action_items": {
            "battery life": "Specific issues regarding battery life include that it is not as expected, runs out quickly, and does not last as long as desired. To improve battery life, the manufacturer could use higher quality components, optimize the power management system, and reduce the power consumption of the device"
        }
    }
    
