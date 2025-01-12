# Amazon Personalize

[Amazon Personalize](https://docs.aws.amazon.com/personalize/latest/dg/what-is-personalize.html) is a fully managed machine learning service that uses your data to generate item recommendations for your users. It can also generate user segments based on the users' affinity for certain items or item metadata.

This notebook goes through how to use Amazon Personalize Chain. You need a Amazon Personalize campaign_arn or a recommender_arn before you get started with the below notebook.

Following is a [tutorial](https://github.com/aws-samples/retail-demo-store/blob/master/workshop/1-Personalization/Lab-1-Introduction-and-data-preparation.ipynb) to setup a campaign_arn/recommender_arn on Amazon Personalize. Once the campaign_arn/recommender_arn is setup, you can use it in the langchain ecosystem. 


## 1. Install Dependencies


```python
!pip install boto3
```

## 2. Sample Use-cases

### 2.1 [Use-case-1] Setup Amazon Personalize Client and retrieve recommendations


```python
from langchain_experimental.recommenders import AmazonPersonalize

recommender_arn = "<insert_arn>"

client = AmazonPersonalize(
    credentials_profile_name="default",
    region_name="us-west-2",
    recommender_arn=recommender_arn,
)
client.get_recommendations(user_id="1")
```

### 2.2 [Use-case-2] Invoke Personalize Chain for summarizing results


```python
from langchain.llms.bedrock import Bedrock
from langchain_experimental.recommenders import AmazonPersonalizeChain

bedrock_llm = Bedrock(model_id="anthropic.claude-v2", region_name="us-west-2")

# Create personalize chain
# Use return_direct=True if you do not want summary
chain = AmazonPersonalizeChain.from_llm(
    llm=bedrock_llm, client=client, return_direct=False
)
response = chain({"user_id": "1"})
print(response)
```

### 2.3 [Use-Case-3] Invoke Amazon Personalize Chain using your own prompt


```python
from langchain.prompts.prompt import PromptTemplate

RANDOM_PROMPT_QUERY = """
You are a skilled publicist. Write a high-converting marketing email advertising several movies available in a video-on-demand streaming platform next week, 
    given the movie and user information below. Your email will leverage the power of storytelling and persuasive language. 
    The movies to recommend and their information is contained in the <movie> tag. 
    All movies in the <movie> tag must be recommended. Give a summary of the movies and why the human should watch them. 
    Put the email between <email> tags.

    <movie>
    {result} 
    </movie>

    Assistant:
    """

RANDOM_PROMPT = PromptTemplate(input_variables=["result"], template=RANDOM_PROMPT_QUERY)

chain = AmazonPersonalizeChain.from_llm(
    llm=bedrock_llm, client=client, return_direct=False, prompt_template=RANDOM_PROMPT
)
chain.run({"user_id": "1", "item_id": "234"})
```

### 2.4 [Use-case-4] Invoke Amazon Personalize in a Sequential Chain 


```python
from langchain.chains import LLMChain, SequentialChain

RANDOM_PROMPT_QUERY_2 = """
You are a skilled publicist. Write a high-converting marketing email advertising several movies available in a video-on-demand streaming platform next week, 
    given the movie and user information below. Your email will leverage the power of storytelling and persuasive language. 
    You want the email to impress the user, so make it appealing to them.
    The movies to recommend and their information is contained in the <movie> tag. 
    All movies in the <movie> tag must be recommended. Give a summary of the movies and why the human should watch them. 
    Put the email between <email> tags.

    <movie>
    {result}
    </movie>

    Assistant:
    """

RANDOM_PROMPT_2 = PromptTemplate(
    input_variables=["result"], template=RANDOM_PROMPT_QUERY_2
)
personalize_chain_instance = AmazonPersonalizeChain.from_llm(
    llm=bedrock_llm, client=client, return_direct=True
)
random_chain_instance = LLMChain(llm=bedrock_llm, prompt=RANDOM_PROMPT_2)
overall_chain = SequentialChain(
    chains=[personalize_chain_instance, random_chain_instance],
    input_variables=["user_id"],
    verbose=True,
)
overall_chain.run({"user_id": "1", "item_id": "234"})
```

### 2.5 [Use-case-5] Invoke Amazon Personalize and retrieve metadata 


```python
recommender_arn = "<insert_arn>"
metadata_column_names = [
    "<insert metadataColumnName-1>",
    "<insert metadataColumnName-2>",
]
metadataMap = {"ITEMS": metadata_column_names}

client = AmazonPersonalize(
    credentials_profile_name="default",
    region_name="us-west-2",
    recommender_arn=recommender_arn,
)
client.get_recommendations(user_id="1", metadataColumns=metadataMap)
```

### 2.6 [Use-Case 6] Invoke Personalize Chain with returned metadata for summarizing results


```python
bedrock_llm = Bedrock(model_id="anthropic.claude-v2", region_name="us-west-2")

# Create personalize chain
# Use return_direct=True if you do not want summary
chain = AmazonPersonalizeChain.from_llm(
    llm=bedrock_llm, client=client, return_direct=False
)
response = chain({"user_id": "1", "metadata_columns": metadataMap})
print(response)
```
