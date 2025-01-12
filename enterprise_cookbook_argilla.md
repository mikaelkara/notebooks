# Data Annotation with Argilla Spaces
_Authored by: [Moritz Laurer](https://huggingface.co/MoritzLaurer)_

This notebook illustrates the workflow for systematically evaluating LLM outputs and creating LLM training data. You can start by using this notebook to evaluate the zero-shot performance of your favorite LLM on your task without any fine-tuning. If you want to improve performance, you can then easily reuse this workflow to create training data.

**Example use case: code generation.** In this tutorial, we demonstrate how to create high-quality test and train data for code generation tasks. The same workflow can, however, be adapted to any other task relevant to your specific use case.

**In this notebook, we:**
1. Download data for the example task.
2. Prompt two LLMs to respond to these tasks. This results in "synthetic data" to speed up manual data creation. 
3. Create an Argilla annotation interface on HF Spaces to compare and evaluate the outputs from the two LLMs.
4. Upload the example data and the zero-shot LLM responses into the Argilla annotation interface.
5. Download the annotated data.

You can adapt this notebook to your needs, e.g., using a different LLM and API provider for step (2) or adapting the annotation task in step (3).

## Install required packages and connect to HF Hub


```python
!pip install argilla~=2.0.0
!pip install transformers~=4.40.0
!pip install datasets~=2.19.0
!pip install huggingface_hub~=0.23.2
```


```python
# Login to the HF Hub. We recommend using this login method 
# to avoid the need to explicitly store your HF token in variables 
import huggingface_hub
!git config --global credential.helper store
huggingface_hub.login(add_to_git_credential=True)
```

## Download example task data

First, we download an example dataset containing LLMs' code generation tasks. We want to evaluate how well two different LLMs perform on these code-generation tasks. We use instructions from the [bigcode/self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k) dataset that was used to train the [StarCoder2-Instruct](https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1) model.


```python
from datasets import load_dataset

# Small sample for faster testing
dataset_codetask = load_dataset("bigcode/self-oss-instruct-sc2-exec-filter-50k", split="train[:3]")
print("Dataset structure:\n", dataset_codetask, "\n")

# We are only interested in the instructions/prompts provided in the dataset
instructions_lst = dataset_codetask["instruction"]
print("Example instructions:\n", instructions_lst[:2])
```

    Dataset structure:
     Dataset({
        features: ['fingerprint', 'sha1', 'seed', 'response', 'concepts', 'prompt', 'instruction', 'id'],
        num_rows: 3
    }) 
    
    Example instructions:
     ['Write a Python function named `get_value` that takes a matrix (represented by a list of lists) and a tuple of indices, and returns the value at that index in the matrix. The function should handle index out of range errors by returning None.', 'Write a Python function `check_collision` that takes a list of `rectangles` as input and checks if there are any collisions between any two rectangles. A rectangle is represented as a tuple (x, y, w, h) where (x, y) is the top-left corner of the rectangle, `w` is the width, and `h` is the height.\n\nThe function should return True if any pair of rectangles collide, and False otherwise. Use an iterative approach and check for collisions based on the bounding box collision detection algorithm. If a collision is found, return True immediately without checking for more collisions.']
    

## Prompt two LLMs on the example task

#### Formatting the instructions with a chat_template
Before sending the instructions to an LLM API, we need to format the instructions with the correct `chat_template` for each of the models we want to evaluate. This essentially entails wrapping some special tokens around the instructions. See the [docs](https://huggingface.co/docs/transformers/main/en/chat_templating) on chat templates for details.


```python
# Apply correct chat formatting to instructions from the dataset 
from transformers import AutoTokenizer

models_to_compare = ["mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-70B-Instruct"]

def format_prompt(prompt, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    messages_tokenized = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    return messages_tokenized


prompts_formatted_dic = {}
for model in models_to_compare:
    tokenizer = AutoTokenizer.from_pretrained(model)

    prompt_formatted = []
    for instruction in instructions_lst: 
        prompt_formatted.append(format_prompt(instruction, tokenizer))
        
    prompts_formatted_dic.update({model: prompt_formatted})


print(f"\nFirst prompt formatted for {models_to_compare[0]}:\n\n", prompts_formatted_dic[models_to_compare[0]][0], "\n\n")
print(f"First prompt formatted for {models_to_compare[1]}:\n\n", prompts_formatted_dic[models_to_compare[1]][0], "\n\n")

```

    None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
    /home/user/miniconda/lib/python3.9/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    

    
    First prompt formatted for mistralai/Mixtral-8x7B-Instruct-v0.1:
    
     <s>[INST] Write a Python function named `get_value` that takes a matrix (represented by a list of lists) and a tuple of indices, and returns the value at that index in the matrix. The function should handle index out of range errors by returning None. [/INST] 
    
    
    First prompt formatted for meta-llama/Meta-Llama-3-70B-Instruct:
    
     <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    
    Write a Python function named `get_value` that takes a matrix (represented by a list of lists) and a tuple of indices, and returns the value at that index in the matrix. The function should handle index out of range errors by returning None.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
     
    
    
    

#### Sending the instructions to the HF Inference API
Now, we can send the instructions to the APIs for both LLMs to get outputs we can evaluate. We first define some parameters for generating the responses correctly. Hugging Face's LLM APIs are powered by [Text Generation Inference (TGI)](https://huggingface.co/docs/text-generation-inference/index) containers. See the TGI OpenAPI specifications [here](https://huggingface.github.io/text-generation-inference/#/Text%20Generation%20Inference/generate) and the explanations of different parameters in the Transformers Generation Parameters [docs](https://huggingface.co/docs/transformers/v4.30.0/main_classes/text_generation#transformers.GenerationConfig). 


```python
generation_params = dict(
    # we use low temperature and top_p to reduce creativity and increase likelihood of highly probable tokens
    temperature=0.2,
    top_p=0.60,
    top_k=None,
    repetition_penalty=1.0,
    do_sample=True,
    max_new_tokens=512*2,
    return_full_text=False,
    seed=42,
    #details=True,
    #stop=["<|END_OF_TURN_TOKEN|>"],
    #grammar={"type": "json"}
    max_time=None, 
    stream=False,
    use_cache=False,
    wait_for_model=False,
)
```

Now, we can make a standard API request to the Serverless Inference API ([docs](https://huggingface.co/docs/api-inference/index)). Note that the Serverless Inference API is mostly for testing and is rate-limited. For testing without rate limits, you can create your own API via the HF Dedicated Endpoints ([docs](https://huggingface.co/docs/inference-endpoints/index)). See also our corresponding tutorials in the [Open Source AI Cookbook](https://huggingface.co/learn/cookbook/index).

> [!TIP]
> The code below will be updated once the Inference API recipe is finished.


```python
import requests
from tqdm.auto import tqdm

# Hint: use asynchronous API calls (and dedicated endpoints) to increase speed
def query(payload=None, api_url=None):
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()

headers = {"Authorization": f"Bearer {huggingface_hub.get_token()}"}

output_dic = {}
for model in models_to_compare:
    # Create API urls for each model
    # When using dedicated endpoints, you can reuse the same code and simply replace this URL
    api_url = "https://api-inference.huggingface.co/models/" + model
    
    # send requests to API 
    output_lst = []
    for prompt in tqdm(prompt_formatted):
        output = query(
            payload={
                "inputs": prompt,
                "parameters": {**generation_params}
            },
            api_url=api_url 
        )
        output_lst.append(output[0]["generated_text"])
    
    output_dic.update({model: output_lst})

print(f"---First generation of {models_to_compare[0]}:\n{output_dic[models_to_compare[0]][0]}\n\n")
print(f"---First generation of {models_to_compare[1]}:\n{output_dic[models_to_compare[1]][0]}")
```


      0%|          | 0/3 [00:00<?, ?it/s]



      0%|          | 0/3 [00:00<?, ?it/s]


    ---First generation of mistralai/Mixtral-8x7B-Instruct-v0.1:
    Here's a Python function that meets your requirements:
    
    ```python
    def get_value(matrix, indices):
        try:
            return matrix[indices[0]][indices[1]]
        except IndexError:
            return None
    ```
    
    This function takes a matrix (represented by a list of lists) and a tuple of indices as input. It first tries to access the value at the given indices in the matrix. If the indices are out of range, it catches the `IndexError` exception and returns `None`.
    
    
    ---First generation of meta-llama/Meta-Llama-3-70B-Instruct:
    Here is a Python function that does what you described:
    ```
    def get_value(matrix, indices):
        try:
            row, col = indices
            return matrix[row][col]
        except IndexError:
            return None
    ```
    Here's an explanation of how the function works:
    
    1. The function takes two arguments: `matrix` (a list of lists) and `indices` (a tuple of two integers, representing the row and column indices).
    2. The function tries to access the value at the specified indices using `matrix[row][col]`.
    3. If the indices are out of range (i.e., `row` or `col` is greater than the length of the corresponding dimension of the matrix), an `IndexError` exception is raised.
    4. The `except` block catches the `IndexError` exception and returns `None` instead of raising an error.
    
    Here's an example usage of the function:
    ```
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    
    print(get_value(matrix, (0, 0)))  # prints 1
    print(get_value(matrix, (1, 1)))  # prints 5
    print(get_value(matrix, (3, 0)))  # prints None (out of range)
    print(get_value(matrix, (0, 3)))  # prints None (out of range)
    ```
    I hope this helps! Let me know if you have any questions.
    

#### Store the LLM outputs in a dataset
We can now store the LLM outputs in a dataset together with the original instructions.


```python
# create a HF dataset with the instructions and model outputs
from datasets import Dataset

dataset = Dataset.from_dict({
    "instructions": instructions_lst,
    "response_model_1": output_dic[models_to_compare[0]],
    "response_model_2": output_dic[models_to_compare[1]]
})

dataset
```




    Dataset({
        features: ['instructions', 'response_model_1', 'response_model_2'],
        num_rows: 3
    })



## Create and configure your Argilla dataset

We use [Argilla](https://argilla.io/), a collaboration tool for AI engineers and domain experts who need to build high-quality datasets for their projects.

We run Argilla via a HF Space, which you can set up with just a few clicks without any local setup. You can create the HF Argilla Space by following [these instructions](https://docs.argilla.io/latest/getting_started/quickstart/). For further configuration on HF Argilla Spaces, see also the detailed [documentation](https://docs.argilla.io/latest/getting_started/how-to-configure-argilla-on-huggingface/). If you want, you can also run Argilla locally via Argilla's docker containers (see [Argilla docs](https://docs.argilla.io/latest/getting_started/how-to-deploy-argilla-with-docker/)).

![Argilla login screen](https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/argilla-login-screen.png)

#### Programmatically interact with Argilla

Before we can tailor the dataset to our specific task and upload the data that will be shown in the UI, we need to first set up a few things.

**Connecting this notebook to Argilla:** We can now connect this notebook to Argilla to programmatically configure your dataset and upload/download data. 


```python
# After starting the Argilla Space (or local docker container) you can connect to the Space with the code below.
import argilla as rg

client = rg.Argilla(
    api_url="https://username-spacename.hf.space",  # Locally: "http://localhost:6900"
    api_key="your-apikey",  # You'll find it in the UI "My Settings > API key"
    # To use a private HF Argilla Space, also pass your HF token
    headers={"Authorization": f"Bearer {huggingface_hub.get_token()}"},
)
```


```python
user = client.me
user
```

#### Write good annotator guidelines 
Writing good guidelines for your human annotators is just as important (and difficult) as writing good training code. Good instructions should fulfill the following criteria: 
- **Simple and clear**: The guidelines should be simple and clear to understand for people who do not know anything about your task yet. Always ask at least one colleague to reread the guidelines to make sure that there are no ambiguities. 
- **Reproducible and explicit**: All information for doing the annotation task should be contained in the guidelines. A common mistake is to create informal interpretations of the guidelines during conversations with selected annotators. Future annotators will not have this information and might do the task differently than intended if it is not made explicit in the guidelines.
- **Short and comprehensive**: The guidelines should as short as possible, while containing all necessary information. Annotators tend not to read long guidelines properly, so try to keep them as short as possible, while remaining comprehensive.

Note that creating annotator guidelines is an iterative process. It is good practice to do a few dozen annotations yourself and refine the guidelines based on your learnings from the data before assigning the task to others. Versioning the guidelines can also help as the task evolves over time. See further tips in this [blog post](https://argilla.io/blog/annotation-guidelines-practices/).


```python
annotator_guidelines = """\
Your task is to evaluate the responses of two LLMs to code generation tasks. 

First, you need to score each response on a scale from 0 to 7. You add points to your final score based on the following criteria:
- Add up to +2 points, if the code is properly commented, with inline comments and doc strings for functions.
- Add up to +2 points, if the code contains a good example for testing. 
- Add up to +3 points, if the code runs and works correctly. Copy the code into an IDE and test it with at least two different inputs. Attribute one point if the code is overall correct, but has some issues. Attribute three points if the code is fully correct and robust against different scenarios. 
Your resulting final score can be any value between 0 to 7. 

If both responses have a final score of <= 4, select one response and correct it manually in the text field. 
The corrected response must fulfill all criteria from above. 
"""

rating_tooltip = """\
- Add up to +2 points, if the code is properly commented, with inline comments and doc strings for functions.
- Add up to +2 points, if the code contains a good example for testing. 
- Add up to +3 points, if the code runs and works correctly. Copy the code into an IDE and test it with at least two different inputs. Attribute one point if the code works mostly correctly, but has some issues. Attribute three points if the code is fully correct and robust against different scenarios. 
"""
```

**Cumulative ratings vs. Likert scales:** Note that the guidelines above ask the annotators to do cumulative ratings by adding points for explicit criteria. An alternative approach are "Likert scales", where annotators are asked to rate responses on a continuous scale e.g. from 1 (very bad) to 3 (mediocre) to 5 (very good). We generally recommend cumulative ratings, because they force you and the annotators to make quality criteria explicit, while just rating a response as "4" (good) is ambiguous and will be interpreted differently by different annotators. 

#### Tailor your Argilla dataset to your specific task

We can now create our own `code-llm` task with the fields, questions, and metadata required for annotation. For more information on configuring the Argilla dataset, see the [Argilla docs](https://docs.argilla.io/latest/how_to_guides/dataset/#create-a-dataset).



```python
dataset_argilla_name = "code-llm"
workspace_name = "argilla"
reuse_existing_dataset = False  # for easier iterative testing

# Configure your dataset settings
settings = rg.Settings(
    # The overall annotation guidelines, which human annotators can refer back to inside of the interface
    guidelines="my guidelines",
    fields=[
        rg.TextField(
            name="instruction", title="Instruction:", use_markdown=True, required=True
        ),
        rg.TextField(
            name="generation_1",
            title="Response model 1:",
            use_markdown=True,
            required=True,
        ),
        rg.TextField(
            name="generation_2",
            title="Response model 2:",
            use_markdown=True,
            required=True,
        ),
    ],
    # These are the questions we ask annotators about the fields in the dataset
    questions=[
        rg.RatingQuestion(
            name="score_response_1",
            title="Your score for the response of model 1:",
            description="0=very bad, 7=very good",
            values=[0, 1, 2, 3, 4, 5, 6, 7],
            required=True,
        ),
        rg.RatingQuestion(
            name="score_response_2",
            title="Your score for the response of model 2:",
            description="0=very bad, 7=very good",
            values=[0, 1, 2, 3, 4, 5, 6, 7],
            required=True,
        ),
        rg.LabelQuestion(
            name="which_response_corrected",
            title="If both responses score below 4, select a response to correct:",
            description="Select the response you will correct in the text field below.",
            labels=["Response 1", "Response 2", "Combination of both", "Neither"],
            required=False,
        ),
        rg.TextQuestion(
            name="correction",
            title="Paste the selected response below and correct it manually:",
            description="Your corrected response must fulfill all criteria from the annotation guidelines.",
            use_markdown=True,
            required=False,
        ),
        rg.TextQuestion(
            name="comments",
            title="Annotator Comments",
            description="Add any additional comments here. E.g.: edge cases, issues with the interface etc.",
            use_markdown=True,
            required=False,
        ),
    ],
    metadata=[
        rg.TermsMetadataProperty(
            name="source-dataset",
            title="Original dataset source",
        ),
    ],
    allow_extra_metadata=False,
)

if reuse_existing_dataset:
    dataset_argilla = client.datasets(dataset_argilla_name, workspace=workspace_name)
else:
    dataset_argilla = rg.Dataset(
        name=dataset_argilla_name,
        settings=settings,
        workspace=workspace_name,
    )
    if client.datasets(dataset_argilla_name, workspace=workspace_name) is not None:
        client.datasets(dataset_argilla_name, workspace=workspace_name).delete()
    dataset_argilla = dataset_argilla.create()

dataset_argilla
```

After running the code above, you will see the new custom `code-llm` dataset in Argilla (and any other dataset you might have created before).



#### Load the data to Argilla

At this point, the dataset is still empty. Let's load some data with the code below.


```python
# Iterate over the samples in the dataset
records = [
    rg.Record(
        fields={
            "instruction": example["instructions"],
            "generation_1": example["response_model_1"],
            "generation_2": example["response_model_2"],
        },
        metadata={
            "source-dataset": "bigcode/self-oss-instruct-sc2-exec-filter-50k",
        },
        # Optional: add suggestions from an LLM-as-a-judge system
        # They will be indicated with a sparkle icon and shown as pre-filled responses
        # It will speed up manual annotation
        # suggestions=[
        #     rg.Suggestion(
        #         question_name="score_response_1",
        #         value=example["llm_judge_rating"],
        #         agent="llama-3-70b-instruct",
        #     ),
        # ],
    )
    for example in dataset
]

try:
    dataset_argilla.records.log(records)
except Exception as e:
    print("Exception:", e)
```

**The Argilla UI for annotation** will look similar to this:

![Argilla UI](https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/argilla-code-llm.png)

## Annotate

That's it, we've created our Argilla dataset and we can now start annotating in the UI! By default, the records will be completed when they have 1 annotation. Check these guides, to know how to [automatically distribute the annotation task](https://docs.argilla.io/latest/how_to_guides/distribution/) and [annotate in Argilla](https://docs.argilla.io/latest/how_to_guides/annotate/).


**Important**: If you use Argilla in a HF Space, you'd to activate persistent storage so that your data is safely stored and not automatically deleted after a while. For production settings, make sure that persistent storage is activated **before** making any annotations to avoid data loss.   

## Download annotated data
After annotating, you can pull the data from Argilla and simply store and process them locally in any tabular format (see [docs here](https://docs.argilla.io/latest/how_to_guides/import_export/)). You can also download the filtered version of the dataset ([docs](https://docs.argilla.io/latest/how_to_guides/query/)).


```python
annotated_dataset = client.datasets(dataset_argilla_name, workspace=workspace_name)

hf_dataset = annotated_dataset.records.to_datasets()

# This HF dataset can then be formatted, stored and processed into any tabular data format
hf_dataset.to_pandas()
```


```python
# Store the dataset locally
hf_dataset.to_csv("argilla-dataset-local.csv")  # Save as CSV
#hf_dataset.to_json("argilla-dataset-local.json")  # Save as JSON
#hf_dataset.save_to_disk("argilla-dataset-local")  # Save as a `datasets.Dataset` in the local filesystem
#hf_dataset.to_parquet()  # Save as Parquet
```

## Next Steps

That's it! You've created synthetic LLM data with the HF inference API, created a dataset in Argilla, uploaded the LLM data into Argilla, evaluated/corrected the data, and after annotation you have downloaded the data in a simple tabular format for downstream use. 

We have specifically designed the pipeline and the interface for **two main use-cases**: 
1. Evaluation: You can now simply use the numeric scores in the `score_response_1` and `score_response_2` columns to calculate which model was better overall. You can also inspect responses with very low or high ratings for a detailed error analysis. As you test or train different models, you can reuse this pipeline and track improvements of different models over time. 
2. Training: After annotating enough data, you can create a train-test split from the data and fine-tune your own model. You can either use highly rated response texts for supervised fine-tuning with the the [TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer), or you can directly use the ratings for preference-tuning techniques like DPO with the [TRL DPOTrainer](https://huggingface.co/docs/trl/en/dpo_trainer). See the [TRL docs](https://huggingface.co/docs/trl/en/index) for the pros and cons of different LLM fine-tuning techniques. 

**Adapt and improve:** Many things can be improved to tailor this pipeline to your specific use-cases. For example, you can prompt an LLM to evaluate the outputs of the two LLMs with instructions very similar to the guidelines for human annotators ("LLM-as-a-judge" approach). This can help further speed up your evaluation pipeline. See our [LLM-as-a-judge recipe](https://huggingface.co/learn/cookbook/llm_judge) for an example implementation of LLM-as-a-judge and our overall [Open-Source AI Cookbook](https://huggingface.co/learn/cookbook/index) for many other ideas. 



