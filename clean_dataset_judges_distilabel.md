# Clean an Existing Preference Dataset with LLMs as Judges

_Authored by: [David Berenstein](https://huggingface.co/davidberenstein1957) and [Sara Han Díaz](https://huggingface.co/sdiazlor)_

- **Libraries**: [argilla](https://github.com/argilla-io/argilla), [hf-inference-endpoints](https://github.com/huggingface/huggingface_hub)
- **Components**: [LoadDataFromDicts](https://distilabel.argilla.io/dev/components-gallery/steps/loaddatafromdicts/), [UltraFeedback](https://distilabel.argilla.io/latest/components-gallery/tasks/ultrafeedback/), [KeepColumns](https://distilabel.argilla.io/latest/components-gallery/steps/groupcolumns/), [PreferenceToArgilla](https://distilabel.argilla.io/latest/components-gallery/steps/textgenerationtoargilla/), [InferenceEndpointsLLM](https://distilabel.argilla.io/latest/components-gallery/llms/inferenceendpointsllm/), [GlobalStep](https://distilabel.argilla.io/latest/sections/how_to_guides/basic/step/global_step/)

In this tutorial, we'll use distilabel to clean a dataset using the LLMs as judges by providing AI feedback on the quality of the data. [distilabel](https://github.com/argilla-io/distilabel) is a synthetic data and AI feedback framework for engineers who need fast, reliable and scalable pipelines based on verified research papers. Check the documentation [here](https://distilabel.argilla.io/latest/).

To evaluate the responses, we will use the [serverless HF Inference API](https://huggingface.co/docs/api-inference/index) integrated with distilabel. This is free but rate-limited, allowing you to test and evaluate over 150,000 public models, or your own private models, via simple HTTP requests, with fast inference hosted on Hugging Face shared infrastructure. If you need more compute power, you can deploy your own inference endpoint with [Hugging Face Inference Endpoints](https://huggingface.co/docs/inference-endpoints/guides/create_endpoint).

Finally, to further curate the data, we will use [Argilla](https://github.com/argilla-io/argilla), which allows us to provide human feedback on the data quality. Argilla is a collaboration tool for AI engineers and domain experts who need to build high-quality datasets for their projects. Check the documentation [here](https://docs.argilla.io/latest/).



## Getting Started

### Install the dependencies

To complete this tutorial, you need to install the distilabel SDK and a few third-party libraries via pip.


```python
!pip install "distilabel[hf-inference-endpoints]"
```


```python
!pip install "transformers~=4.0" "torch~=2.0"
```

Let's make the required imports:


```python
import random

from datasets import load_dataset

from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    KeepColumns,
    LoadDataFromDicts,
    PreferenceToArgilla,
)
from distilabel.steps.tasks import UltraFeedback
```

You'll need an `HF_TOKEN` to use the HF Inference Endpoints. Login to use it directly within this notebook.


```python
import os
from huggingface_hub import login

login(token=os.getenv("HF_TOKEN"), add_to_git_credential=True)
```

### (optional) Deploy Argilla

You can skip this step or replace it with any other data evaluation tool, but the quality of your model will suffer from a lack of data quality, so we do recommend looking at your data. If you already deployed Argilla, you can skip this step. Otherwise, you can quickly deploy Argilla following [this guide](https://docs.argilla.io/latest/getting_started/quickstart/). 

Along with that, you will need to install Argilla as a distilabel extra.


```python
!pip install "distilabel[argilla, hf-inference-endpoints]"
```

## The dataset

In this case, we will clean a preference dataset, so we will use the [`Intel/orca_dpo_pairs`](https://huggingface.co/datasets/Intel/orca_dpo_pairs) dataset from the Hugging Face Hub.

<iframe
  src="https://huggingface.co/datasets/Intel/orca_dpo_pairs/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>


```python
dataset = load_dataset("Intel/orca_dpo_pairs", split="train[:20]")
```

Next, we will shuffle the `chosen` and `rejected` columns to avoid any bias in the dataset.


```python
def shuffle_and_track(chosen, rejected):
    pair = [chosen, rejected]
    random.shuffle(pair)
    order = ["chosen" if x == chosen else "rejected" for x in pair]
    return {"generations": pair, "order": order}

dataset = dataset.map(lambda x: shuffle_and_track(x["chosen"], x["rejected"]))
```


```python
dataset = dataset.to_list()
```

### (optional) Create a custom step

A step is a block in a distilabel pipeline used to manipulate, generate, or evaluate data, among other tasks. A set of predefined steps is provided, but you can also create your [own custom steps](https://distilabel.argilla.io/latest/sections/how_to_guides/basic/step/#defining-custom-steps). Instead of preprocessing the data as in the previous section, it is possible to use a custom step to shuffle the columns. This step should be in a separate module to be imported and used in the pipeline. In this case, the pipeline would start by loading the `orca_dpo_pairs` dataset using the `LoadDataFromHub` step and then applying the `ShuffleStep`.


```python
# "shuffle_step.py"
from typing import TYPE_CHECKING, List
from distilabel.steps import GlobalStep, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput
    
import random

class ShuffleStep(GlobalStep):
    @property
    def inputs(self) -> List[str]:
        return ["instruction", "chosen", "rejected"]

    @property
    def outputs(self) -> List[str]:
        return ["instruction", "generations", "order"]

    def process(self, inputs: StepInput) -> "StepOutput":
        outputs = []

        for input in inputs:
            chosen = input["chosen"]
            rejected = input["rejected"]
            pair = [chosen, rejected]
            random.shuffle(pair)
            order = ["chosen" if x == chosen else "rejected" for x in pair]
            
            outputs.append({"instruction": input["instruction"], "generations": pair, "order": order})

        yield outputs
```


```python
from shuffle_step import ShuffleStep
```

## Define the pipeline

To clean an existing preference dataset, we will need to define a `Pipeline` with all the necessary steps. However, a similar workflow can be used to clean an SFT dataset. Below, we will go over each step in detail.

### Load the dataset
We will use the dataset we just shuffled as source data.

- Component: `LoadDataFromDicts`
- Input columns: `system`, `question`, `chosen`, `rejected`, `generations` and `order`, the same keys as in the loaded list of dictionaries.
- Output columns: `system`, `instruction`, `chosen`, `rejected`, `generations` and `order`. We will use `output_mappings` to rename the columns.


```python
load_dataset = LoadDataFromDicts(
    data=dataset[:1],
    output_mappings={"question": "instruction"},
    pipeline=Pipeline(name="showcase-pipeline"),
)
load_dataset.load()
next(load_dataset.process())
```




    ([{'system': '',
       'question': "You will be given a definition of a task first, then some input of the task.\nThis task is about using the specified sentence and converting the sentence to Resource Description Framework (RDF) triplets of the form (subject, predicate object). The RDF triplets generated must be such that the triplets accurately capture the structure and semantics of the input sentence. The input is a sentence and the output is a list of triplets of the form [subject, predicate, object] that capture the relationships present in the sentence. When a sentence has more than 1 RDF triplet possible, the output must contain all of them.\n\nAFC Ajax (amateurs)'s ground is Sportpark De Toekomst where Ajax Youth Academy also play.\nOutput:",
       'chosen': '[\n  ["AFC Ajax (amateurs)", "has ground", "Sportpark De Toekomst"],\n  ["Ajax Youth Academy", "plays at", "Sportpark De Toekomst"]\n]',
       'rejected': " Sure, I'd be happy to help! Here are the RDF triplets for the input sentence:\n\n[AFC Ajax (amateurs), hasGround, Sportpark De Toekomst]\n[Ajax Youth Academy, playsAt, Sportpark De Toekomst]\n\nExplanation:\n\n* AFC Ajax (amateurs) is the subject of the first triplet, and hasGround is the predicate that describes the relationship between AFC Ajax (amateurs) and Sportpark De Toekomst.\n* Ajax Youth Academy is the subject of the second triplet, and playsAt is the predicate that describes the relationship between Ajax Youth Academy and Sportpark De Toekomst.\n\nNote that there may be other possible RDF triplets that could be derived from the input sentence, but the above triplets capture the main relationships present in the sentence.",
       'generations': [" Sure, I'd be happy to help! Here are the RDF triplets for the input sentence:\n\n[AFC Ajax (amateurs), hasGround, Sportpark De Toekomst]\n[Ajax Youth Academy, playsAt, Sportpark De Toekomst]\n\nExplanation:\n\n* AFC Ajax (amateurs) is the subject of the first triplet, and hasGround is the predicate that describes the relationship between AFC Ajax (amateurs) and Sportpark De Toekomst.\n* Ajax Youth Academy is the subject of the second triplet, and playsAt is the predicate that describes the relationship between Ajax Youth Academy and Sportpark De Toekomst.\n\nNote that there may be other possible RDF triplets that could be derived from the input sentence, but the above triplets capture the main relationships present in the sentence.",
        '[\n  ["AFC Ajax (amateurs)", "has ground", "Sportpark De Toekomst"],\n  ["Ajax Youth Academy", "plays at", "Sportpark De Toekomst"]\n]'],
       'order': ['rejected', 'chosen']}],
     True)



### Evaluate the responses

To evaluate the quality of the responses, we will use [`meta-llama/Meta-Llama-3.1-70B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct), applying the `UltraFeedback` task that judges the responses according to different dimensions (helpfulness, honesty, instruction-following, truthfulness). For an SFT dataset, you can use [`PrometheusEval`](../papers/prometheus.md) instead.

- Component: `UltraFeedback` task with LLMs using `InferenceEndpointsLLM`
- Input columns: `instruction`, `generations`
- Output columns: `ratings`, `rationales`, `distilabel_metadata`, `model_name`

For your use case and to improve the results, you can use any [other LLM of your choice](https://distilabel.argilla.io/latest/components-gallery/llms/).


```python
evaluate_responses = UltraFeedback(
    aspect="overall-rating",
    llm=InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
    ),
    pipeline=Pipeline(name="showcase-pipeline"),
)
evaluate_responses.load()
next(
    evaluate_responses.process(
        [
            {
                "instruction": "What's the capital of Spain?",
                "generations": ["Madrid", "Barcelona"],
            }
        ]
    )
)
```




    [{'instruction': "What's the capital of Spain?",
      'generations': ['Madrid', 'Barcelona'],
      'ratings': [5, 1],
      'rationales': ["The answer is correct, directly addressing the question, and is free of hallucinations or unnecessary details. It confidently provides the accurate information, aligning perfectly with the user's intent.",
       "The answer is incorrect as Barcelona is not the capital of Spain. This introduces a significant inaccuracy, failing to provide helpful information and deviating entirely from the user's intent."],
      'distilabel_metadata': {'raw_output_ultra_feedback_0': "#### Output for Text 1\nRating: 5 (Excellent)\nRationale: The answer is correct, directly addressing the question, and is free of hallucinations or unnecessary details. It confidently provides the accurate information, aligning perfectly with the user's intent.\n\n#### Output for Text 2\nRating: 1 (Low Quality)\nRationale: The answer is incorrect as Barcelona is not the capital of Spain. This introduces a significant inaccuracy, failing to provide helpful information and deviating entirely from the user's intent."},
      'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct'}]



### Keep only the required columns

We will get rid of the unneeded columns.

- Component: `KeepColumns`
- Input columns: `system`, `instruction`, `chosen`, `rejected`, `generations`, `ratings`, `rationales`, `distilabel_metadata` and `model_name`
- Output columns: `instruction`, `chosen`, `rejected`, `generations` and `order`


```python
keep_columns = KeepColumns(
    columns=[
        "instruction",
        "generations",
        "order",
        "ratings",
        "rationales",
        "model_name",
    ],
    pipeline=Pipeline(name="showcase-pipeline"),
)
keep_columns.load()
next(
    keep_columns.process(
        [
            {
                "system": "",
                "instruction": "What's the capital of Spain?",
                "chosen": "Madrid",
                "rejected": "Barcelona",
                "generations": ["Madrid", "Barcelona"],
                "order": ["chosen", "rejected"],
                "ratings": [5, 1],
                "rationales": ["", ""],
                "model_name": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            }
        ]
    )
)
```




    [{'instruction': "What's the capital of Spain?",
      'generations': ['Madrid', 'Barcelona'],
      'order': ['chosen', 'rejected'],
      'ratings': [5, 1],
      'rationales': ['', ''],
      'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct'}]



### (Optional) Further data curation

You can use Argilla to further curate your data.

-  Component: `PreferenceToArgilla` step
- Input columns: `instruction`, `generations`, `generation_models`, `ratings`
- Output columns: `instruction`, `generations`, `generation_models`, `ratings`


```python
to_argilla = PreferenceToArgilla(
    dataset_name="cleaned-dataset",
    dataset_workspace="argilla",
    api_url="https://[your-owner-name]-[your-space-name].hf.space",
    api_key="[your-api-key]",
    num_generations=2
)
```

## Run the pipeline

Below, you can see the full pipeline definition:


```python
with Pipeline(name="clean-dataset") as pipeline:

    load_dataset = LoadDataFromDicts(
        data=dataset, output_mappings={"question": "instruction"}
    )

    evaluate_responses = UltraFeedback(
        aspect="overall-rating",
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
        ),
    )

    keep_columns = KeepColumns(
        columns=[
            "instruction",
            "generations",
            "order",
            "ratings",
            "rationales",
            "model_name",
        ]
    )

    to_argilla = PreferenceToArgilla(
        dataset_name="cleaned-dataset",
        dataset_workspace="argilla",
        api_url="https://[your-owner-name]-[your-space-name].hf.space",
        api_key="[your-api-key]",
        num_generations=2,
    )

    load_dataset.connect(evaluate_responses)
    evaluate_responses.connect(keep_columns)
    keep_columns.connect(to_argilla)
```

Let's now run the pipeline and clean our preference dataset.


```python
distiset = pipeline.run()
```

Let's check it! If you have loaded the data to Argilla, you can [start annotating in the Argilla UI](https://docs.argilla.io/latest/how_to_guides/annotate/).

You can push the dataset to the Hub for sharing with the community and [embed it to explore the data](https://huggingface.co/docs/hub/datasets-viewer-embed).


```python
distiset.push_to_hub("[your-owner-name]/example-cleaned-preference-dataset")
```

<iframe
  src="https://huggingface.co/datasets/distilabel-internal-testing/example-cleaned-preference-dataset/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

## Conclusions

In this tutorial, we showcased the detailed steps to build a pipeline for cleaning a preference dataset using distilabel. However, you can customize this pipeline for your own use cases, such as cleaning an SFT dataset or adding custom steps.

We used a preference dataset as our starting point and shuffled the data to avoid any bias. Next, we evaluated the responses using a model through the serverless Hugging Face Inference API, following the UltraFeedback standards. Finally, we kept the needed columns and used Argilla for further curation.
