# Distillation step-by-step
### A step-by-step guide

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/language/tuning/distilling_step_by_step/distilling_step_by_step.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/tuning/distilling_step_by_step/distilling_step_by_step.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/tuning/distilling_step_by_step/distilling_step_by_step.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Anirudh Haritas Murali](https://github.com/anihm136) |

# Overview

**Distillation** is a technique in machine learning that allows us to extract the learnings of a large model and represent it using a smaller model. This allows for improved scalability, as the smaller model requires less resources to run and less time to generate inferences while still achieving accuracy close to that of the larger model.

Traditionally, distillation uses the internal parameters of the larger model (specifically, the logits) to train the smaller model. However, some of the best performing large language models today, including Google's [PaLM 2](https://ai.google/discover/palm2/) model, are exposed to consumers as an API, with no means to access the internal parameters. Until recently, this has prohibited the use of these models as teacher models for distillation.

## Objectives
In this notebook, we will go over the technique described in the paper [Distilling step-by-step](https://blog.research.google/2023/09/distilling-step-by-step-outperforming.html), which describes a novel approach to distill the knowledge of a large LLM into a smaller LLM without requiring the internal parameters of the larger model. The original code from the research is available at [https://github.com/google-research/distilling-step-by-step](https://github.com/google-research/distilling-step-by-step).

We will go through each step of training a small (student) model to mimic the reasoning ability of a larger (teacher) model. By training the student model to mimic the reasoning ability rather than the actual outputs, we can make the smaller model generalize better to other unseen inputs.

The steps performed include:

- Preparing a dataset for distillation
- Setting up a distillation pipeline
- Training a student model using PaLM as a teacher model
- Evaluating the distilled model's performance
- Deploying the distilled model to Vertex AI

## Costs
This tutorial uses billable components of Google Cloud:
- Vertex AI
- Cloud Storage
- Artifact Registry
- Cloud Build
    
Learn about [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing), [Cloud Storage pricing](https://cloud.google.com/storage/pricing), [Artifact Registry pricing](https://cloud.google.com/artifact-registry/pricing) and [Cloud Build pricing](https://cloud.google.com/build/pricing) and use the [Pricing Calculator](https://cloud.google.com/products/calculator/) to generate a cost estimate based on your projected usage.

# Getting started

## (Only on Colab) Authenticate as user
On Colab, we will authenticate as a user that has access to the Google Cloud resources mentioned above. This will be needed when we deploy the model


```
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

## Download supporting files
To simplify the process of running this demo, some supporting files are provided (PaLM outputs for the dataset and code for building the model serving container)


```
! gsutil -m cp -r gs://github-repo/distillation/* .
! wget https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/tuning/distilling_step_by_step/requirements.txt
! wget https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/tuning/distilling_step_by_step/prediction_container/Dockerfile -P prediction_container
! wget https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/tuning/distilling_step_by_step/prediction_container/app/main.py -P prediction_container/app
! wget https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/tuning/distilling_step_by_step/prediction_container/app/requirements.txt -P prediction_container/app
! wget https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/tuning/distilling_step_by_step/prediction_container/app/requirements-torch.txt -P prediction_container/app
! wget https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/tuning/distilling_step_by_step/prediction_container/app/prestart.sh -P prediction_container/app
```

## Install required libraries


```
%pip install -r requirements.txt
```

## Enable required Google Cloud APIs
For ease of cleaning up resources, you can create a new project and delete it at the end of this tutorial


```
PROJECT = ""  # @param {type:"string"}
REGION = "us-central1"  # @param {type:"string"}
```


```
!gcloud services enable aiplatform.googleapis.com --project {PROJECT}
!gcloud services enable artifactregistry.googleapis.com --project {PROJECT}
!gcloud services enable cloudbuild.googleapis.com --project {PROJECT}
```

# Step 1: Data preparation

Our dataset will need three fields -
1. An input prompt for the LLM
2. A ground truth label, which is the expected output
3. A 'rationale', which is the reasoning generated by the teacher model (using CoT prompting)

Here, we will use the [Common Sense Explanations](https://huggingface.co/datasets/cos_e) dataset from Hugging Face to train our student model. This dataset contains around 10k training samples and 1.2k test samples. We will use pre-generated rationales from the PaLM model as a teacher, and we will preprocess the dataset to fit the above schema


```
from typing import Any

from datasets import DatasetDict, load_dataset
```


```
SOURCE_DATASET = "cos_e"  # @param {type:"string"}
SOURCE_DATASET_VERSION = "v1.11"  # @param {type:"string"}

dataset = load_dataset(SOURCE_DATASET, SOURCE_DATASET_VERSION)
dataset["test"] = dataset["validation"]
del dataset["validation"]
```


```
def prepare_input(example: dict[str, Any]) -> dict[str, Any]:
    question = example["question"]
    c_0 = example["choices"][0]
    c_1 = example["choices"][1]
    c_2 = example["choices"][2]
    c_3 = example["choices"][3]
    c_4 = example["choices"][4]

    input = f"{question}\nAnswer Choices:\n(a) {c_0}\n(b) {c_1}\n(c) {c_2}\n(d) {c_3}\n(e) {c_4}"

    example["input"] = input
    example["label"] = example["answer"]

    return example


dataset = dataset.map(
    prepare_input,
    remove_columns=[
        "id",
        "question",
        "choices",
        "answer",
        "abstractive_explanation",
        "extractive_explanation",
    ],
)
```


```
LLM_OUTPUTS_FILE_PREFIX = "PaLM_CoT"  # @param {type:"string"}
LLM_OUTPUTS_FILE = LLM_OUTPUTS_FILE_PREFIX + "_{split}.json"


def add_llm_outputs(dataset: DatasetDict, split: str) -> None:
    llm_ds = load_dataset("json", data_files=LLM_OUTPUTS_FILE.format(split=split))[
        "train"
    ]

    def _add(example: dict[str, Any], idx: int) -> dict[str, Any]:
        example["llm_rationale"] = llm_ds[idx]["rationale"]
        example["llm_label"] = llm_ds[idx]["label"]
        return example

    dataset[split] = dataset[split].map(_add, with_indices=True)


for split in ["train", "test"]:
    add_llm_outputs(dataset, split)
```

# Step 2: Build the model


```
import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
)
```

Here, we will use the T5 model as a pretrained base for distillation, and we will use the corresponding tokenizer. You can use a different pretrained model (and corresponding tokenizer) by changing the name of the model below to a different model on Hugging Face Hub, or use a custom model/train a tokenizer from scratch on your own dataset. Note that you will need significantly more data and compute to train a good model from scratch


```
PRETRAINED_BASE_MODEL = "google/flan-t5-base"  # @param {type:"string"}
MAX_INPUT_LENGTH = 1024  # @param {type:"integer"}
MAX_OUTPUT_LENGTH = 256  # @param {type:"integer"}
```

## a) Prepare the tokenizer and tokenize the dataset


```
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_BASE_MODEL)


def tokenize_function(examples: dict[str, list[Any]]):
    # Encode input to generate predictions and rationales
    model_inputs = tokenizer(
        ["predict: " + text for text in examples["input"]],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )
    expl_model_inputs = tokenizer(
        ["explain: " + text for text in examples["input"]],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )
    model_inputs["expl_input_ids"] = expl_model_inputs["input_ids"]
    model_inputs["expl_attention_mask"] = expl_model_inputs["attention_mask"]

    # Encode target label and target rationale
    label_output_encodings = tokenizer(
        text_target=examples["label"], max_length=MAX_OUTPUT_LENGTH, truncation=True
    )
    rationale_output_encodings = tokenizer(
        text_target=examples["llm_rationale"],
        max_length=MAX_OUTPUT_LENGTH,
        truncation=True,
    )
    model_inputs["labels"] = label_output_encodings["input_ids"]
    model_inputs["expl_labels"] = rationale_output_encodings["input_ids"]

    return model_inputs


tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=["input", "llm_rationale", "label", "llm_label"],
    batched=True,
)
```

## b) Prepare the model


```
model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_BASE_MODEL)
# Uncomment if you have more than one GPU to enable parallelism
# model.parallelize()
```

## c) Prepare data collator for multi-task training
Since we need to generate predictions for both the answer as well as the rationale on each training and prediction step, we will use a custom DataCollator which will take each batch of features and return two sets of features and labels, one each for the answer and for the rationale


```
class TaskPrefixDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)

        # Generate features for answers
        ans_features = features_df.loc[
            :, features_df.columns.isin(["labels", "input_ids", "attention_mask"])
        ].to_dict("records")
        ans_features = super().__call__(ans_features, return_tensors)

        # Generate features for explanations
        expl_features = (
            features_df.loc[
                :,
                features_df.columns.isin(
                    ["expl_labels", "expl_input_ids", "expl_attention_mask"]
                ),
            ]
            .rename(
                columns={
                    "expl_labels": "labels",
                    "expl_input_ids": "input_ids",
                    "expl_attention_mask": "attention_mask",
                }
            )
            .to_dict("records")
        )
        expl_features = super().__call__(expl_features, return_tensors)

        return {
            "ans": ans_features,
            "expl": expl_features,
        }


data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
```

## d) Prepare trainer for multi-task training
Similarly, we will use a custom Trainer for training the model, which takes into account both the losses for answer generation as well as rationale generation. We will use a hyperparameter `alpha` to control the relative contribution of the two losses to the overall model loss


```
class TaskPrefixTrainer(Seq2SeqTrainer):
    def __init__(self, alpha, output_rationale, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.output_rationale = output_rationale

    def compute_loss(self, model, inputs, return_outputs=False):
        ans_outputs = model(**inputs["ans"])
        expl_outputs = model(**inputs["expl"])

        loss = self.alpha * ans_outputs.loss + (1.0 - self.alpha) * expl_outputs.loss

        return (
            (loss, {"ans": ans_outputs, "expl": expl_outputs})
            if return_outputs
            else loss
        )

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        ans_outputs = super().prediction_step(
            model, inputs["ans"], prediction_loss_only=False, ignore_keys=ignore_keys
        )
        if self.output_rationale:
            expl_outputs = super().prediction_step(
                model,
                inputs["expl"],
                prediction_loss_only=False,
                ignore_keys=ignore_keys,
            )
        else:
            expl_outputs = ans_outputs  # placeholder only

        loss = self.alpha * ans_outputs[0] + (1 - self.alpha) * expl_outputs[0]

        return (
            loss,
            [ans_outputs[1], expl_outputs[1]],
            [ans_outputs[2], expl_outputs[2]],
        )
```

# Step 3: Train the model


```
import numpy as np
from transformers import Seq2SeqTrainingArguments
from transformers.trainer_utils import set_seed
```


```
RUN_ID = 0  # @param {type:"integer"}
CONFIG_DIR = "distillation_outputs"  # @param {type:"string"}
CKPT_DIR = f"{CONFIG_DIR}/ckpts/{RUN_ID}"  # for model checkpoints
LOG_DIR = f"{CONFIG_DIR}/logs/{RUN_ID}"  # for training logs

EVAL_STEPS = 500  # @param {type:"integer"}
SAVE_STEPS = 1000  # @param {type:"integer"}
MAX_STEPS = 4000  # @param {type:"integer"}

LEARNING_RATE = 5e-5
BATCH_SIZE = 16

ALPHA = 0.5
```


```
set_seed(RUN_ID)

training_args = Seq2SeqTrainingArguments(
    CKPT_DIR,
    remove_unused_columns=False,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    logging_dir=LOG_DIR,
    logging_strategy="steps",
    logging_steps=EVAL_STEPS,
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    predict_with_generate=True,
    seed=RUN_ID,
    local_rank=-1,
    bf16=False,
    generation_max_length=64,
    prediction_loss_only=False,
)
```


```
from collections.abc import Callable

from transformers import AutoTokenizer


def compute_metrics_text(tokenizer: AutoTokenizer) -> Callable:
    def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions[0], skip_special_tokens=True)

        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {"accuracy": acc}

    return compute_metrics


compute_metrics = compute_metrics_text(tokenizer)
```


```
trainer_kwargs = {
    "alpha": ALPHA,
    "output_rationale": False,
    "model": model,
    "args": training_args,
    "train_dataset": tokenized_dataset["train"],
    "eval_dataset": {
        "test": tokenized_dataset["test"],
    },
    "data_collator": data_collator,
    "tokenizer": tokenizer,
    "compute_metrics": compute_metrics,
}
```


```
trainer = TaskPrefixTrainer(**trainer_kwargs)
trainer.train()
```

# Step 4: Evaluate the model

Now let's compare the performance of our distilled student model against the PaLM model. We will also try to generate outputs from the base student model to compare the difference that the distilled training method has made.


```
from transformers import pipeline
```


```
CHECKPOINT = f"{CKPT_DIR}/checkpoint-4000"

distilled_tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
distilled_model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)

base_tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_BASE_MODEL)
base_model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_BASE_MODEL)
```


```
distill_generator = pipeline(
    "text2text-generation", model=distilled_model, tokenizer=distilled_tokenizer
)
base_generator = pipeline(
    "text2text-generation", model=base_model, tokenizer=base_tokenizer
)


def generate_answers(sample: dict[str, Any]) -> dict[str, Any]:
    sample["distill_label"] = distill_generator(["predict: " + sample["input"]])[0][
        "generated_text"
    ]
    sample["base_label"] = base_generator(sample["input"])[0]["generated_text"]
    return sample


output_dataset = dataset["test"].map(generate_answers)
```


```
output_df = output_dataset.to_pandas().drop("llm_rationale", axis=1)
display_df = output_df.copy().rename(
    columns={
        "input": "Question",
        "label": "True answer",
        "llm_label": "PaLM answer",
        "base_label": "T5 answer",
        "distill_label": "Distilled T5 answer",
    }
)
display_df.head(10)
```


```
print(
    "The accuracy of PaLM model is {:.2f}%".format(
        output_df[output_df["label"] == output_df["llm_label"]]["label"].count()
        / len(output_df)
        * 100
    )
)
print(
    "The accuracy of raw student model is {:.2f}%".format(
        output_df[output_df["label"] == output_df["base_label"]]["label"].count()
        / len(output_df)
        * 100
    )
)
print(
    "The accuracy of distilled student model is {:.2f}%".format(
        output_df[output_df["label"] == output_df["distill_label"]]["label"].count()
        / len(output_df)
        * 100
    )
)
```

As we can see, the raw pretrained student model is unable to generate answers. However, with just a few training samples and epochs, we are able to approach the accuracy of the PaLM model using the much smaller T5 model.

# Step 5: Deploy the model to Vertex AI
*Note: The steps below will create a Cloud Storage bucket and an Artifact Registry Docker repository with the given names. If you would like to use an existing bucket or repository, provide their names below and comment out the steps to create the resources as indicated*


```
STAGING_BUCKET = ""  # @param {type:"string"}
ARTIFACTS_DIR = f"{STAGING_BUCKET}/distilled-t5"
CHECKPOINT_STEP = 4000  # @param {type:"integer"}
CHECKPOINT = f"{CKPT_DIR}/checkpoint-{CHECKPOINT_STEP}"
DOCKER_REPO_NAME = "distill-step-by-step"  # @param {type:"string"}
```

## Upload artifacts to Cloud Storage


```
! gsutil mb gs://{STAGING_BUCKET} # comment to use existing bucket
! gsutil -m cp {CHECKPOINT}/* gs://{ARTIFACTS_DIR}
```

## Create a model serving container


```
!gcloud artifacts repositories create {DOCKER_REPO_NAME} --location {REGION} --repository-format=docker  # comment to use existing bucket
!gcloud auth configure-docker {REGION}-docker.pkg.dev --quiet
!gcloud builds submit --tag {REGION}-docker.pkg.dev/{PROJECT}/{DOCKER_REPO_NAME}/distilled-flan-t5:latest ./prediction_container
```

## Upload model


```
from google.cloud import aiplatform

aiplatform.init(project=PROJECT, location=REGION, staging_bucket=STAGING_BUCKET)

DEPLOY_IMAGE = (
    f"{REGION}-docker.pkg.dev/{PROJECT}/{DOCKER_REPO_NAME}/distilled-flan-t5:latest"
)
HEALTH_ROUTE = "/health"
PREDICT_ROUTE = "/predict"
SERVING_CONTAINER_PORTS = [7080]

model = aiplatform.Model.upload(
    display_name=f"distilled-flan-t5",
    description=f"Distilled Flan T5 model using Step-By-Step Distillation",
    serving_container_image_uri=DEPLOY_IMAGE,
    serving_container_predict_route=PREDICT_ROUTE,
    serving_container_health_route=HEALTH_ROUTE,
    serving_container_ports=SERVING_CONTAINER_PORTS,
    artifact_uri=f"gs://{ARTIFACTS_DIR}",
)
print(model.resource_name)
```

## Deploy model


```
model = aiplatform.Model(model.resource_name)

endpoint = model.deploy(
    machine_type="n1-standard-4",
    traffic_split={"0": 100},
    min_replica_count=1,
    max_replica_count=1,
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    traffic_percentage=100,
    deploy_request_timeout=1200,
    sync=True,
)
endpoint.wait()
```

# Wrap up
In this notebook, we have learnt how we can use a large teacher LLM to teach a smaller student LLM to reason like it, which greatly improves the performance of smaller models over simple instruction tuning.

If you are interested in running a similar distillation pipeline on LLMs available on Google Cloud, check out [Distilling text models on Google Cloud](https://cloud.google.com/vertex-ai/docs/generative-ai/models/distill-text-models/)
