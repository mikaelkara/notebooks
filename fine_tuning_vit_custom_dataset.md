# Fine-tuning a Vision Transformer Model With a Custom Biomedical Dataset
_Authored by: [Emre Albayrak](https://github.com/emre570)_

This guide outlines the process for fine-tuning a Vision Transformer (ViT) model on a custom biomedical dataset. It includes steps for loading and preparing the dataset, setting up image transformations for different data splits, configuring and initializing the ViT model, and defining the training process with evaluation and visualization tools.

## Dataset Info
The custom dataset is hand-made, containing 780 images with 3 classes (benign, malignant, normal). 

![attachment:datasetinfo.png](https://huggingface.co/datasets/huggingface/cookbook-images/resolve/102d6c23e6cc24db857fbc60186461ded6cdfb75/datasetinfo.png)

## Model Info
The model we fine-tune will be Google's [`"vit-large-patch16-224"`](https://huggingface.co/google/vit-large-patch16-224). It is trained on ImageNet-21k (14M images, 21.843 classes), and fine-tuned on ImageNet 2012 (1M images, 1.000 classes) at resolution 224x224. Google has several other ViT models with different image sizes and patches.

Let's get started.

## Getting Started
First, let's install libraries first.


```python
!pip install datasets transformers accelerate torch scikit-learn matplotlib wandb
```

(Optional) We will push our model to Hugging Face Hub so we must login.


```python
#from huggingface_hub import notebook_login
#notebook_login()
```


    VBox(children=(HTML(value='<center> <img\nsrc=https://huggingface.co/front/assets/huggingface_logo-noborder.sv…


## Dataset Preparation
Datasets library automatically pulls images and classes from the dataset. For detailed info, you can visit [`this link`](https://huggingface.co/docs/datasets/image_load).


```python
from datasets import load_dataset

dataset = load_dataset("emre570/breastcancer-ultrasound-images")
dataset
```




    DatasetDict({
        train: Dataset({
            features: ['image', 'label'],
            num_rows: 624
        })
        test: Dataset({
            features: ['image', 'label'],
            num_rows: 156
        })
    })



We got our dataset. But we don't have a validation set. To create the validation set, we will calculate the size of the validation set as a fraction of the training set based on the size of the test set. Then we split the training dataset into new training and validation subsets.


```python
# Get the numbers of each set
test_num = len(dataset["test"])
train_num = len(dataset["train"])

val_size = test_num / train_num

train_val_split = dataset["train"].train_test_split(test_size=val_size)
train_val_split
```




    DatasetDict({
        train: Dataset({
            features: ['image', 'label'],
            num_rows: 468
        })
        test: Dataset({
            features: ['image', 'label'],
            num_rows: 156
        })
    })



We got our seperated train set. Let's merge them with test set.


```python
from datasets import DatasetDict

dataset = DatasetDict({
    "train": train_val_split["train"],
    "validation": train_val_split["test"],
    "test": dataset["test"]
})
dataset
```




    DatasetDict({
        train: Dataset({
            features: ['image', 'label'],
            num_rows: 468
        })
        validation: Dataset({
            features: ['image', 'label'],
            num_rows: 156
        })
        test: Dataset({
            features: ['image', 'label'],
            num_rows: 156
        })
    })



Perfect! Our dataset is ready. Let's assign subsets to different variables. We will use them later for easy reference.


```python
train_ds = dataset['train']
val_ds = dataset['validation']
test_ds = dataset['test']
```

We can see the image is a PIL.Image with a label associated with it.


```python
train_ds[0]
```




    {'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=460x391>,
     'label': 0}



We can also see the features of train set.


```python
train_ds.features
```




    {'image': Image(mode=None, decode=True, id=None),
     'label': ClassLabel(names=['benign', 'malignant', 'normal'], id=None)}



Let's show one image from each class from dataset.


```python
import matplotlib.pyplot as plt

# Initialize a set to keep track of shown labels
shown_labels = set()

# Initialize the figure for plotting
plt.figure(figsize=(10, 10))

# Loop through the dataset and plot the first image of each label
for i, sample in enumerate(train_ds):
    label = train_ds.features['label'].names[sample['label']]
    if label not in shown_labels:
        plt.subplot(1, len(train_ds.features['label'].names), len(shown_labels) + 1)
        plt.imshow(sample['image'])
        plt.title(label)
        plt.axis('off')
        shown_labels.add(label)
        if len(shown_labels) == len(train_ds.features['label'].names):
            break

plt.show()
```


    
![png](output_18_0.png)
    


## Data Processing
The dataset is ready. But we are not ready for fine-tuning. We will follow this procedures respectively:

- **Label Mapping:** We convert between label IDs and their corresponding names, useful for model training and evaluation.

- **Image Processing:** Then, we utilize the ViTImageProcessor to standardize input image sizes and applies normalization specific to the pretrained model. Also, will define different transformations for training, validation, and testing to improve model generalization using torchvision.

- **Transform Functions:** Implement functions to apply the transformations to the dataset, converting images to the required format and dimensions for the ViT model.

- **Data Loading:** Set up a custom collate function to properly batch images and labels, and create a DataLoader for efficient loading and batching during model training.

- **Batch Preparation:** Retrieve and display the shape of data in a sample batch to verify correct processing and readiness for model input.

### Label Mapping


```python
id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
label2id = {label:id for id,label in id2label.items()}
id2label, id2label[train_ds[0]['label']]
```




    ({0: 'benign', 1: 'malignant', 2: 'normal'}, 'benign')



### Image Processing


```python
from transformers import ViTImageProcessor

model_name = "google/vit-large-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
```


```python
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Resize

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = Normalize(mean=image_mean, std=image_std)

train_transforms = Compose([        
    RandomResizedCrop(size),
    RandomHorizontalFlip(),
    ToTensor(),
    normalize,
])
val_transforms = Compose([
    Resize(size),
    CenterCrop(size),
    ToTensor(),
    normalize,
])
test_transforms = Compose([
    Resize(size),
    CenterCrop(size),
    ToTensor(),
    normalize,
])
```

### Create transform functions


```python
def apply_train_transforms(examples):
    examples['pixel_values'] = [train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def apply_val_transforms(examples):
    examples['pixel_values'] = [val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def apply_test_transforms(examples):
    examples['pixel_values'] = [val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples
```

### Apply transform functions to each set


```python
train_ds.set_transform(apply_train_transforms)
val_ds.set_transform(apply_val_transforms)
test_ds.set_transform(apply_test_transforms)
```


```python
train_ds.features
```




    {'image': Image(mode=None, decode=True, id=None),
     'label': ClassLabel(names=['benign', 'malignant', 'normal'], id=None)}




```python
train_ds[0]
```




    {'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=460x391>,
     'label': 0,
     'pixel_values': tensor([[[-0.2000, -0.1765, -0.1529,  ..., -0.3098, -0.3490, -0.3412],
              [-0.2471, -0.2392, -0.2471,  ..., -0.2392, -0.2235, -0.2000],
              [-0.3255, -0.3412, -0.3647,  ..., -0.1765, -0.1608, -0.1529],
              ...,
              [-0.7333, -0.7412, -0.7647,  ..., -0.7490, -0.7647, -0.7725],
              [-0.7255, -0.7176, -0.7333,  ..., -0.7882, -0.7804, -0.7882],
              [-0.7412, -0.7333, -0.7412,  ..., -0.7804, -0.7725, -0.7804]],
     
             [[-0.2000, -0.1765, -0.1529,  ..., -0.3098, -0.3490, -0.3412],
              [-0.2471, -0.2392, -0.2471,  ..., -0.2392, -0.2235, -0.2000],
              [-0.3255, -0.3412, -0.3647,  ..., -0.1765, -0.1608, -0.1529],
              ...,
              [-0.7333, -0.7412, -0.7647,  ..., -0.7490, -0.7647, -0.7725],
              [-0.7255, -0.7176, -0.7333,  ..., -0.7882, -0.7804, -0.7882],
              [-0.7412, -0.7333, -0.7412,  ..., -0.7804, -0.7725, -0.7804]],
     
             [[-0.2000, -0.1765, -0.1529,  ..., -0.3098, -0.3490, -0.3412],
              [-0.2471, -0.2392, -0.2471,  ..., -0.2392, -0.2235, -0.2000],
              [-0.3255, -0.3412, -0.3647,  ..., -0.1765, -0.1608, -0.1529],
              ...,
              [-0.7333, -0.7412, -0.7647,  ..., -0.7490, -0.7647, -0.7725],
              [-0.7255, -0.7176, -0.7333,  ..., -0.7882, -0.7804, -0.7882],
              [-0.7412, -0.7333, -0.7412,  ..., -0.7804, -0.7725, -0.7804]]])}



Looks like we converted our pixel values into tensors. 

### Data Loading


```python
import torch
from torch.utils.data import DataLoader

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)
```

### Batch Preparation


```python
batch = next(iter(train_dl))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k, v.shape)
```

    pixel_values torch.Size([4, 3, 224, 224])
    labels torch.Size([4])
    

Perfect! Now we are ready for fine-tuning process.

## Fine-tuning the Model
Now we will configure and fine-tune the model. We started by initializing the model with specific label mappings and pre-trained settings, adjusting for size mismatches. Training parameters are set up to define the model's learning process, including the save strategy, batch sizes, and training epochs, with results logged via Weights & Biases. Hugging Face Trainer will then instantiate to manage the training and evaluation, utilizing a custom data collator and the model's built-in processor. Finally, after training, the model's performance is evaluated on a test dataset, with metrics printed to assess its accuracy.

First, we call our model.


```python
from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(model_name, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
```

    Some weights of ViTForImageClassification were not initialized from the model checkpoint at google/vit-large-patch16-224 and are newly initialized because the shapes did not match:
    - classifier.weight: found shape torch.Size([1000, 1024]) in the checkpoint and torch.Size([3, 1024]) in the model instantiated
    - classifier.bias: found shape torch.Size([1000]) in the checkpoint and torch.Size([3]) in the model instantiated
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

There is a subtle detail in here. The `ignore_mismatched_sizes` parameter.

When you fine-tune a pre-trained model on a new dataset, sometimes the input size of your images or the model architecture specifics (like the number of labels in the classification layer) might not match exactly with what the model was originally trained on. This can happen for various reasons, such as when using a model trained on one type of image data (like natural images from ImageNet) on a completely different type of image data (like medical images or specialized camera images).

Setting `ignore_mismatched_sizes` to `True` allows the model to adjust its layers to accommodate size differences without throwing an error. 

For example, the number of classes this model is trained on is 1000, which is `torch.Size([1000])` and it expects an input with `torch.Size([1000])` classes. Our dataset has 3, which is `torch.Size([3])` classes. If we give it directly, it will raise an error because the class numbers do not match.

Then, define training arguments from Google for this model.

(Optional) Note that the metrics will be saved in Weights & Biases because we set the `report_to` parameter to `wandb`. W&B will ask you for an API key, so you should create an account and an API key. If you don't want, you can remove `report_to` parameter.


```python
from transformers import TrainingArguments, Trainer
import numpy as np

train_args = TrainingArguments(
    output_dir = "output-models",
    save_total_limit=2,
    report_to="wandb",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=40,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir='logs',
    remove_unused_columns=False,
)
```

We can now begin the fine-tuning process with `Trainer`. 


```python
trainer = Trainer(
    model,
    train_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    tokenizer=processor,
)
trainer.train()
```



    <div>

      <progress value='1880' max='1880' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [1880/1880 16:42, Epoch 40/40]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>0.428426</td>
    </tr>
    <tr>
      <td>2</td>
      <td>No log</td>
      <td>0.394955</td>
    </tr>
    <tr>
      <td>3</td>
      <td>No log</td>
      <td>0.370801</td>
    </tr>
    <tr>
      <td>4</td>
      <td>No log</td>
      <td>0.364052</td>
    </tr>
    <tr>
      <td>5</td>
      <td>No log</td>
      <td>0.427605</td>
    </tr>
    <tr>
      <td>6</td>
      <td>No log</td>
      <td>0.441180</td>
    </tr>
    <tr>
      <td>7</td>
      <td>No log</td>
      <td>0.377579</td>
    </tr>
    <tr>
      <td>8</td>
      <td>No log</td>
      <td>0.387463</td>
    </tr>
    <tr>
      <td>9</td>
      <td>No log</td>
      <td>0.380499</td>
    </tr>
    <tr>
      <td>10</td>
      <td>No log</td>
      <td>0.346761</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.390300</td>
      <td>0.469292</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.390300</td>
      <td>0.389932</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.390300</td>
      <td>0.435536</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.390300</td>
      <td>0.296190</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.390300</td>
      <td>0.436435</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.390300</td>
      <td>0.446079</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.390300</td>
      <td>0.577235</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.390300</td>
      <td>0.401280</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.390300</td>
      <td>0.501154</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.390300</td>
      <td>0.490980</td>
    </tr>
    <tr>
      <td>21</td>
      <td>0.390300</td>
      <td>0.458035</td>
    </tr>
    <tr>
      <td>22</td>
      <td>0.238200</td>
      <td>0.426354</td>
    </tr>
    <tr>
      <td>23</td>
      <td>0.238200</td>
      <td>0.411909</td>
    </tr>
    <tr>
      <td>24</td>
      <td>0.238200</td>
      <td>0.435578</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0.238200</td>
      <td>0.430924</td>
    </tr>
    <tr>
      <td>26</td>
      <td>0.238200</td>
      <td>0.498050</td>
    </tr>
    <tr>
      <td>27</td>
      <td>0.238200</td>
      <td>0.501461</td>
    </tr>
    <tr>
      <td>28</td>
      <td>0.238200</td>
      <td>0.559837</td>
    </tr>
    <tr>
      <td>29</td>
      <td>0.238200</td>
      <td>0.420119</td>
    </tr>
    <tr>
      <td>30</td>
      <td>0.238200</td>
      <td>0.416809</td>
    </tr>
    <tr>
      <td>31</td>
      <td>0.238200</td>
      <td>0.635555</td>
    </tr>
    <tr>
      <td>32</td>
      <td>0.163100</td>
      <td>0.421264</td>
    </tr>
    <tr>
      <td>33</td>
      <td>0.163100</td>
      <td>0.445050</td>
    </tr>
    <tr>
      <td>34</td>
      <td>0.163100</td>
      <td>0.453854</td>
    </tr>
    <tr>
      <td>35</td>
      <td>0.163100</td>
      <td>0.442983</td>
    </tr>
    <tr>
      <td>36</td>
      <td>0.163100</td>
      <td>0.432370</td>
    </tr>
    <tr>
      <td>37</td>
      <td>0.163100</td>
      <td>0.442086</td>
    </tr>
    <tr>
      <td>38</td>
      <td>0.163100</td>
      <td>0.478380</td>
    </tr>
    <tr>
      <td>39</td>
      <td>0.163100</td>
      <td>0.477927</td>
    </tr>
    <tr>
      <td>40</td>
      <td>0.163100</td>
      <td>0.479882</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=1880, training_loss=0.23721330723863968, metrics={'train_runtime': 1003.2398, 'train_samples_per_second': 18.66, 'train_steps_per_second': 1.874, 'total_flos': 5.128065177052447e+18, 'train_loss': 0.23721330723863968, 'epoch': 40.0})



| Epoch | Training Loss | Validation Loss | Accuracy |
|-------|---------------|-----------------|----------|
| 40    | 0.174700      | 0.596288        | 0.903846 |

The fine-tuning process is done. Let's continue with evaluating the model to test set.


```python
outputs = trainer.predict(test_ds)
print(outputs.metrics)
```





    {'test_loss': 0.40843912959098816, 'test_runtime': 4.9934, 'test_samples_per_second': 31.242, 'test_steps_per_second': 7.81}
    

`{'test_loss': 0.3219967782497406, 'test_accuracy': 0.9102564102564102, 'test_runtime': 4.0543, 'test_samples_per_second': 38.478, 'test_steps_per_second': 9.619}`

### (Optional) Push Model to Hub
We can push our model to Hugging Face Hub using `push_to_hub`


```python
model.push_to_hub("your_model_name")
```

That's great! Let's visualize the results.

## Results
We made the fine-tuning. Let's see how our model predicted the classes using scikit-learn's Confusion Matrix Display and show Recall Score.

### What is Confusion Matrix?
A confusion matrix is a specific table layout that allows visualization of the performance of an algorithm, typically a supervised learning model, on a set of test data for which the true values are known. It's especially useful for checking how well a classification model is performing because it shows the frequency of true versus predicted labels.

Let's draw our model's Confusion Matrix


```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

labels = train_ds.features['label'].names
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7f44b1f289a0>




    
![png](output_51_1.png)
    


### What is Recall Score?
The recall score is a performance metric used in classification tasks to measure the ability of a model to correctly identify all relevant instances within a dataset. Specifically, recall assesses the proportion of actual positives that are correctly predicted as such by the model.

Let's print recall scores using scikit-learn


```python
from sklearn.metrics import recall_score

# Calculate the recall scores
# 'None' calculates recall for each class separately
recall = recall_score(y_true, y_pred, average=None)

# Print the recall for each class
for label, score in zip(labels, recall):
    print(f'Recall for {label}: {score:.2f}')

```

    Recall for benign: 0.90
    Recall for malignant: 0.86
    Recall for normal: 0.78
    

`Recall for benign: 0.90,
Recall for malignant: 0.86,
Recall for normal: 0.78`

## Conclusion
In this cookbook, we covered how to train a ViT model with a medical dataset. It covers crucial steps such as dataset preparation, image preprocessing, model configuration, training, evaluation, and result visualization. By leveraging Hugging Face's Transformers library scikit-learn and PyTorch Torchvision, it facilitates efficient model training and evaluation, providing valuable insights into the model's performance and its ability to classify biomedical images accurately.
