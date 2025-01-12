# Analyzing Artistic Styles with Multimodal Embeddings

*Authored by: [Jacob Marks](https://huggingface.co/jamarks)*

![Art Analysis Cover Image](https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/art_analysis_cover_image.jpg)

Visual data like images is incredibly information-rich, but the unstructured nature of that data makes it difficult to analyze. 

In this notebook, we'll explore how to use multimodal embeddings and computed attributes to analyze artistic styles in images. We'll use the [WikiArt dataset](https://huggingface.co/datasets/huggan/wikiart) from 🤗 Hub, which we will load into FiftyOne for data analysis and visualization. We'll dive into the data in a variety of ways:

- **Image Similarity Search and Semantic Search**: We'll generate multimodal embeddings for the images in the dataset using a pre-trained [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) model from 🤗 Transformers and index the data to allow for unstructured searches.

- **Clustering and Visualization**: We'll cluster the images based on their artistic style using the embeddings and visualize the results using UMAP dimensionality reduction.

- **Uniqueness Analysis**: We'll use our embeddings to assign a uniqueness score to each image based on how similar it is to other images in the dataset.

- **Image Quality Analysis**: We'll compute image quality metrics like brightness, contrast, and saturation for each image and see how these metrics correlate with the artistic style of the images.

## Let's get started! 🚀

To run this notebook, you'll need to install the following libraries:


```python
!pip install -U transformers huggingface_hub fiftyone umap-learn
```

To make downloads lightning-fast, install [HF Transfer](https://pypi.org/project/hf-transfer/):

```bash
pip install hf-transfer
```

And enable by setting the environment variable `HF_HUB_ENABLE_HF_TRANSFER`:

```bash
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
```

<div class="alert alert-block alert-info">
<b>Note:</b> This notebook was tested with <code>transformers==4.40.0</code>, <code>huggingface_hub==0.22.2</code>, and <code>fiftyone==0.23.8</code>.
</div>

Now let's import the modules that we'll need for this notebook:


```python
import fiftyone as fo # base library and app
import fiftyone.zoo as foz # zoo datasets and models
import fiftyone.brain as fob # ML routines
from fiftyone import ViewField as F # for defining custom views
import fiftyone.utils.huggingface as fouh # for loading datasets from Hugging Face
```

We'll start by loading the WikiArt dataset from 🤗 Hub into FiftyOne. This dataset can also be loaded through Hugging Face's `datasets` library, but we'll use [FiftyOne's 🤗 Hub integration](https://docs.voxel51.com/integrations/huggingface.html#huggingface-hub) to get the data directly from the Datasets server. To make the computations fast, we'll just download the first $1,000$ samples.


```python
dataset = fouh.load_from_hub(
    "huggan/wikiart", ## repo_id
    format="parquet", ## for Parquet format
    classification_fields=["artist", "style", "genre"], # columns to store as classification fields
    max_samples=1000, # number of samples to load
    name="wikiart", # name of the dataset in FiftyOne
)
```

Print out a summary of the dataset to see what it contains:


```python
print(dataset)
```

    Name:        wikiart
    Media type:  image
    Num samples: 1000
    Persistent:  False
    Tags:        []
    Sample fields:
        id:       fiftyone.core.fields.ObjectIdField
        filepath: fiftyone.core.fields.StringField
        tags:     fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
        metadata: fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.ImageMetadata)
        artist:   fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)
        style:    fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)
        genre:    fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)
        row_idx:  fiftyone.core.fields.IntField
    

Visualize the dataset in the [FiftyOne App](https://docs.voxel51.com/user_guide/app.html):


```python
session = fo.launch_app(dataset)
```

![WikiArt Dataset](https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/art_analysis_wikiart_dataset.jpg)

Let's list out the names of the artists whose styles we'll be analyzing:


```python
artists = dataset.distinct("artist.label")
print(artists)
```

    ['Unknown Artist', 'albrecht-durer', 'boris-kustodiev', 'camille-pissarro', 'childe-hassam', 'claude-monet', 'edgar-degas', 'eugene-boudin', 'gustave-dore', 'ilya-repin', 'ivan-aivazovsky', 'ivan-shishkin', 'john-singer-sargent', 'marc-chagall', 'martiros-saryan', 'nicholas-roerich', 'pablo-picasso', 'paul-cezanne', 'pierre-auguste-renoir', 'pyotr-konchalovsky', 'raphael-kirchner', 'rembrandt', 'salvador-dali', 'vincent-van-gogh']
    

## Finding Similar Artwork

When you find a piece of art that you like, it's natural to want to find similar pieces. We can do this with vector embeddings! What's more, by using multimodal embeddings, we will unlock the ability to find paintings that closely resemble a given text query, which could be a description of a painting or even a poem.

Let's generate multimodal embeddings for the images using a pre-trained CLIP Vision Transformer (ViT) model from 🤗 Transformers. Running `compute_similarity()` from the [FiftyOne Brain](https://docs.voxel51.com/user_guide/brain.html) will compute these embeddings and use them to generate a similarity index on the dataset.


```python
fob.compute_similarity(
    dataset, 
    model="zero-shot-classification-transformer-torch", ## type of model to load from model zoo
    name_or_path="openai/clip-vit-base-patch32", ## repo_id of checkpoint
    embeddings="clip_embeddings", ## name of the field to store embeddings
    brain_key="clip_sim", ## key to store similarity index info
    batch_size=32, ## batch size for inference
    )
```

    Computing embeddings...
     100% |███████████████| 1000/1000 [5.0m elapsed, 0s remaining, 3.3 samples/s]    
    




    <fiftyone.brain.internal.core.sklearn.SklearnSimilarityIndex at 0x2ad67ecd0>



<div style="padding: 10px; border-left: 5px solid #0078d4; font-family: Arial, sans-serif; margin: 10px 0;">

Alternatively, you could load the model directly from the 🤗 Transformers library and pass the model in directly:

```python
from transformers import CLIPModel
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
fob.compute_similarity(
    dataset, 
    model=model,
    embeddings="clip_embeddings", ## name of the field to store embeddings
    brain_key="clip_sim" ## key to store similarity index info
)
```

For a comprehensive guide to this and more, check out <a href="https://docs.voxel51.com/integrations/huggingface.html#transformers-library">FiftyOne's 🤗 Transformers integration</a>.
</div>


Refresh the FiftyOne App, select the checkbox for an image in the sample grid, and click the photo icon to see the most similar images in the dataset. On the backend, clicking this button triggers a query to the similarity index to find the most similar images to the selected image, based on the pre-computed embeddings, and displays them in the App.

![Image Similarity Search](https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/art_analysis_image_search.gif)

We can use this to see what art pieces are most similar to a given art piece. This can be useful for finding similar art pieces (to recommend to users or add to a collection) or getting inspiration for a new piece.

But there's more! Because CLIP is multimodal, we can also use it to perform semantic searches. This means we can search for images based on text queries. For example, we can search for "pastel trees" and see all the images in the dataset that are similar to that query. To do this, click on the search icon in the FiftyOne App and enter a text query:

![Semantic Search](https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/art_analysis_semantic_search.gif)

Behind the scenes, the text is tokenized, embedded with CLIP's text encoder, and then used to query the similarity index to find the most similar images in the dataset. This is a powerful way to search for images based on text queries and can be useful for finding images that match a particular theme or style. And this is not limited to CLIP; you can use any CLIP-like model from 🤗 Transformers that can generate embeddings for images and text!

<div class="alert alert-block alert-info">
💡 For efficient vector search and indexing over large datasets, FiftyOne has native <a href="https://voxel51.com/vector-search">integrations with open source vector databases</a>.
</div>


## Uncovering Artistic Motifs with Clustering and Visualization

By performing similarity and semantic searches, we can begin to interact with the data more effectively. But we can also take this a step further and add some unsupervised learning into the mix. This will help us identify artistic patterns in the WikiArt dataset, from stylistic, to topical, and even motifs that are hard to put into words. 

We will do this in two ways:

1. **Dimensionality Reduction**: We'll use UMAP to reduce the dimensionality of the embeddings to 2D and visualize the data in a scatter plot. This will allow us to see how the images cluster based on their style, genre, and artist.
2. **Clustering**: We'll use K-Means clustering to cluster the images based on their embeddings and see what groups emerge.

For dimensionality reduction, we will run `compute_visualization()` from the FiftyOne Brain, passing in the previously computed embeddings. We specify `method="umap"` to use UMAP for dimensionality reduction, but we could also use PCA or t-SNE:


```python
fob.compute_visualization(dataset, embeddings="clip_embeddings", method="umap", brain_key="clip_vis")
```

    Generating visualization...
    

    /opt/homebrew/Caskroom/miniforge/base/envs/fdev/lib/python3.9/site-packages/numba/cpython/hashing.py:482: UserWarning: FNV hashing is not implemented in Numba. See PEP 456 https://www.python.org/dev/peps/pep-0456/ for rationale over not using FNV. Numba will continue to work, but hashes for built in types will be computed using siphash24. This will permit e.g. dictionaries to continue to behave as expected, however anything relying on the value of the hash opposed to hash as a derived property is likely to not work as expected.
      warnings.warn(msg)
    

    UMAP( verbose=True)
    Tue Apr 30 11:51:45 2024 Construct fuzzy simplicial set
    Tue Apr 30 11:51:46 2024 Finding Nearest Neighbors
    Tue Apr 30 11:51:47 2024 Finished Nearest Neighbor Search
    Tue Apr 30 11:51:48 2024 Construct embedding
    


    Epochs completed:   0%|            0/500 [00:00]


    	completed  0  /  500 epochs
    	completed  50  /  500 epochs
    	completed  100  /  500 epochs
    	completed  150  /  500 epochs
    	completed  200  /  500 epochs
    	completed  250  /  500 epochs
    	completed  300  /  500 epochs
    	completed  350  /  500 epochs
    	completed  400  /  500 epochs
    	completed  450  /  500 epochs
    Tue Apr 30 11:51:49 2024 Finished embedding
    




    <fiftyone.brain.visualization.VisualizationResults at 0x29f468760>



Now we can open a panel in the FiftyOne App, where we will see one 2D point for each image in the dataset. We can color the points by any field in the dataset, such as the artist or genre, to see how strongly these attributes are captured by our image features:

![UMAP Visualization](https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/art_analysis_visualize_embeddings.gif)

We can also run clustering on the embeddings to group similar images together — perhaps the dominant features of these works of art are not captured by the existing labels, or maybe there are distinct sub-genres that we want to identify. To cluster our data, we will need to download the [FiftyOne Clustering Plugin](https://github.com/jacobmarks/clustering-plugin):


```python
!fiftyone plugins download https://github.com/jacobmarks/clustering-plugin
```

Refreshing the app again, we can then access the clustering functionality via an operator in the app. Hit the backtick key to open the operator list, type "cluster" and select the operator from the dropdown. This will open an interactive panel where we can specify the clustering algorithm, hyperparameters, and the field to cluster on. To keep it simple, we'll use K-Means clustering with $10$ clusters.

We can then visualize the clusters in the app and see how the images group together based on their embeddings:

![K-means Clustering](https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/art_analysis_clustering.gif)

We can see that some of the clusters select for artist; others select for genre or style. Others are more abstract and may represent sub-genres or other groupings that are not immediately obvious from the data.

## Identifying the Most Unique Works of Art

One interesting question we can ask about our dataset is how *unique* each image is. This question is important for many applications, such as recommending similar images, detecting duplicates, or identifying outliers. In the context of art, how unique a painting is could be an important factor in determining its value.

While there are a million ways to characterize uniqueness, our image embeddings allow us to quantitatively assign each sample a uniqueness score based on how similar it is to other samples in the dataset. Explicitly, the FiftyOne Brain's `compute_uniqueness()` function looks at the distance between each sample's embedding and its nearest neighbors, and computes a score between $0$ and $1$ based on this distance. A score of $0$ means the sample is nondescript or very similar to others, while a score of $1$ means the sample is very unique.


```python
fob.compute_uniqueness(dataset, embeddings="clip_embeddings") # compute uniqueness using CLIP embeddings
```

    Computing uniqueness...
    Uniqueness computation complete
    

We can then color by this in the embeddings panel, filter by uniqueness score, or even sort by it to see the most unique images in the dataset:


```python
most_unique_view = dataset.sort_by("uniqueness", reverse=True)
session.view = most_unique_view.view() # Most unique images
```

![Most Unique Images](https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/art_analysis_most_unique.jpg)


```python
least_unique_view = dataset.sort_by("uniqueness", reverse=False)
session.view = least_unique_view.view() # Least unique images
```

![Least Unique Images](https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/art_analysis_least_unique.jpg)

Going a step further, we can also answer the question of which artist tends to produce the most unique works. We can compute the average uniqueness score for each artist across all of their works of art:


```python
artist_unique_scores = {
    artist: dataset.match(F("artist.label") == artist).mean("uniqueness")
    for artist in artists
}

sorted_artists = sorted(
    artist_unique_scores, key=artist_unique_scores.get, reverse=True
)

for artist in sorted_artists:
    print(f"{artist}: {artist_unique_scores[artist]}")
```

    Unknown Artist: 0.7932221632002723
    boris-kustodiev: 0.7480731948424676
    salvador-dali: 0.7368807620414014
    raphael-kirchner: 0.7315448102204755
    ilya-repin: 0.7204744626806383
    marc-chagall: 0.7169373812321908
    rembrandt: 0.715205220292227
    martiros-saryan: 0.708560775790436
    childe-hassam: 0.7018343391132756
    edgar-degas: 0.699912746806587
    albrecht-durer: 0.6969358680800216
    john-singer-sargent: 0.6839955708720844
    pablo-picasso: 0.6835137858302969
    pyotr-konchalovsky: 0.6780653000855895
    nicholas-roerich: 0.6676504687452387
    ivan-aivazovsky: 0.6484361530090199
    vincent-van-gogh: 0.6472004520699081
    gustave-dore: 0.6307283287457358
    pierre-auguste-renoir: 0.6271467146993583
    paul-cezanne: 0.6251076007168186
    eugene-boudin: 0.6103397516167454
    camille-pissarro: 0.6046182609119615
    claude-monet: 0.5998234558947573
    ivan-shishkin: 0.589796389836674
    

It would seem that the artist with the most unique works in our dataset is Boris Kustodiev! Let's take a look at some of his works:


```python
kustodiev_view = dataset.match(F("artist.label") == "boris-kustodiev")
session.view = kustodiev_view.view()
```

![Boris Kustodiev Artwork](https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/art_analysis_kustodiev_view.jpg)

## Characterizing Art with Visual Qualities

To round things out, let's go back to the basics and analyze some core qualities of the images in our dataset. We'll compute standard metrics like brightness, contrast, and saturation for each image and see how these metrics correlate with the artistic style and genre of the art pieces.

To run these analyses, we will need to download the [FiftyOne Image Quality Plugin](https://github.com/jacobmarks/image-quality-issues):


```python
!fiftyone plugins download https://github.com/jacobmarks/image-quality-issues/
```

Refresh the app and open the operators list again. This time type `compute` and select one of the image quality operators. We'll start with brightness:

![Compute Brightness](https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/art_analysis_compute_brightness.gif)

When the operator finishes running, we will have a new field in our dataset that contains the brightness score for each image. We can then visualize this data in the app:

![Brightness](https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/art_analysis_brightness.gif)

We can also color by brightness, and even see how it correlates with other fields in the dataset like style:

![Style by Brightness](https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/art_analysis_style_by_brightness.gif)

Now do the same for contrast and saturation. Here are the results for saturation:

![Filter by Saturation](https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/art_analysis_filter_by_saturation.jpg)

Hopefully this illustrates how not everything boils down to applying deep neural networks to your data. Sometimes, simple metrics can be just as informative and can provide a different perspective on your data 🤓!

<div class="alert alert-block alert-info">
📚 For larger datasets, you may want to <a href="https://docs.voxel51.com/plugins/using_plugins.html#delegated-operations">delegate the operations</a> for later execution.
</div>

## What's Next?

In this notebook, we've explored how to use multimodal embeddings, unsupervised learning, and traditional image processing techniques to analyze artistic styles in images. We've seen how to perform image similarity and semantic searches, cluster images based on their style, analyze the uniqueness of images, and compute image quality metrics. These techniques can be applied to a wide range of visual datasets, from art collections to medical images to satellite imagery. Try [loading a different dataset from the Hugging Face Hub](https://docs.voxel51.com/integrations/huggingface.html#loading-datasets-from-the-hub) and see what insights you can uncover!

If you want to go even further, here are some additional analyses you could try:

- **Zero-Shot Classification**: Use a pre-trained vision-language model from 🤗 Transformers to categorize images in the dataset by topic or subject, without any training data. Check out this [Zero-Shot Classification tutorial](https://docs.voxel51.com/tutorials/zero_shot_classification.html) for more info.
- **Image Captioning**: Use a pre-trained vision-language model from 🤗 Transformers to generate captions for the images in the dataset. Then use this for topic modeling or cluster artwork based on embeddings for these captions. Check out FiftyOne's [Image Captioning Plugin](https://github.com/jacobmarks/fiftyone-image-captioning-plugin) for more info.

### 📚 Resources

- [FiftyOne 🤝 🤗 Hub Integration](https://docs.voxel51.com/integrations/huggingface.html#huggingface-hub)
- [FiftyOne 🤝 🤗 Transformers Integration](https://docs.voxel51.com/integrations/huggingface.html#transformers-library)
- [FiftyOne Vector Search Integrations](https://voxel51.com/vector-search/)
- [Visualizing Data with Dimensionality Reduction Techniques](https://docs.voxel51.com/tutorials/dimension_reduction.html)
- [Clustering Images with Embeddings](https://docs.voxel51.com/tutorials/clustering.html)
- [Exploring Image Uniqueness with FiftyOne](https://docs.voxel51.com/tutorials/uniqueness.html)

## FiftyOne Open Source Project

[FiftyOne](https://github.com/voxel51/fiftyone/) is the leading open source toolkit for building high-quality datasets and computer vision models. With over 2M downloads, FiftyOne is trusted by developers and researchers across the globe.

💪 The FiftyOne team welcomes contributions from the open source community! If you're interested in contributing to FiftyOne, check out the [contributing guide](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md).
