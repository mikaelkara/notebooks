# Google Cloud Document AI


Document AI is a document understanding platform from Google Cloud to transform unstructured data from documents into structured data, making it easier to understand, analyze, and consume.

Learn more:

- [Document AI overview](https://cloud.google.com/document-ai/docs/overview)
- [Document AI videos and labs](https://cloud.google.com/document-ai/docs/videos)
- [Try it!](https://cloud.google.com/document-ai/docs/drag-and-drop)


The module contains a `PDF` parser based on DocAI from Google Cloud.

You need to install two libraries to use this parser:



```python
%pip install --upgrade --quiet  langchain-google-community[docai]
```

First, you need to set up a Google Cloud Storage (GCS) bucket and create your own Optical Character Recognition (OCR) processor as described here: https://cloud.google.com/document-ai/docs/create-processor

The `GCS_OUTPUT_PATH` should be a path to a folder on GCS (starting with `gs://`) and a `PROCESSOR_NAME` should look like `projects/PROJECT_NUMBER/locations/LOCATION/processors/PROCESSOR_ID` or `projects/PROJECT_NUMBER/locations/LOCATION/processors/PROCESSOR_ID/processorVersions/PROCESSOR_VERSION_ID`. You can get it either programmatically or copy from the `Prediction endpoint` section of the `Processor details` tab in the Google Cloud Console.



```python
GCS_OUTPUT_PATH = "gs://BUCKET_NAME/FOLDER_PATH"
PROCESSOR_NAME = "projects/PROJECT_NUMBER/locations/LOCATION/processors/PROCESSOR_ID"
```


```python
from langchain_core.document_loaders.blob_loaders import Blob
from langchain_google_community import DocAIParser
```

Now, create a `DocAIParser`.



```python
parser = DocAIParser(
    location="us", processor_name=PROCESSOR_NAME, gcs_output_path=GCS_OUTPUT_PATH
)
```

For this example, you can use an Alphabet earnings report that's uploaded to a public GCS bucket.

[2022Q1_alphabet_earnings_release.pdf](https://storage.googleapis.com/cloud-samples-data/gen-app-builder/search/alphabet-investor-pdfs/2022Q1_alphabet_earnings_release.pdf)

Pass the document to the `lazy_parse()` method to



```python
blob = Blob(
    path="gs://cloud-samples-data/gen-app-builder/search/alphabet-investor-pdfs/2022Q1_alphabet_earnings_release.pdf"
)
```

We'll get one document per page, 11 in total:



```python
docs = list(parser.lazy_parse(blob))
print(len(docs))
```

    11
    

You can run end-to-end parsing of a blob one-by-one. If you have many documents, it might be a better approach to batch them together and maybe even detach parsing from handling the results of parsing.



```python
operations = parser.docai_parse([blob])
print([op.operation.name for op in operations])
```

    ['projects/543079149601/locations/us/operations/16447136779727347991']
    

You can check whether operations are finished:



```python
parser.is_running(operations)
```




    True



And when they're finished, you can parse the results:



```python
parser.is_running(operations)
```




    False




```python
results = parser.get_results(operations)
print(results[0])
```

    DocAIParsingResults(source_path='gs://vertex-pgt/examples/goog-exhibit-99-1-q1-2023-19.pdf', parsed_path='gs://vertex-pgt/test/run1/16447136779727347991/0')
    

And now we can finally generate Documents from parsed results:



```python
docs = list(parser.parse_from_results(results))
```


```python
print(len(docs))
```

    11
    
