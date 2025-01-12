# Email

This notebook shows how to load email (`.eml`) or `Microsoft Outlook` (`.msg`) files.

Please see [this guide](/docs/integrations/providers/unstructured/) for more instructions on setting up Unstructured locally, including setting up required system dependencies.

## Using Unstructured


```python
%pip install --upgrade --quiet unstructured
```


```python
from langchain_community.document_loaders import UnstructuredEmailLoader

loader = UnstructuredEmailLoader("./example_data/fake-email.eml")

data = loader.load()

data
```




    [Document(page_content='This is a test email to use for unit tests.\n\nImportant points:\n\nRoses are red\n\nViolets are blue', metadata={'source': './example_data/fake-email.eml'})]



### Retain Elements

Under the hood, Unstructured creates different "elements" for different chunks of text. By default we combine those together, but you can easily keep that separation by specifying `mode="elements"`.


```python
loader = UnstructuredEmailLoader("example_data/fake-email.eml", mode="elements")

data = loader.load()

data[0]
```




    Document(page_content='This is a test email to use for unit tests.', metadata={'source': 'example_data/fake-email.eml', 'file_directory': 'example_data', 'filename': 'fake-email.eml', 'last_modified': '2022-12-16T17:04:16-05:00', 'sent_from': ['Matthew Robinson <mrobinson@unstructured.io>'], 'sent_to': ['Matthew Robinson <mrobinson@unstructured.io>'], 'subject': 'Test Email', 'languages': ['eng'], 'filetype': 'message/rfc822', 'category': 'NarrativeText'})



### Processing Attachments

You can process attachments with `UnstructuredEmailLoader` by setting `process_attachments=True` in the constructor. By default, attachments will be partitioned using the `partition` function from `unstructured`. You can use a different partitioning function by passing the function to the `attachment_partitioner` kwarg.


```python
loader = UnstructuredEmailLoader(
    "example_data/fake-email.eml",
    mode="elements",
    process_attachments=True,
)

data = loader.load()

data[0]
```




    Document(page_content='This is a test email to use for unit tests.', metadata={'source': 'example_data/fake-email.eml', 'file_directory': 'example_data', 'filename': 'fake-email.eml', 'last_modified': '2022-12-16T17:04:16-05:00', 'sent_from': ['Matthew Robinson <mrobinson@unstructured.io>'], 'sent_to': ['Matthew Robinson <mrobinson@unstructured.io>'], 'subject': 'Test Email', 'languages': ['eng'], 'filetype': 'message/rfc822', 'category': 'NarrativeText'})



## Using OutlookMessageLoader


```python
%pip install --upgrade --quiet extract_msg
```


```python
from langchain_community.document_loaders import OutlookMessageLoader

loader = OutlookMessageLoader("example_data/fake-email.msg")

data = loader.load()

data[0]
```




    Document(page_content='This is a test email to experiment with the MS Outlook MSG Extractor\r\n\r\n\r\n-- \r\n\r\n\r\nKind regards\r\n\r\n\r\n\r\n\r\nBrian Zhou\r\n\r\n', metadata={'source': 'example_data/fake-email.msg', 'subject': 'Test for TIF files', 'sender': 'Brian Zhou <brizhou@gmail.com>', 'date': datetime.datetime(2013, 11, 18, 0, 26, 24, tzinfo=zoneinfo.ZoneInfo(key='America/Los_Angeles'))})




```python

```
