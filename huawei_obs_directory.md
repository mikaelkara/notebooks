# Huawei OBS Directory
The following code demonstrates how to load objects from the Huawei OBS (Object Storage Service) as documents.


```python
# Install the required package
# pip install esdk-obs-python
```


```python
from langchain_community.document_loaders import OBSDirectoryLoader
```


```python
endpoint = "your-endpoint"
```


```python
# Configure your access credentials\n
config = {"ak": "your-access-key", "sk": "your-secret-key"}
loader = OBSDirectoryLoader("your-bucket-name", endpoint=endpoint, config=config)
```


```python
loader.load()
```

## Specify a Prefix for Loading
If you want to load objects with a specific prefix from the bucket, you can use the following code:


```python
loader = OBSDirectoryLoader(
    "your-bucket-name", endpoint=endpoint, config=config, prefix="test_prefix"
)
```


```python
loader.load()
```

## Get Authentication Information from ECS
If your langchain is deployed on Huawei Cloud ECS and [Agency is set up](https://support.huaweicloud.com/intl/en-us/usermanual-ecs/ecs_03_0166.html#section7), the loader can directly get the security token from ECS without needing access key and secret key. 


```python
config = {"get_token_from_ecs": True}
loader = OBSDirectoryLoader("your-bucket-name", endpoint=endpoint, config=config)
```


```python
loader.load()
```

## Use a Public Bucket
If your bucket's bucket policy allows anonymous access (anonymous users have `listBucket` and `GetObject` permissions), you can directly load the objects without configuring the `config` parameter.


```python
loader = OBSDirectoryLoader("your-bucket-name", endpoint=endpoint)
```


```python
loader.load()
```
