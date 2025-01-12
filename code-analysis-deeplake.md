# Use LangChain, GPT and Activeloop's Deep Lake to work with code base
In this tutorial, we are going to use Langchain + Activeloop's Deep Lake with GPT to analyze the code base of the LangChain itself. 

## Design

1. Prepare data:
   1. Upload all python project files using the `langchain_community.document_loaders.TextLoader`. We will call these files the **documents**.
   2. Split all documents to chunks using the `langchain_text_splitters.CharacterTextSplitter`.
   3. Embed chunks and upload them into the DeepLake using `langchain.embeddings.openai.OpenAIEmbeddings` and `langchain_community.vectorstores.DeepLake`
2. Question-Answering:
   1. Build a chain from `langchain.chat_models.ChatOpenAI` and `langchain.chains.ConversationalRetrievalChain`
   2. Prepare questions.
   3. Get answers running the chain.


## Implementation

### Integration preparations

We need to set up keys for external services and install necessary python libraries.


```python
#!python3 -m pip install --upgrade langchain deeplake openai
```

Set up OpenAI embeddings, Deep Lake multi-modal vector store api and authenticate. 

For full documentation of Deep Lake please follow https://docs.activeloop.ai/ and API reference https://docs.deeplake.ai/en/latest/


```python
import os
from getpass import getpass

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass()
# Please manually enter OpenAI Key
```

Authenticate into Deep Lake if you want to create your own dataset and publish it. You can get an API key from the platform at [app.activeloop.ai](https://app.activeloop.ai)


```python
activeloop_token = getpass("Activeloop Token:")
os.environ["ACTIVELOOP_TOKEN"] = activeloop_token
```

### Prepare data 

Load all repository files. Here we assume this notebook is downloaded as the part of the langchain fork and we work with the python files of the `langchain` repo.

If you want to use files from different repo, change `root_dir` to the root dir of your repo.


```python
!ls "../../../../../../libs"
```

    CITATION.cff  MIGRATE.md  README.md  libs	  poetry.toml
    LICENSE       Makefile	  docs	     poetry.lock  pyproject.toml
    


```python
from langchain_community.document_loaders import TextLoader

root_dir = "../../../../../../libs"

docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith(".py") and "*venv/" not in dirpath:
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception:
                pass
print(f"{len(docs)}")
```

    2554
    

Then, chunk the files


```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
print(f"{len(texts)}")
```

    Created a chunk of size 1010, which is longer than the specified 1000
    Created a chunk of size 3466, which is longer than the specified 1000
    Created a chunk of size 1375, which is longer than the specified 1000
    Created a chunk of size 1928, which is longer than the specified 1000
    Created a chunk of size 1075, which is longer than the specified 1000
    Created a chunk of size 1063, which is longer than the specified 1000
    Created a chunk of size 1083, which is longer than the specified 1000
    Created a chunk of size 1074, which is longer than the specified 1000
    Created a chunk of size 1591, which is longer than the specified 1000
    Created a chunk of size 2300, which is longer than the specified 1000
    Created a chunk of size 1040, which is longer than the specified 1000
    Created a chunk of size 1018, which is longer than the specified 1000
    Created a chunk of size 2787, which is longer than the specified 1000
    Created a chunk of size 1018, which is longer than the specified 1000
    Created a chunk of size 2311, which is longer than the specified 1000
    Created a chunk of size 2811, which is longer than the specified 1000
    Created a chunk of size 1186, which is longer than the specified 1000
    Created a chunk of size 1497, which is longer than the specified 1000
    Created a chunk of size 1043, which is longer than the specified 1000
    Created a chunk of size 1020, which is longer than the specified 1000
    Created a chunk of size 1232, which is longer than the specified 1000
    Created a chunk of size 1334, which is longer than the specified 1000
    Created a chunk of size 1221, which is longer than the specified 1000
    Created a chunk of size 2229, which is longer than the specified 1000
    Created a chunk of size 1027, which is longer than the specified 1000
    Created a chunk of size 1361, which is longer than the specified 1000
    Created a chunk of size 1057, which is longer than the specified 1000
    Created a chunk of size 1204, which is longer than the specified 1000
    Created a chunk of size 1420, which is longer than the specified 1000
    Created a chunk of size 1298, which is longer than the specified 1000
    Created a chunk of size 1062, which is longer than the specified 1000
    Created a chunk of size 1008, which is longer than the specified 1000
    Created a chunk of size 1025, which is longer than the specified 1000
    Created a chunk of size 1206, which is longer than the specified 1000
    Created a chunk of size 1202, which is longer than the specified 1000
    Created a chunk of size 1206, which is longer than the specified 1000
    Created a chunk of size 1272, which is longer than the specified 1000
    Created a chunk of size 1092, which is longer than the specified 1000
    Created a chunk of size 1303, which is longer than the specified 1000
    Created a chunk of size 1029, which is longer than the specified 1000
    Created a chunk of size 1117, which is longer than the specified 1000
    Created a chunk of size 1438, which is longer than the specified 1000
    Created a chunk of size 3055, which is longer than the specified 1000
    Created a chunk of size 1628, which is longer than the specified 1000
    Created a chunk of size 1566, which is longer than the specified 1000
    Created a chunk of size 1179, which is longer than the specified 1000
    Created a chunk of size 1006, which is longer than the specified 1000
    Created a chunk of size 1213, which is longer than the specified 1000
    Created a chunk of size 2461, which is longer than the specified 1000
    Created a chunk of size 1849, which is longer than the specified 1000
    Created a chunk of size 1398, which is longer than the specified 1000
    Created a chunk of size 1469, which is longer than the specified 1000
    Created a chunk of size 1220, which is longer than the specified 1000
    Created a chunk of size 1048, which is longer than the specified 1000
    Created a chunk of size 1040, which is longer than the specified 1000
    Created a chunk of size 1052, which is longer than the specified 1000
    Created a chunk of size 1052, which is longer than the specified 1000
    Created a chunk of size 1304, which is longer than the specified 1000
    Created a chunk of size 1147, which is longer than the specified 1000
    Created a chunk of size 1236, which is longer than the specified 1000
    Created a chunk of size 1411, which is longer than the specified 1000
    Created a chunk of size 1181, which is longer than the specified 1000
    Created a chunk of size 1357, which is longer than the specified 1000
    Created a chunk of size 1706, which is longer than the specified 1000
    Created a chunk of size 1099, which is longer than the specified 1000
    Created a chunk of size 1221, which is longer than the specified 1000
    Created a chunk of size 1066, which is longer than the specified 1000
    Created a chunk of size 1223, which is longer than the specified 1000
    Created a chunk of size 1202, which is longer than the specified 1000
    Created a chunk of size 2806, which is longer than the specified 1000
    Created a chunk of size 1180, which is longer than the specified 1000
    Created a chunk of size 1338, which is longer than the specified 1000
    Created a chunk of size 1074, which is longer than the specified 1000
    Created a chunk of size 1025, which is longer than the specified 1000
    Created a chunk of size 1017, which is longer than the specified 1000
    Created a chunk of size 1497, which is longer than the specified 1000
    Created a chunk of size 1151, which is longer than the specified 1000
    Created a chunk of size 1287, which is longer than the specified 1000
    Created a chunk of size 1359, which is longer than the specified 1000
    Created a chunk of size 1075, which is longer than the specified 1000
    Created a chunk of size 1037, which is longer than the specified 1000
    Created a chunk of size 1080, which is longer than the specified 1000
    Created a chunk of size 1354, which is longer than the specified 1000
    Created a chunk of size 1033, which is longer than the specified 1000
    Created a chunk of size 1473, which is longer than the specified 1000
    Created a chunk of size 1074, which is longer than the specified 1000
    Created a chunk of size 2091, which is longer than the specified 1000
    Created a chunk of size 1388, which is longer than the specified 1000
    Created a chunk of size 1040, which is longer than the specified 1000
    Created a chunk of size 1040, which is longer than the specified 1000
    Created a chunk of size 1158, which is longer than the specified 1000
    Created a chunk of size 1683, which is longer than the specified 1000
    Created a chunk of size 2424, which is longer than the specified 1000
    Created a chunk of size 1877, which is longer than the specified 1000
    Created a chunk of size 1002, which is longer than the specified 1000
    Created a chunk of size 2175, which is longer than the specified 1000
    Created a chunk of size 1011, which is longer than the specified 1000
    Created a chunk of size 1915, which is longer than the specified 1000
    Created a chunk of size 1587, which is longer than the specified 1000
    Created a chunk of size 1969, which is longer than the specified 1000
    Created a chunk of size 1687, which is longer than the specified 1000
    Created a chunk of size 1732, which is longer than the specified 1000
    Created a chunk of size 1322, which is longer than the specified 1000
    Created a chunk of size 1339, which is longer than the specified 1000
    Created a chunk of size 3083, which is longer than the specified 1000
    Created a chunk of size 2148, which is longer than the specified 1000
    Created a chunk of size 1647, which is longer than the specified 1000
    Created a chunk of size 1698, which is longer than the specified 1000
    Created a chunk of size 1012, which is longer than the specified 1000
    Created a chunk of size 1919, which is longer than the specified 1000
    Created a chunk of size 1676, which is longer than the specified 1000
    Created a chunk of size 1581, which is longer than the specified 1000
    Created a chunk of size 2559, which is longer than the specified 1000
    Created a chunk of size 1247, which is longer than the specified 1000
    Created a chunk of size 1220, which is longer than the specified 1000
    Created a chunk of size 1768, which is longer than the specified 1000
    Created a chunk of size 1287, which is longer than the specified 1000
    Created a chunk of size 1300, which is longer than the specified 1000
    Created a chunk of size 1390, which is longer than the specified 1000
    Created a chunk of size 1423, which is longer than the specified 1000
    Created a chunk of size 1018, which is longer than the specified 1000
    Created a chunk of size 1185, which is longer than the specified 1000
    Created a chunk of size 2858, which is longer than the specified 1000
    Created a chunk of size 1149, which is longer than the specified 1000
    Created a chunk of size 1730, which is longer than the specified 1000
    Created a chunk of size 1026, which is longer than the specified 1000
    Created a chunk of size 1913, which is longer than the specified 1000
    Created a chunk of size 1362, which is longer than the specified 1000
    Created a chunk of size 1324, which is longer than the specified 1000
    Created a chunk of size 1073, which is longer than the specified 1000
    Created a chunk of size 1455, which is longer than the specified 1000
    Created a chunk of size 1621, which is longer than the specified 1000
    Created a chunk of size 1516, which is longer than the specified 1000
    Created a chunk of size 1633, which is longer than the specified 1000
    Created a chunk of size 1620, which is longer than the specified 1000
    Created a chunk of size 1856, which is longer than the specified 1000
    Created a chunk of size 1562, which is longer than the specified 1000
    Created a chunk of size 1729, which is longer than the specified 1000
    Created a chunk of size 1203, which is longer than the specified 1000
    Created a chunk of size 1307, which is longer than the specified 1000
    Created a chunk of size 1331, which is longer than the specified 1000
    Created a chunk of size 1295, which is longer than the specified 1000
    Created a chunk of size 1101, which is longer than the specified 1000
    Created a chunk of size 1090, which is longer than the specified 1000
    Created a chunk of size 1241, which is longer than the specified 1000
    Created a chunk of size 1138, which is longer than the specified 1000
    Created a chunk of size 1076, which is longer than the specified 1000
    Created a chunk of size 1210, which is longer than the specified 1000
    Created a chunk of size 1183, which is longer than the specified 1000
    Created a chunk of size 1353, which is longer than the specified 1000
    Created a chunk of size 1271, which is longer than the specified 1000
    Created a chunk of size 1778, which is longer than the specified 1000
    Created a chunk of size 1141, which is longer than the specified 1000
    Created a chunk of size 1099, which is longer than the specified 1000
    Created a chunk of size 2090, which is longer than the specified 1000
    Created a chunk of size 1056, which is longer than the specified 1000
    Created a chunk of size 1120, which is longer than the specified 1000
    Created a chunk of size 1048, which is longer than the specified 1000
    Created a chunk of size 1072, which is longer than the specified 1000
    Created a chunk of size 1367, which is longer than the specified 1000
    Created a chunk of size 1246, which is longer than the specified 1000
    Created a chunk of size 1766, which is longer than the specified 1000
    Created a chunk of size 1105, which is longer than the specified 1000
    Created a chunk of size 1400, which is longer than the specified 1000
    Created a chunk of size 1488, which is longer than the specified 1000
    Created a chunk of size 1672, which is longer than the specified 1000
    Created a chunk of size 1137, which is longer than the specified 1000
    Created a chunk of size 1500, which is longer than the specified 1000
    Created a chunk of size 1224, which is longer than the specified 1000
    Created a chunk of size 1414, which is longer than the specified 1000
    Created a chunk of size 1242, which is longer than the specified 1000
    Created a chunk of size 1551, which is longer than the specified 1000
    Created a chunk of size 1268, which is longer than the specified 1000
    Created a chunk of size 1130, which is longer than the specified 1000
    Created a chunk of size 2023, which is longer than the specified 1000
    Created a chunk of size 1878, which is longer than the specified 1000
    Created a chunk of size 1364, which is longer than the specified 1000
    Created a chunk of size 1212, which is longer than the specified 1000
    Created a chunk of size 1792, which is longer than the specified 1000
    Created a chunk of size 1055, which is longer than the specified 1000
    Created a chunk of size 1496, which is longer than the specified 1000
    Created a chunk of size 1045, which is longer than the specified 1000
    Created a chunk of size 1501, which is longer than the specified 1000
    Created a chunk of size 1208, which is longer than the specified 1000
    Created a chunk of size 1356, which is longer than the specified 1000
    Created a chunk of size 1351, which is longer than the specified 1000
    Created a chunk of size 1130, which is longer than the specified 1000
    Created a chunk of size 1133, which is longer than the specified 1000
    Created a chunk of size 1381, which is longer than the specified 1000
    Created a chunk of size 1120, which is longer than the specified 1000
    Created a chunk of size 1200, which is longer than the specified 1000
    Created a chunk of size 1202, which is longer than the specified 1000
    Created a chunk of size 1149, which is longer than the specified 1000
    Created a chunk of size 1196, which is longer than the specified 1000
    Created a chunk of size 3173, which is longer than the specified 1000
    Created a chunk of size 1106, which is longer than the specified 1000
    Created a chunk of size 1211, which is longer than the specified 1000
    Created a chunk of size 1530, which is longer than the specified 1000
    Created a chunk of size 1471, which is longer than the specified 1000
    Created a chunk of size 1353, which is longer than the specified 1000
    Created a chunk of size 1279, which is longer than the specified 1000
    Created a chunk of size 1101, which is longer than the specified 1000
    Created a chunk of size 1123, which is longer than the specified 1000
    Created a chunk of size 1848, which is longer than the specified 1000
    Created a chunk of size 1197, which is longer than the specified 1000
    Created a chunk of size 1235, which is longer than the specified 1000
    Created a chunk of size 1314, which is longer than the specified 1000
    Created a chunk of size 1043, which is longer than the specified 1000
    Created a chunk of size 1183, which is longer than the specified 1000
    Created a chunk of size 1182, which is longer than the specified 1000
    Created a chunk of size 1269, which is longer than the specified 1000
    Created a chunk of size 1416, which is longer than the specified 1000
    Created a chunk of size 1462, which is longer than the specified 1000
    Created a chunk of size 1120, which is longer than the specified 1000
    Created a chunk of size 1033, which is longer than the specified 1000
    Created a chunk of size 1143, which is longer than the specified 1000
    Created a chunk of size 1537, which is longer than the specified 1000
    Created a chunk of size 1381, which is longer than the specified 1000
    Created a chunk of size 2286, which is longer than the specified 1000
    Created a chunk of size 1175, which is longer than the specified 1000
    Created a chunk of size 1187, which is longer than the specified 1000
    Created a chunk of size 1494, which is longer than the specified 1000
    Created a chunk of size 1597, which is longer than the specified 1000
    Created a chunk of size 1203, which is longer than the specified 1000
    Created a chunk of size 1058, which is longer than the specified 1000
    Created a chunk of size 1261, which is longer than the specified 1000
    Created a chunk of size 1189, which is longer than the specified 1000
    Created a chunk of size 1388, which is longer than the specified 1000
    Created a chunk of size 1224, which is longer than the specified 1000
    Created a chunk of size 1226, which is longer than the specified 1000
    Created a chunk of size 1289, which is longer than the specified 1000
    Created a chunk of size 1157, which is longer than the specified 1000
    Created a chunk of size 1095, which is longer than the specified 1000
    Created a chunk of size 2196, which is longer than the specified 1000
    Created a chunk of size 1029, which is longer than the specified 1000
    Created a chunk of size 1077, which is longer than the specified 1000
    Created a chunk of size 1848, which is longer than the specified 1000
    Created a chunk of size 1095, which is longer than the specified 1000
    Created a chunk of size 1418, which is longer than the specified 1000
    Created a chunk of size 1069, which is longer than the specified 1000
    Created a chunk of size 2573, which is longer than the specified 1000
    Created a chunk of size 1512, which is longer than the specified 1000
    Created a chunk of size 1046, which is longer than the specified 1000
    Created a chunk of size 1792, which is longer than the specified 1000
    Created a chunk of size 1042, which is longer than the specified 1000
    Created a chunk of size 1125, which is longer than the specified 1000
    Created a chunk of size 1165, which is longer than the specified 1000
    Created a chunk of size 1030, which is longer than the specified 1000
    Created a chunk of size 1484, which is longer than the specified 1000
    Created a chunk of size 2796, which is longer than the specified 1000
    Created a chunk of size 1026, which is longer than the specified 1000
    Created a chunk of size 1726, which is longer than the specified 1000
    Created a chunk of size 1628, which is longer than the specified 1000
    Created a chunk of size 1881, which is longer than the specified 1000
    Created a chunk of size 1441, which is longer than the specified 1000
    Created a chunk of size 1175, which is longer than the specified 1000
    Created a chunk of size 1360, which is longer than the specified 1000
    Created a chunk of size 1210, which is longer than the specified 1000
    Created a chunk of size 1425, which is longer than the specified 1000
    Created a chunk of size 1560, which is longer than the specified 1000
    Created a chunk of size 1131, which is longer than the specified 1000
    Created a chunk of size 1276, which is longer than the specified 1000
    Created a chunk of size 1068, which is longer than the specified 1000
    Created a chunk of size 1494, which is longer than the specified 1000
    Created a chunk of size 1246, which is longer than the specified 1000
    Created a chunk of size 2621, which is longer than the specified 1000
    Created a chunk of size 1264, which is longer than the specified 1000
    Created a chunk of size 1166, which is longer than the specified 1000
    Created a chunk of size 1332, which is longer than the specified 1000
    Created a chunk of size 3499, which is longer than the specified 1000
    Created a chunk of size 1651, which is longer than the specified 1000
    Created a chunk of size 1794, which is longer than the specified 1000
    Created a chunk of size 2162, which is longer than the specified 1000
    Created a chunk of size 1061, which is longer than the specified 1000
    Created a chunk of size 1083, which is longer than the specified 1000
    Created a chunk of size 1018, which is longer than the specified 1000
    Created a chunk of size 1751, which is longer than the specified 1000
    Created a chunk of size 1301, which is longer than the specified 1000
    Created a chunk of size 1025, which is longer than the specified 1000
    Created a chunk of size 1489, which is longer than the specified 1000
    Created a chunk of size 1481, which is longer than the specified 1000
    Created a chunk of size 1505, which is longer than the specified 1000
    Created a chunk of size 1497, which is longer than the specified 1000
    Created a chunk of size 1505, which is longer than the specified 1000
    Created a chunk of size 1282, which is longer than the specified 1000
    Created a chunk of size 1224, which is longer than the specified 1000
    Created a chunk of size 1261, which is longer than the specified 1000
    Created a chunk of size 1123, which is longer than the specified 1000
    Created a chunk of size 1137, which is longer than the specified 1000
    Created a chunk of size 2183, which is longer than the specified 1000
    Created a chunk of size 1039, which is longer than the specified 1000
    Created a chunk of size 1135, which is longer than the specified 1000
    Created a chunk of size 1254, which is longer than the specified 1000
    Created a chunk of size 1234, which is longer than the specified 1000
    Created a chunk of size 1111, which is longer than the specified 1000
    Created a chunk of size 1135, which is longer than the specified 1000
    Created a chunk of size 2023, which is longer than the specified 1000
    Created a chunk of size 1216, which is longer than the specified 1000
    Created a chunk of size 1013, which is longer than the specified 1000
    Created a chunk of size 1152, which is longer than the specified 1000
    Created a chunk of size 1087, which is longer than the specified 1000
    Created a chunk of size 1040, which is longer than the specified 1000
    Created a chunk of size 1330, which is longer than the specified 1000
    Created a chunk of size 2342, which is longer than the specified 1000
    Created a chunk of size 1940, which is longer than the specified 1000
    Created a chunk of size 1621, which is longer than the specified 1000
    Created a chunk of size 2169, which is longer than the specified 1000
    Created a chunk of size 1824, which is longer than the specified 1000
    Created a chunk of size 1554, which is longer than the specified 1000
    Created a chunk of size 1457, which is longer than the specified 1000
    Created a chunk of size 1486, which is longer than the specified 1000
    Created a chunk of size 1556, which is longer than the specified 1000
    Created a chunk of size 1012, which is longer than the specified 1000
    Created a chunk of size 1484, which is longer than the specified 1000
    Created a chunk of size 1039, which is longer than the specified 1000
    Created a chunk of size 1335, which is longer than the specified 1000
    Created a chunk of size 1684, which is longer than the specified 1000
    Created a chunk of size 1537, which is longer than the specified 1000
    Created a chunk of size 1136, which is longer than the specified 1000
    Created a chunk of size 1219, which is longer than the specified 1000
    Created a chunk of size 1011, which is longer than the specified 1000
    Created a chunk of size 1055, which is longer than the specified 1000
    Created a chunk of size 1433, which is longer than the specified 1000
    Created a chunk of size 1263, which is longer than the specified 1000
    Created a chunk of size 1014, which is longer than the specified 1000
    Created a chunk of size 1107, which is longer than the specified 1000
    Created a chunk of size 2702, which is longer than the specified 1000
    Created a chunk of size 1237, which is longer than the specified 1000
    Created a chunk of size 1172, which is longer than the specified 1000
    Created a chunk of size 1517, which is longer than the specified 1000
    Created a chunk of size 1589, which is longer than the specified 1000
    Created a chunk of size 1681, which is longer than the specified 1000
    Created a chunk of size 2244, which is longer than the specified 1000
    Created a chunk of size 1505, which is longer than the specified 1000
    Created a chunk of size 1228, which is longer than the specified 1000
    Created a chunk of size 1801, which is longer than the specified 1000
    Created a chunk of size 1856, which is longer than the specified 1000
    Created a chunk of size 2171, which is longer than the specified 1000
    Created a chunk of size 2450, which is longer than the specified 1000
    Created a chunk of size 1110, which is longer than the specified 1000
    Created a chunk of size 1148, which is longer than the specified 1000
    Created a chunk of size 1050, which is longer than the specified 1000
    Created a chunk of size 1014, which is longer than the specified 1000
    Created a chunk of size 1458, which is longer than the specified 1000
    Created a chunk of size 1270, which is longer than the specified 1000
    Created a chunk of size 1287, which is longer than the specified 1000
    Created a chunk of size 1127, which is longer than the specified 1000
    Created a chunk of size 1576, which is longer than the specified 1000
    Created a chunk of size 1350, which is longer than the specified 1000
    Created a chunk of size 2283, which is longer than the specified 1000
    Created a chunk of size 2211, which is longer than the specified 1000
    Created a chunk of size 1167, which is longer than the specified 1000
    Created a chunk of size 1038, which is longer than the specified 1000
    Created a chunk of size 1117, which is longer than the specified 1000
    Created a chunk of size 1160, which is longer than the specified 1000
    Created a chunk of size 1163, which is longer than the specified 1000
    Created a chunk of size 1013, which is longer than the specified 1000
    Created a chunk of size 1226, which is longer than the specified 1000
    Created a chunk of size 1336, which is longer than the specified 1000
    Created a chunk of size 1012, which is longer than the specified 1000
    Created a chunk of size 2833, which is longer than the specified 1000
    Created a chunk of size 1201, which is longer than the specified 1000
    Created a chunk of size 1172, which is longer than the specified 1000
    Created a chunk of size 1438, which is longer than the specified 1000
    Created a chunk of size 1259, which is longer than the specified 1000
    Created a chunk of size 1452, which is longer than the specified 1000
    Created a chunk of size 1377, which is longer than the specified 1000
    Created a chunk of size 1001, which is longer than the specified 1000
    Created a chunk of size 1240, which is longer than the specified 1000
    Created a chunk of size 1142, which is longer than the specified 1000
    Created a chunk of size 1338, which is longer than the specified 1000
    Created a chunk of size 1057, which is longer than the specified 1000
    Created a chunk of size 1040, which is longer than the specified 1000
    Created a chunk of size 1579, which is longer than the specified 1000
    Created a chunk of size 1176, which is longer than the specified 1000
    Created a chunk of size 1081, which is longer than the specified 1000
    Created a chunk of size 1751, which is longer than the specified 1000
    Created a chunk of size 1064, which is longer than the specified 1000
    Created a chunk of size 1029, which is longer than the specified 1000
    Created a chunk of size 1937, which is longer than the specified 1000
    Created a chunk of size 1972, which is longer than the specified 1000
    Created a chunk of size 1417, which is longer than the specified 1000
    Created a chunk of size 1203, which is longer than the specified 1000
    Created a chunk of size 1314, which is longer than the specified 1000
    Created a chunk of size 1088, which is longer than the specified 1000
    Created a chunk of size 1455, which is longer than the specified 1000
    Created a chunk of size 1467, which is longer than the specified 1000
    Created a chunk of size 1476, which is longer than the specified 1000
    Created a chunk of size 1354, which is longer than the specified 1000
    Created a chunk of size 1403, which is longer than the specified 1000
    Created a chunk of size 1366, which is longer than the specified 1000
    Created a chunk of size 1112, which is longer than the specified 1000
    Created a chunk of size 1512, which is longer than the specified 1000
    Created a chunk of size 1262, which is longer than the specified 1000
    Created a chunk of size 1405, which is longer than the specified 1000
    Created a chunk of size 2221, which is longer than the specified 1000
    Created a chunk of size 1128, which is longer than the specified 1000
    Created a chunk of size 1021, which is longer than the specified 1000
    Created a chunk of size 1532, which is longer than the specified 1000
    Created a chunk of size 1535, which is longer than the specified 1000
    Created a chunk of size 1230, which is longer than the specified 1000
    Created a chunk of size 2456, which is longer than the specified 1000
    Created a chunk of size 1047, which is longer than the specified 1000
    Created a chunk of size 1320, which is longer than the specified 1000
    Created a chunk of size 1144, which is longer than the specified 1000
    Created a chunk of size 1509, which is longer than the specified 1000
    Created a chunk of size 1003, which is longer than the specified 1000
    Created a chunk of size 1025, which is longer than the specified 1000
    Created a chunk of size 1197, which is longer than the specified 1000
    

    8244
    

Then embed chunks and upload them to the DeepLake.

This can take several minutes. 


```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
embeddings
```




    OpenAIEmbeddings(client=<class 'openai.api_resources.embedding.Embedding'>, model='text-embedding-ada-002', deployment='text-embedding-ada-002', openai_api_version='', openai_api_base='', openai_api_type='', openai_proxy='', embedding_ctx_length=8191, openai_api_key='', openai_organization='', allowed_special=set(), disallowed_special='all', chunk_size=1000, max_retries=6, request_timeout=None, headers=None, tiktoken_model_name=None, show_progress_bar=False, model_kwargs={})




```python
from langchain_community.vectorstores import DeepLake

username = "<USERNAME_OR_ORG>"


db = DeepLake.from_documents(
    texts, embeddings, dataset_path=f"hub://{username}/langchain-code", overwrite=True
)
db
```

    Your Deep Lake dataset has been successfully created!
    

     

    Dataset(path='hub://adilkhan/langchain-code', tensors=['embedding', 'id', 'metadata', 'text'])
    
      tensor      htype       shape       dtype  compression
      -------    -------     -------     -------  ------- 
     embedding  embedding  (8244, 1536)  float32   None   
        id        text      (8244, 1)      str     None   
     metadata     json      (8244, 1)      str     None   
       text       text      (8244, 1)      str     None   
    

    




    <langchain_community.vectorstores.deeplake.DeepLake at 0x7fe1b67d7a30>



`Optional`: You can also use Deep Lake's Managed Tensor Database as a hosting service and run queries there. In order to do so, it is necessary to specify the runtime parameter as {'tensor_db': True} during the creation of the vector store. This configuration enables the execution of queries on the Managed Tensor Database, rather than on the client side. It should be noted that this functionality is not applicable to datasets stored locally or in-memory. In the event that a vector store has already been created outside of the Managed Tensor Database, it is possible to transfer it to the Managed Tensor Database by following the prescribed steps.


```python
# from langchain_community.vectorstores import DeepLake

# db = DeepLake.from_documents(
#     texts, embeddings, dataset_path=f"hub://{<org_id>}/langchain-code", runtime={"tensor_db": True}
# )
# db
```

### Question Answering
First load the dataset, construct the retriever, then construct the Conversational Chain


```python
db = DeepLake(
    dataset_path=f"hub://{username}/langchain-code",
    read_only=True,
    embedding=embeddings,
)
```

    Deep Lake Dataset in hub://adilkhan/langchain-code already exists, loading from the storage
    


```python
retriever = db.as_retriever()
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["fetch_k"] = 20
retriever.search_kwargs["maximal_marginal_relevance"] = True
retriever.search_kwargs["k"] = 20
```

You can also specify user defined functions using [Deep Lake filters](https://docs.deeplake.ai/en/latest/deeplake.core.dataset.html#deeplake.core.dataset.Dataset.filter)


```python
def filter(x):
    # filter based on source code
    if "something" in x["text"].data()["value"]:
        return False

    # filter based on path e.g. extension
    metadata = x["metadata"].data()["value"]
    return "only_this" in metadata["source"] or "also_that" in metadata["source"]


### turn on below for custom filtering
# retriever.search_kwargs['filter'] = filter
```


```python
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model_name="gpt-3.5-turbo-0613"
)  # 'ada' 'gpt-3.5-turbo-0613' 'gpt-4',
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)
```


```python
questions = [
    "What is the class hierarchy?",
    "What classes are derived from the Chain class?",
    "What kind of retrievers does LangChain have?",
]
chat_history = []
qa_dict = {}

for question in questions:
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    qa_dict[question] = result["answer"]
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")
```

    -> **Question**: What is the class hierarchy? 
    
    **Answer**: The class hierarchy for Memory is as follows:
    
        BaseMemory --> BaseChatMemory --> <name>Memory  # Examples: ZepMemory, MotorheadMemory
    
    The class hierarchy for ChatMessageHistory is as follows:
    
        BaseChatMessageHistory --> <name>ChatMessageHistory  # Example: ZepChatMessageHistory
    
    The class hierarchy for Prompt is as follows:
    
        BasePromptTemplate --> PipelinePromptTemplate
                               StringPromptTemplate --> PromptTemplate
                                                        FewShotPromptTemplate
                                                        FewShotPromptWithTemplates
                               BaseChatPromptTemplate --> AutoGPTPrompt
                                                          ChatPromptTemplate --> AgentScratchPadChatPromptTemplate
     
    
    -> **Question**: What classes are derived from the Chain class? 
    
    **Answer**: The classes derived from the Chain class are:
    
    - APIChain
    - OpenAPIEndpointChain
    - AnalyzeDocumentChain
    - MapReduceDocumentsChain
    - MapRerankDocumentsChain
    - ReduceDocumentsChain
    - RefineDocumentsChain
    - StuffDocumentsChain
    - ConstitutionalChain
    - ConversationChain
    - ChatVectorDBChain
    - ConversationalRetrievalChain
    - FalkorDBQAChain
    - FlareChain
    - ArangoGraphQAChain
    - GraphQAChain
    - GraphCypherQAChain
    - HugeGraphQAChain
    - KuzuQAChain
    - NebulaGraphQAChain
    - NeptuneOpenCypherQAChain
    - GraphSparqlQAChain
    - HypotheticalDocumentEmbedder
    - LLMChain
    - LLMBashChain
    - LLMCheckerChain
    - LLMMathChain
    - LLMRequestsChain
    - LLMSummarizationCheckerChain
    - MapReduceChain
    - OpenAIModerationChain
    - NatBotChain
    - QAGenerationChain
    - QAWithSourcesChain
    - RetrievalQAWithSourcesChain
    - VectorDBQAWithSourcesChain
    - RetrievalQA
    - VectorDBQA
    - LLMRouterChain
    - MultiPromptChain
    - MultiRetrievalQAChain
    - MultiRouteChain
    - RouterChain
    - SequentialChain
    - SimpleSequentialChain
    - TransformChain
    - TaskPlaningChain
    - QueryChain
    - CPALChain
     
    
    -> **Question**: What kind of retrievers does LangChain have? 
    
    **Answer**: The LangChain class includes various types of retrievers such as:
    
    - ArxivRetriever
    - AzureAISearchRetriever
    - BM25Retriever
    - ChaindeskRetriever
    - ChatGPTPluginRetriever
    - ContextualCompressionRetriever
    - DocArrayRetriever
    - ElasticSearchBM25Retriever
    - EnsembleRetriever
    - GoogleVertexAISearchRetriever
    - AmazonKendraRetriever
    - KNNRetriever
    - LlamaIndexGraphRetriever and LlamaIndexRetriever
    - MergerRetriever
    - MetalRetriever
    - MilvusRetriever
    - MultiQueryRetriever
    - ParentDocumentRetriever
    - PineconeHybridSearchRetriever
    - PubMedRetriever
    - RePhraseQueryRetriever
    - RemoteLangChainRetriever
    - SelfQueryRetriever
    - SVMRetriever
    - TFIDFRetriever
    - TimeWeightedVectorStoreRetriever
    - VespaRetriever
    - WeaviateHybridSearchRetriever
    - WebResearchRetriever
    - WikipediaRetriever
    - ZepRetriever
    - ZillizRetriever 
    
    


```python
qa_dict
```




    {'question': 'LangChain possesses a variety of retrievers including:\n\n1. ArxivRetriever\n2. AzureAISearchRetriever\n3. BM25Retriever\n4. ChaindeskRetriever\n5. ChatGPTPluginRetriever\n6. ContextualCompressionRetriever\n7. DocArrayRetriever\n8. ElasticSearchBM25Retriever\n9. EnsembleRetriever\n10. GoogleVertexAISearchRetriever\n11. AmazonKendraRetriever\n12. KNNRetriever\n13. LlamaIndexGraphRetriever\n14. LlamaIndexRetriever\n15. MergerRetriever\n16. MetalRetriever\n17. MilvusRetriever\n18. MultiQueryRetriever\n19. ParentDocumentRetriever\n20. PineconeHybridSearchRetriever\n21. PubMedRetriever\n22. RePhraseQueryRetriever\n23. RemoteLangChainRetriever\n24. SelfQueryRetriever\n25. SVMRetriever\n26. TFIDFRetriever\n27. TimeWeightedVectorStoreRetriever\n28. VespaRetriever\n29. WeaviateHybridSearchRetriever\n30. WebResearchRetriever\n31. WikipediaRetriever\n32. ZepRetriever\n33. ZillizRetriever\n\nIt also includes self query translators like:\n\n1. ChromaTranslator\n2. DeepLakeTranslator\n3. MyScaleTranslator\n4. PineconeTranslator\n5. QdrantTranslator\n6. WeaviateTranslator\n\nAnd remote retrievers like:\n\n1. RemoteLangChainRetriever'}




```python
print(qa_dict["What is the class hierarchy?"])
```

    The class hierarchy for Memory is as follows:
    
        BaseMemory --> BaseChatMemory --> <name>Memory  # Examples: ZepMemory, MotorheadMemory
    
    The class hierarchy for ChatMessageHistory is as follows:
    
        BaseChatMessageHistory --> <name>ChatMessageHistory  # Example: ZepChatMessageHistory
    
    The class hierarchy for Prompt is as follows:
    
        BasePromptTemplate --> PipelinePromptTemplate
                               StringPromptTemplate --> PromptTemplate
                                                        FewShotPromptTemplate
                                                        FewShotPromptWithTemplates
                               BaseChatPromptTemplate --> AutoGPTPrompt
                                                          ChatPromptTemplate --> AgentScratchPadChatPromptTemplate
    
    


```python
print(qa_dict["What classes are derived from the Chain class?"])
```

    The classes derived from the Chain class are:
    
    - APIChain
    - OpenAPIEndpointChain
    - AnalyzeDocumentChain
    - MapReduceDocumentsChain
    - MapRerankDocumentsChain
    - ReduceDocumentsChain
    - RefineDocumentsChain
    - StuffDocumentsChain
    - ConstitutionalChain
    - ConversationChain
    - ChatVectorDBChain
    - ConversationalRetrievalChain
    - FlareChain
    - ArangoGraphQAChain
    - GraphQAChain
    - GraphCypherQAChain
    - HugeGraphQAChain
    - KuzuQAChain
    - NebulaGraphQAChain
    - NeptuneOpenCypherQAChain
    - GraphSparqlQAChain
    - HypotheticalDocumentEmbedder
    - LLMChain
    - LLMBashChain
    - LLMCheckerChain
    - LLMMathChain
    - LLMRequestsChain
    - LLMSummarizationCheckerChain
    - MapReduceChain
    - OpenAIModerationChain
    - NatBotChain
    - QAGenerationChain
    - QAWithSourcesChain
    - RetrievalQAWithSourcesChain
    - VectorDBQAWithSourcesChain
    - RetrievalQA
    - VectorDBQA
    - LLMRouterChain
    - MultiPromptChain
    - MultiRetrievalQAChain
    - MultiRouteChain
    - RouterChain
    - SequentialChain
    - SimpleSequentialChain
    - TransformChain
    - TaskPlaningChain
    - QueryChain
    - CPALChain
    
    


```python
print(qa_dict["What kind of retrievers does LangChain have?"])
```

    The LangChain class includes various types of retrievers such as:
    
    - ArxivRetriever
    - AzureAISearchRetriever
    - BM25Retriever
    - ChaindeskRetriever
    - ChatGPTPluginRetriever
    - ContextualCompressionRetriever
    - DocArrayRetriever
    - ElasticSearchBM25Retriever
    - EnsembleRetriever
    - GoogleVertexAISearchRetriever
    - AmazonKendraRetriever
    - KNNRetriever
    - LlamaIndexGraphRetriever and LlamaIndexRetriever
    - MergerRetriever
    - MetalRetriever
    - MilvusRetriever
    - MultiQueryRetriever
    - ParentDocumentRetriever
    - PineconeHybridSearchRetriever
    - PubMedRetriever
    - RePhraseQueryRetriever
    - RemoteLangChainRetriever
    - SelfQueryRetriever
    - SVMRetriever
    - TFIDFRetriever
    - TimeWeightedVectorStoreRetriever
    - VespaRetriever
    - WeaviateHybridSearchRetriever
    - WebResearchRetriever
    - WikipediaRetriever
    - ZepRetriever
    - ZillizRetriever
    
