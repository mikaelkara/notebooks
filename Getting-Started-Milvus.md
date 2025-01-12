# Get Started with Milvus

This notebook shows how to get started with [Milvus vector database](https://milvus.io/).

## Start Milvus DB (optional)

Milvus is available through the [Zilliz Cloud for Azure](https://zilliz.com/blog/zilliz-cloud-now-available-on-microsoft-azure) managed offering or [Azure Container Apps Add-Ons](https://learn.microsoft.com/azure/container-apps/services).

If you don't already have an instance of Milvus, you can run a local version for dev/test scenarios.


```polyglot-notebook
docker-compose up -d
```

    [31;1mCreating network "milvus" with the default driver[0m
    [31;1mPulling etcd (quay.io/coreos/etcd:v3.5.5)...[0m
    v3.5.5: Pulling from coreos/etcd
    Digest: sha256:89b6debd43502d1088f3e02f39442fd3e951aa52bee846ed601cf4477114b89e
    Status: Downloaded newer image for quay.io/coreos/etcd:v3.5.5
    [31;1mPulling minio (minio/minio:RELEASE.2023-03-20T20-16-18Z)...[0m
    RELEASE.2023-03-20T20-16-18Z: Pulling from minio/minio
    Digest: sha256:6d770d7f255cda1f18d841ffc4365cb7e0d237f6af6a15fcdb587480cd7c3b93
    Status: Downloaded newer image for minio/minio:RELEASE.2023-03-20T20-16-18Z
    [31;1mPulling standalone (milvusdb/milvus:v2.2.14)...[0m
    v2.2.14: Pulling from milvusdb/milvus
    Digest: sha256:099bd246ae15242eeb6e4ba9263b977eb9d92d069f28b3f51e1a4986cab0f90d
    Status: Downloaded newer image for milvusdb/milvus:v2.2.14
    [31;1mCreating milvus-etcd ... [0m
    [31;1mCreating milvus-minio ... [0m
    [31;1mCreating milvus-minio ... done[0m
    [31;1mCreating milvus-etcd  ... done[0m
    [31;1mCreating milvus-standalone ... [0m
    [31;1mCreating milvus-standalone ... done[0m
    

## Install Milvus C# SDK


```polyglot-notebook
#r "nuget: Milvus.Client, 2.2.2-preview.6"
```


<div><div></div><div></div><div><strong>Installed Packages</strong><ul><li><span>Milvus.Client, 2.2.2-preview.6</span></li></ul></div></div>



```polyglot-notebook
using Milvus.Client;
```

## Initialize Milvus Client

NOTE: If you have an instance of Milvus deployed in the cloud or your datacenter, replace `localhost` with your instance's host.


```polyglot-notebook
var milvusClient = new MilvusClient("localhost");
```

## Cleanup (optional)


```polyglot-notebook
await milvusClient.GetCollection("movies").DropAsync()
```

## List collections


```polyglot-notebook
var collections = milvusClient.GetCollection("movies");
```


```polyglot-notebook
collections
```


<details open="open" class="dni-treeview"><summary><span class="dni-code-hint"><code>Milvus.Client.MilvusCollection</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Name</td><td><div class="dni-plaintext"><pre>movies</pre></div></td></tr></tbody></table></div></details><style>
.dni-code-hint {
    font-style: italic;
    overflow: hidden;
    white-space: nowrap;
}
.dni-treeview {
    white-space: nowrap;
}
.dni-treeview td {
    vertical-align: top;
    text-align: start;
}
details.dni-treeview {
    padding-left: 1em;
}
table td {
    text-align: start;
}
table tr { 
    vertical-align: top; 
    margin: 0em 0px;
}
table tr td pre 
{ 
    vertical-align: top !important; 
    margin: 0em 0px !important;
} 
table th {
    text-align: start;
}
</style>


## Create collection

### Define schema


```polyglot-notebook
var schema = new CollectionSchema
{
    Fields =
    {
        FieldSchema.Create<long>("movie_id", isPrimaryKey: true),
        FieldSchema.CreateVarchar("movie_name", maxLength: 200),
        FieldSchema.CreateFloatVector("movie_description", dimension: 2)
    },
    Description = "Test movie search",
    EnableDynamicFields = true
};
```

### Create collection


```polyglot-notebook
var collection = await milvusClient.CreateCollectionAsync(collectionName: "movies",schema: schema, shardsNum: 2);
```


```polyglot-notebook
await collection.DescribeAsync()
```


<details open="open" class="dni-treeview"><summary><span class="dni-code-hint"><code>Milvus.Client.MilvusCollectionDescription</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Aliases</td><td><div class="dni-plaintext"><pre>[  ]</pre></div></td></tr><tr><td>CollectionName</td><td><div class="dni-plaintext"><pre>movies</pre></div></td></tr><tr><td>CollectionId</td><td><div class="dni-plaintext"><pre>447879535187460250</pre></div></td></tr><tr><td>ConsistencyLevel</td><td><span>Session</span></td></tr><tr><td>CreationTimestamp</td><td><div class="dni-plaintext"><pre>1708524868330</pre></div></td></tr><tr><td>Schema</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>Milvus.Client.CollectionSchema</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Name</td><td><div class="dni-plaintext"><pre>movies</pre></div></td></tr><tr><td>Description</td><td><div class="dni-plaintext"><pre>Test movie search</pre></div></td></tr><tr><td>Fields</td><td><table><thead><tr><th><i>index</i></th><th>value</th></tr></thead><tbody><tr><td>0</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>Milvus.Client.FieldSchema</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Name</td><td><div class="dni-plaintext"><pre>movie_id</pre></div></td></tr><tr><td>DataType</td><td><span>Int64</span></td></tr><tr><td>IsPrimaryKey</td><td><div class="dni-plaintext"><pre>True</pre></div></td></tr><tr><td>AutoId</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td>IsPartitionKey</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td>Description</td><td><div class="dni-plaintext"><pre></pre></div></td></tr><tr><td>IsDynamic</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td>MaxLength</td><td><div class="dni-plaintext"><pre>&lt;null&gt;</pre></div></td></tr><tr><td>Dimension</td><td><div class="dni-plaintext"><pre>&lt;null&gt;</pre></div></td></tr><tr><td>State</td><td><span>FieldCreated</span></td></tr><tr><td>FieldId</td><td><div class="dni-plaintext"><pre>100</pre></div></td></tr></tbody></table></div></details></td></tr><tr><td>1</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>Milvus.Client.FieldSchema</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Name</td><td><div class="dni-plaintext"><pre>movie_name</pre></div></td></tr><tr><td>DataType</td><td><span>VarChar</span></td></tr><tr><td>IsPrimaryKey</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td>AutoId</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td>IsPartitionKey</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td>Description</td><td><div class="dni-plaintext"><pre></pre></div></td></tr><tr><td>IsDynamic</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td>MaxLength</td><td><div class="dni-plaintext"><pre>200</pre></div></td></tr><tr><td>Dimension</td><td><div class="dni-plaintext"><pre>&lt;null&gt;</pre></div></td></tr><tr><td>State</td><td><span>FieldCreated</span></td></tr><tr><td>FieldId</td><td><div class="dni-plaintext"><pre>101</pre></div></td></tr></tbody></table></div></details></td></tr><tr><td>2</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>Milvus.Client.FieldSchema</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Name</td><td><div class="dni-plaintext"><pre>movie_description</pre></div></td></tr><tr><td>DataType</td><td><span>FloatVector</span></td></tr><tr><td>IsPrimaryKey</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td>AutoId</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td>IsPartitionKey</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td>Description</td><td><div class="dni-plaintext"><pre></pre></div></td></tr><tr><td>IsDynamic</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr><tr><td>MaxLength</td><td><div class="dni-plaintext"><pre>&lt;null&gt;</pre></div></td></tr><tr><td>Dimension</td><td><div class="dni-plaintext"><pre>2</pre></div></td></tr><tr><td>State</td><td><span>FieldCreated</span></td></tr><tr><td>FieldId</td><td><div class="dni-plaintext"><pre>102</pre></div></td></tr></tbody></table></div></details></td></tr></tbody></table></td></tr><tr><td>EnableDynamicFields</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr></tbody></table></div></details></td></tr><tr><td>ShardsNum</td><td><div class="dni-plaintext"><pre>2</pre></div></td></tr><tr><td>StartPositions</td><td><i>(empty)</i></td></tr></tbody></table></div></details><style>
.dni-code-hint {
    font-style: italic;
    overflow: hidden;
    white-space: nowrap;
}
.dni-treeview {
    white-space: nowrap;
}
.dni-treeview td {
    vertical-align: top;
    text-align: start;
}
details.dni-treeview {
    padding-left: 1em;
}
table td {
    text-align: start;
}
table tr { 
    vertical-align: top; 
    margin: 0em 0px;
}
table tr td pre 
{ 
    vertical-align: top !important; 
    margin: 0em 0px !important;
} 
table th {
    text-align: start;
}
</style>


## Add data to collection


```polyglot-notebook
var movieIds = new [] {1L,2L,3L,4L,5L};
var movieNames = new [] {"The Lion King","Inception","Toy Story","Pulp Function","Shrek"};
var movieDescriptions = new ReadOnlyMemory<float>[] { 
        new [] {0.10022575f, -0.23998135f},
        new [] {0.10327095f, 0.2563685f},
        new [] {0.095857024f, -0.201278f},
        new [] {0.106827796f, 0.21676421f},
        new [] {0.09568083f, -0.21177962f}
    };
```


```polyglot-notebook
await collection.InsertAsync(new FieldData[]
{
    FieldData.Create("movie_id", movieIds),
    FieldData.Create("movie_name", movieNames),
    FieldData.CreateFloatVector("movie_description", movieDescriptions)
});
```

## Persist data


```polyglot-notebook
await collection.FlushAsync();
```


```polyglot-notebook
await collection.GetEntityCountAsync()
```


<div class="dni-plaintext"><pre>5</pre></div><style>
.dni-code-hint {
    font-style: italic;
    overflow: hidden;
    white-space: nowrap;
}
.dni-treeview {
    white-space: nowrap;
}
.dni-treeview td {
    vertical-align: top;
    text-align: start;
}
details.dni-treeview {
    padding-left: 1em;
}
table td {
    text-align: start;
}
table tr { 
    vertical-align: top; 
    margin: 0em 0px;
}
table tr td pre 
{ 
    vertical-align: top !important; 
    margin: 0em 0px !important;
} 
table th {
    text-align: start;
}
</style>


## Search for data

### Create Index


```polyglot-notebook
await collection.CreateIndexAsync(
    fieldName: "movie_description", 
    indexType: IndexType.Flat, 
    metricType: SimilarityMetricType.L2, 
    extraParams: new Dictionary<string,string> {["nlist"] = "1024"}, 
    indexName: "movie_idx");
```

### Load collection


```polyglot-notebook
await collection.LoadAsync();
await collection.WaitForCollectionLoadAsync();
```

### Define search parameters


```polyglot-notebook
var parameters = new SearchParameters
{
    OutputFields = { "movie_name" },
    ConsistencyLevel = ConsistencyLevel.Strong,
    ExtraParameters = { ["nprobe"] = "1024" }
};
```

### Search for data

Search for data using embedding vectors for the query "A movie that's fun for the whole family".


```polyglot-notebook
var results = await collection.SearchAsync(
    vectorFieldName: "movie_description",
    vectors: new ReadOnlyMemory<float>[] { new[] {0.12217915f, -0.034832448f } },
    SimilarityMetricType.L2,
    limit: 3,
    parameters);
```


```polyglot-notebook
results
```


<details open="open" class="dni-treeview"><summary><span class="dni-code-hint"><code>Milvus.Client.SearchResults</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>CollectionName</td><td><div class="dni-plaintext"><pre>movies</pre></div></td></tr><tr><td>FieldsData</td><td><table><thead><tr><th><i>index</i></th><th>value</th></tr></thead><tbody><tr><td>0</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>Field: {FieldName: movie_name, DataType: VarChar, Data: 3, RowCount: 3}</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Data</td><td><div class="dni-plaintext"><pre>[ Toy Story, Shrek, The Lion King ]</pre></div></td></tr><tr><td>RowCount</td><td><div class="dni-plaintext"><pre>3</pre></div></td></tr><tr><td>FieldName</td><td><div class="dni-plaintext"><pre>movie_name</pre></div></td></tr><tr><td>FieldId</td><td><div class="dni-plaintext"><pre>0</pre></div></td></tr><tr><td>DataType</td><td><span>VarChar</span></td></tr><tr><td>IsDynamic</td><td><div class="dni-plaintext"><pre>False</pre></div></td></tr></tbody></table></div></details></td></tr></tbody></table></td></tr><tr><td>Ids</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>Milvus.Client.MilvusIds</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>LongIds</td><td><div class="dni-plaintext"><pre>[ 3, 5, 1 ]</pre></div></td></tr><tr><td>StringIds</td><td><div class="dni-plaintext"><pre>&lt;null&gt;</pre></div></td></tr></tbody></table></div></details></td></tr><tr><td>NumQueries</td><td><div class="dni-plaintext"><pre>1</pre></div></td></tr><tr><td>Scores</td><td><div class="dni-plaintext"><pre>[ 0.028396975, 0.032012466, 0.042568024 ]</pre></div></td></tr><tr><td>Limit</td><td><div class="dni-plaintext"><pre>3</pre></div></td></tr><tr><td>Limits</td><td><div class="dni-plaintext"><pre>[ 3 ]</pre></div></td></tr></tbody></table></div></details><style>
.dni-code-hint {
    font-style: italic;
    overflow: hidden;
    white-space: nowrap;
}
.dni-treeview {
    white-space: nowrap;
}
.dni-treeview td {
    vertical-align: top;
    text-align: start;
}
details.dni-treeview {
    padding-left: 1em;
}
table td {
    text-align: start;
}
table tr { 
    vertical-align: top; 
    margin: 0em 0px;
}
table tr td pre 
{ 
    vertical-align: top !important; 
    margin: 0em 0px !important;
} 
table th {
    text-align: start;
}
</style>


##
