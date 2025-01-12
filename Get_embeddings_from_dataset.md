## Get embeddings
This notebook contains some helpful snippets you can use to embed text with the 'text-embedding-ada-002' model via Azure OpenAI API.

## Installation
Install the Azure Open AI SDK using the below command.


```csharp
#r "nuget: Azure.AI.OpenAI, 1.0.0-beta.14"
```


<div><div></div><div></div><div><strong>Installed Packages</strong><ul><li><span>Azure.AI.OpenAI, 1.0.0-beta.14</span></li></ul></div></div>



```csharp
#r "nuget:Microsoft.DotNet.Interactive.AIUtilities, 1.0.0-beta.24129.1"
```


```csharp
using Microsoft.DotNet.Interactive;
using Microsoft.DotNet.Interactive.AIUtilities;
```

## Run this cell, it will prompt you for the apiKey, endPoint, and embedding deployment


```csharp
var azureOpenAIKey = await Kernel.GetPasswordAsync("Provide your OPEN_AI_KEY");

// Your endpoint should look like the following https://YOUR_OPEN_AI_RESOURCE_NAME.openai.azure.com/
var azureOpenAIEndpoint = await Kernel.GetInputAsync("Provide the OPEN_AI_ENDPOINT");

// Enter the deployment name you chose when you deployed the model.
var deployment = await Kernel.GetInputAsync("Provide embedding deployment name");
```

### Import namesapaces and create an instance of `OpenAiClient` using the `azureOpenAIEndpoint` and the `azureOpenAIKey`


```csharp
using Azure;
using Azure.AI.OpenAI;
```


```csharp
OpenAIClient client = new (new Uri(azureOpenAIEndpoint), new AzureKeyCredential(azureOpenAIKey.GetClearTextPassword()));
```

### 1. Load the dataset
The dataset used in this example is [fine-food reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews) from Amazon. The dataset contains a total of 568,454 food reviews Amazon users left up to October 2012. We will use a subset of this dataset, consisting of 1,000 most recent reviews for illustration purposes. The reviews are in English and tend to be positive or negative. Each review has a ProductId, UserId, Score, review title (Summary) and review body (Text).

We will combine the review summary and review text into a single combined text. The model will encode this combined text and it will output a single vector embedding.

Let's load the `fine_food_reviews_1k.csv` dataset using the `value` kernel


```csharp
#!value --name dataSet --from-url https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/data/fine_food_reviews_1k.csv
```

### Loading `Microsoft.Data.Analysis` lastest package


```csharp
#i "nuget:https://pkgs.dev.azure.com/dnceng/public/_packaging/dotnet-libraries/nuget/v3/index.json"
```


<div><div><strong>Restore sources</strong><ul><li><span>https://pkgs.dev.azure.com/dnceng/public/_packaging/dotnet-libraries/nuget/v3/index.json</span></li></ul></div><div></div><div></div></div>



```csharp
#r "nuget: Microsoft.Data.Analysis, 0.21.0"
```


<div><div></div><div></div><div><strong>Installed Packages</strong><ul><li><span>Microsoft.Data.Analysis, 0.21.0</span></li></ul></div></div>



    Loading extensions from `C:\Users\dicolomb\.nuget\packages\microsoft.data.analysis\0.21.0\interactive-extensions\dotnet\Microsoft.Data.Analysis.Interactive.dll`



```csharp
using Microsoft.Data.Analysis;
```


```csharp
#!set --name dataSet --value @value:dataSet

var dataFrame = DataFrame.LoadCsvFromString(dataSet);
dataFrame.Head(3).Display();
```


<table id="table_638349565166114303"><thead><tr><th><i>index</i></th><th>Column0</th><th>Time</th><th>ProductId</th><th>UserId</th><th>Score</th><th>Summary</th><th>Text</th></tr></thead><tbody><tr><td><i><div class="dni-plaintext"><pre>0</pre></div></i></td><td><div class="dni-plaintext"><pre>0</pre></div></td><td><div class="dni-plaintext"><pre>1.3511232E+09</pre></div></td><td>B003XPF9BO</td><td>A3R7JR3FMEBXQB</td><td><div class="dni-plaintext"><pre>5</pre></div></td><td>where does one  start...and stop... with a treat like this</td><td>Wanted to save some to bring to my Chicago family but my North Carolina family ate all 4 boxes before I could pack. These are excellent...could serve to anyone</td></tr><tr><td><i><div class="dni-plaintext"><pre>1</pre></div></i></td><td><div class="dni-plaintext"><pre>1</pre></div></td><td><div class="dni-plaintext"><pre>1.3511232E+09</pre></div></td><td>B003JK537S</td><td>A3JBPC3WFUT5ZP</td><td><div class="dni-plaintext"><pre>1</pre></div></td><td>Arrived in pieces</td><td>Not pleased at all. When I opened the box, most of the rings were broken in pieces. A total waste of money.</td></tr><tr><td><i><div class="dni-plaintext"><pre>2</pre></div></i></td><td><div class="dni-plaintext"><pre>2</pre></div></td><td><div class="dni-plaintext"><pre>1.3511232E+09</pre></div></td><td>B000JMBE7M</td><td>AQX1N6A51QOKG</td><td><div class="dni-plaintext"><pre>4</pre></div></td><td>It isn&#39;t blanc mange, but isn&#39;t bad . . .</td><td>I&#39;m not sure that custard is really custard without eggs.  But this comes close.  I got it for use in a &quot;Vegan pancake&quot; recipe.  We were having houseguests who were Vegan and I wanted to make some special breakfasts while they were here.  One of the cooking/recipe sites had a recipe using this and there were lots of great reviews.  I tried the recipe and it turned out like wallpaper paste -- yuck!&lt;br /&gt;However, the  so-called custard isn&#39;t so bad.  I think it&#39;s probably just cornstarch and annatto (yellow coloring with a slight flavor).  It&#39;s fun playing with it.  You could dress it up with fruit.  Seems to come out on the thin side when you make it as directed, so I use less milk because I like my custards to set firm.  As a custard sauce it&#39;s fine.  I would say it tastes something between a pudding and a custard.&lt;br /&gt;&lt;br /&gt;If you want a really good egg-free &quot;custard&quot; get an original recipe for &quot;blanc mange.&quot;  It takes a lot longer to make, but it&#39;s certainly worth the difference.</td></tr></tbody></table><style>
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


### use tokenizer to calculate the token count


```csharp
var tokenizer = await Tokenizer.CreateAsync(TokenizerModel.ada2);
var maxTokens = 200;
var subset = dataFrame.Clone();
var tokenCount = ((IEnumerable<string>)subset["Text"]).Select(x => tokenizer.GetTokenCount(x));
subset.Columns.Add( new Int32DataFrameColumn("tokens", tokenCount));

```


```csharp
subset = subset.Filter(subset["tokens"].ElementwiseLessThanOrEqual(maxTokens));
```


```csharp
subset.Head(6).Display();
```


<table id="table_638349565237387160"><thead><tr><th><i>index</i></th><th>Column0</th><th>Time</th><th>ProductId</th><th>UserId</th><th>Score</th><th>Summary</th><th>Text</th><th>tokens</th></tr></thead><tbody><tr><td><i><div class="dni-plaintext"><pre>0</pre></div></i></td><td><div class="dni-plaintext"><pre>0</pre></div></td><td><div class="dni-plaintext"><pre>1.3511232E+09</pre></div></td><td>B003XPF9BO</td><td>A3R7JR3FMEBXQB</td><td><div class="dni-plaintext"><pre>5</pre></div></td><td>where does one  start...and stop... with a treat like this</td><td>Wanted to save some to bring to my Chicago family but my North Carolina family ate all 4 boxes before I could pack. These are excellent...could serve to anyone</td><td><div class="dni-plaintext"><pre>34</pre></div></td></tr><tr><td><i><div class="dni-plaintext"><pre>1</pre></div></i></td><td><div class="dni-plaintext"><pre>1</pre></div></td><td><div class="dni-plaintext"><pre>1.3511232E+09</pre></div></td><td>B003JK537S</td><td>A3JBPC3WFUT5ZP</td><td><div class="dni-plaintext"><pre>1</pre></div></td><td>Arrived in pieces</td><td>Not pleased at all. When I opened the box, most of the rings were broken in pieces. A total waste of money.</td><td><div class="dni-plaintext"><pre>26</pre></div></td></tr><tr><td><i><div class="dni-plaintext"><pre>2</pre></div></i></td><td><div class="dni-plaintext"><pre>4</pre></div></td><td><div class="dni-plaintext"><pre>1.3511232E+09</pre></div></td><td>B001BORBHO</td><td>A1AFOYZ9HSM2CZ</td><td><div class="dni-plaintext"><pre>5</pre></div></td><td>Happy with the product</td><td>My dog was suffering with itchy skin.  He had been eating Natural Choice brand (cheaper) since he was a puppy.  I was nervous to change foods.  The vet suggested to change foods sand see if the skin issues cleared up.  Wellness brand did the job.  My dog seems to love the food and the skin issues cleared up within a few weeks.</td><td><div class="dni-plaintext"><pre>77</pre></div></td></tr><tr><td><i><div class="dni-plaintext"><pre>3</pre></div></i></td><td><div class="dni-plaintext"><pre>5</pre></div></td><td><div class="dni-plaintext"><pre>1.3511232E+09</pre></div></td><td>B008PSM0BQ</td><td>A3OUFIMGL2K6RS</td><td><div class="dni-plaintext"><pre>4</pre></div></td><td>Good Sauce</td><td>This is a good all purpose sauce.  Has good flavor that the heat doesn&#39;t overpower.  Not really that spicy unless you use a whole bunch.  10 good drops is about enough to add a little heat to a pot of soup, but a lot more is needed if you want a lingering burn.  Heat isn&#39;t quite up to par with other products out there, (such as Spontaneous Combustion) but this has the true aged cayenne hot sauce flavor.</td><td><div class="dni-plaintext"><pre>100</pre></div></td></tr><tr><td><i><div class="dni-plaintext"><pre>4</pre></div></i></td><td><div class="dni-plaintext"><pre>6</pre></div></td><td><div class="dni-plaintext"><pre>1.3511232E+09</pre></div></td><td>B008YA1LQK</td><td>A9YEAAQVHFUTX</td><td><div class="dni-plaintext"><pre>5</pre></div></td><td>Blackcat</td><td>Great coffee!  Love all Green Mountain coffee and all the wonderful flavors.  Would and do recommend this coffee to all my friends.</td><td><div class="dni-plaintext"><pre>27</pre></div></td></tr><tr><td><i><div class="dni-plaintext"><pre>5</pre></div></i></td><td><div class="dni-plaintext"><pre>7</pre></div></td><td><div class="dni-plaintext"><pre>1.3511232E+09</pre></div></td><td>B001KP6B98</td><td>ABWCUS3HBDZRS</td><td><div class="dni-plaintext"><pre>5</pre></div></td><td>Excellent product</td><td>After scouring every store in town for orange peels and not finding anything satisfactory I turned to the online options.&lt;br /&gt;&lt;br /&gt; I received the candied orange peels today and I found exactly what I was looking for. The peels are perfect for the fruit cake I plan to bake. The peels are not crystallized with sugar which is great  I like the texture and the taste of the peels and I am gonna order another box soon.</td><td><div class="dni-plaintext"><pre>93</pre></div></td></tr></tbody></table><style>
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


### 2. Get embeddings and save them for future reuse


```csharp
using Microsoft.ML.Data;
```

Use the batch approach when calculating a lot of embeddings.


```csharp
var texts = ((IEnumerable<string>)subset["Text"]).ToArray();
var chunks = texts.Chunk(16).ToArray();
var embeddings = new List<VBuffer<float>>();

foreach(var chunk in chunks)
{
    var response = await client.GetEmbeddingsAsync(new EmbeddingsOptions(deployment, chunk));
    embeddings.AddRange( response.Value.Data.Select(e => new VBuffer<float>(1536, e.Embedding.ToArray())));
}
var embeddingsColumn = new VBufferDataFrameColumn<float>("embeddings", embeddings);
subset.Columns.Add(embeddingsColumn);
subset.Head(1).Display();
```


<table id="table_638349565874222296"><thead><tr><th><i>index</i></th><th>Column0</th><th>Time</th><th>ProductId</th><th>UserId</th><th>Score</th><th>Summary</th><th>Text</th><th>tokens</th><th>embeddings</th></tr></thead><tbody><tr><td><i><div class="dni-plaintext"><pre>0</pre></div></i></td><td><div class="dni-plaintext"><pre>0</pre></div></td><td><div class="dni-plaintext"><pre>1.3511232E+09</pre></div></td><td>B003XPF9BO</td><td>A3R7JR3FMEBXQB</td><td><div class="dni-plaintext"><pre>5</pre></div></td><td>where does one  start...and stop... with a treat like this</td><td>Wanted to save some to bring to my Chicago family but my North Carolina family ate all 4 boxes before I could pack. These are excellent...could serve to anyone</td><td><div class="dni-plaintext"><pre>34</pre></div></td><td><details open="open" class="dni-treeview"><summary><span class="dni-code-hint"><code>[ 0.0068575335, -0.028527338, 0.0065081255, -0.017594472, -0.0020066448, 0.013636695, -0.007001215, -0.03471871, -0.004424742, -0.037722964, 0.011899453, 0.0034026427, -0.017999392, 0.002989558, 0.008568651, 0.017163426, 0.025928007, -0.03450972, -0.006083612, -0.024412818 ... (more) ]</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>IsDense</td><td><div class="dni-plaintext"><pre>True</pre></div></td></tr><tr><td>Length</td><td><div class="dni-plaintext"><pre>1536</pre></div></td></tr><tr><td><i>(values)</i></td><td><div class="dni-plaintext"><pre>[ 0.0068575335, -0.028527338, 0.0065081255, -0.017594472, -0.0020066448, 0.013636695, -0.007001215, -0.03471871, -0.004424742, -0.037722964, 0.011899453, 0.0034026427, -0.017999392, 0.002989558, 0.008568651, 0.017163426, 0.025928007, -0.03450972, -0.006083612, -0.024412818 ... (more) ]</pre></div></td></tr></tbody></table></div></details></td></tr></tbody></table><style>
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


### save the data for later use


```csharp
record DataRow(string ProducIt, string UserId, int Score, string Summary, string Text, int TokenCount, float[] Embedding);

var data = subset.Rows.Select(r => new DataRow(
    r["ProductId"].ToString(), 
    r["UserId"].ToString(), 
    (r["Score"].ToString() == null ? 0 : Convert.ToInt32(r["Score"].ToString())), 
    r["Summary"].ToString(), 
    r["Text"].ToString(), 
    (int)r["tokens"], 
    ((VBuffer<float>)r["embeddings"]).DenseValues().ToArray())
    ).ToArray();

```


```csharp
using System.Text.Json;
using System.Text.Json.Serialization;
using System.IO;


var filePath = Path.Combine("..","..","..","Data","fine_food_reviews_with_embeddings_1k.json");

var options = new JsonSerializerOptions
{
    WriteIndented = true,
};

var jsonString = JsonSerializer.Serialize(data, options);
await System.IO.File.WriteAllTextAsync(filePath, jsonString);

```
