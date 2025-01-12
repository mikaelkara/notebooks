## Comparing text using embeddings

## Installation
Install the Azure Open AI SDK using the below command.


```csharp
#r "nuget: Azure.AI.OpenAI, 1.0.0-beta.14"
#r "nuget:Microsoft.DotNet.Interactive.AIUtilities, 1.0.0-beta.24129.1"
```


<div><div></div><div></div><div><strong>Installed Packages</strong><ul><li><span>Azure.AI.OpenAI, 1.0.0-beta.14</span></li><li><span>Microsoft.DotNet.Interactive.AIUtilities, 1.0.0-beta.24129.1</span></li></ul></div></div>



```csharp
using Microsoft.DotNet.Interactive;
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

### Create some text


```csharp
var firstSentence = "The quick brown fox jumps over the lazy dog";
var secondSentence = "The quick fox jumps over the lazy dog";
```

### Create text embeddings using the `deployment`


```csharp
var firstEmbeddings = (await client.GetEmbeddingsAsync(new EmbeddingsOptions(deployment, new []{ firstSentence }))).Value.Data[0].Embedding.ToArray();
var secondEmbeddings = (await client.GetEmbeddingsAsync(new EmbeddingsOptions(deployment, new []{ secondSentence }))).Value.Data[0].Embedding.ToArray();
```


```csharp
firstEmbeddings.Take(7).Display();
secondEmbeddings.Take(7).Display();
```


<div class="dni-plaintext"><pre>[ -0.0035237537, 0.008311155, -0.014132736, -0.0045390725, -0.015415244, 0.018583793, -0.02041954 ]</pre></div><style>
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



<div class="dni-plaintext"><pre>[ -0.003907447, 0.008527633, -0.013624029, -0.006285345, -0.0026570011, 0.016840814, -0.019313322 ]</pre></div><style>
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


### calculate Cosine Similarity


```csharp
using Microsoft.DotNet.Interactive.AIUtilities;

var similarityComparer = new CosineSimilarityComparer<float[]>(f => f);
var similarity = similarityComparer.Score(firstEmbeddings, secondEmbeddings);
similarity.Display();
```


<div class="dni-plaintext"><pre>0.9821579</pre></div><style>
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

