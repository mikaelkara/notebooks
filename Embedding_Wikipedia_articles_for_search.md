## Embedding Wikipedia articles for search
This notebook gives an example on how to get embeddings from a large dataset.
This notebook shows how we prepared a dataset of Wikipedia articles for search, used in [Question_answering_using_embeddings.ipynb](Question_answering_using_embeddings.ipynb).

## Installation
Install the Azure Open AI SDK using the below command.


```csharp
#r "nuget: Azure.AI.OpenAI, 1.0.0-beta.14"
```


<div><div></div><div></div><div><strong>Installed Packages</strong><ul><li><span>Azure.AI.OpenAI, 1.0.0-beta.14</span></li></ul></div></div>



```csharp
using Microsoft.DotNet.Interactive;
```

## Run this cell, it will prompt you for the apiKey, endPoint, and embedding deployment


```csharp
var azureOpenAIKey = await Kernel.GetPasswordAsync("Provide your OPEN_AI_KEY");

// Your endpoint should look like the following https://YOUR_OPEN_AI_RESOURCE_NAME.openai.azure.com/
var azureOpenAIEndpoint = await Kernel.GetInputAsync("Provide the OPEN_AI_ENDPOINT");

// Enter the deployment name you chose when you deployed the model.
var deployment = await Kernel.GetInputAsync("Provide EMBEDDING deployment name");
```

### Import namesapaces and create an instance of `OpenAiClient` using the `azureOpenAIEndpoint` and the `azureOpenAIKey`


```csharp
using Azure;
using Azure.AI.OpenAI;
```


```csharp
OpenAIClient client = new (new Uri(azureOpenAIEndpoint), new AzureKeyCredential(azureOpenAIKey.GetClearTextPassword()));
```

## 1. Collect documents
In this example, we'll download a few hundred Wikipedia articles related to the 2022 Winter Olympics.


```csharp
#r "nuget: CXuesong.MW.WikiClientLibrary, 0.7.5"
#r "nuget: CXuesong.MW.MwParserFromScratch, 0.2.1"
```


<div><div></div><div></div><div><strong>Installed Packages</strong><ul><li><span>CXuesong.MW.MwParserFromScratch, 0.2.1</span></li><li><span>CXuesong.MW.WikiClientLibrary, 0.7.5</span></li></ul></div></div>



```csharp
using WikiClientLibrary;
using WikiClientLibrary.Client;
using WikiClientLibrary.Sites;
using WikiClientLibrary.Pages;
using WikiClientLibrary.Pages.Queries;
using WikiClientLibrary.Pages.Queries.Properties;
using WikiClientLibrary.Generators;

var wikiClient = new WikiClient
    {
        ClientUserAgent = "WCLQuickStart/1.0 (your user name or contact information here)"
    };

var site = new WikiSite(wikiClient, await WikiSite.SearchApiEndpointAsync(wikiClient, "en.wikipedia.org"));
await site.Initialization;
        

var contentPages = new List<WikiPage>();

var generator = new CategoryMembersGenerator(site, "2022 Winter Olympics") { PaginationSize = 50, MemberTypes = CategoryMemberTypes.Page }  ;
var pages = await generator.EnumPagesAsync(PageQueryOptions.FetchContent).ToListAsync();

foreach (var page in pages)
{
    contentPages.Add(page);
}

Console.WriteLine($"Total pages: {contentPages.Count}");


```

    Total pages: 17



```csharp
using System.Text.RegularExpressions;

public bool Filter(string text){
    if (string.IsNullOrEmpty(text)){
        return false;
    }
    if(Regex.IsMatch(text, @"\{\|\s*class=\s*""wikitable")){
        return false;
    }

    return true;
}
```

Next, we'll recursively split long sections into smaller sections.
There's no perfect recipe for splitting text into sections.
Some tradeoffs include:
 - Longer sections may be better for questions that require more context
 - Longer sections may be worse for retrieval, as they may have more topics muddled together
 - Shorter sections are better for reducing costs (which are proportional to the number of tokens)
 - Overlapping sections may help prevent answers from being cut by section boundaries

 Here, we'll use a simple approach and limit sections to 1,600 tokens each, recursively halving any sections that are too long.


```csharp
#r "nuget:Microsoft.DotNet.Interactive.AIUtilities, 1.0.0-beta.24129.1"
```


```csharp
using Microsoft.DotNet.Interactive.AIUtilities;

var tokenizer = await Tokenizer.CreateAsync(TokenizerModel.ada2);
```


```csharp
using MwParserFromScratch;
using MwParserFromScratch.Nodes;

record PageBlockWithEmbeddings(string PageTitle, string Block, float[] Embedding);
var parser = new WikitextParser();
var pageBlocks = new List<PageBlockWithEmbeddings>();
foreach (var page in contentPages) {
    var content = page.Content;
    var ast = parser.Parse(content);
    
    // split page into block
    var blocks = ast.EnumChildren().OfType<Paragraph>().Where(p => Filter(p.ToPlainText())).Select(b => b.ToPlainText()).ToList();
    foreach(var block in blocks){
        //split blocks by 1600 tokens
        var blockChunks = tokenizer.ChunkByTokenCountWithOverlap(block, 1600, 10, true).ToArray();

        // generate embeddings
        foreach(var chunk in blockChunks.Chunk(16)) {
            var response = await client.GetEmbeddingsAsync(new EmbeddingsOptions(deployment, chunk));
            foreach( var embeddingItem in response.Value.Data){
                    var embedding = embeddingItem.Embedding.ToArray();
                    var blockWithEmbeddings = new PageBlockWithEmbeddings(page.Title, chunk[embeddingItem.Index], embedding);
                    pageBlocks.Add(blockWithEmbeddings);
                }
        }
    }
}

```


```csharp
pageBlocks.Count.Display();
```


<div class="dni-plaintext"><pre>389</pre></div><style>
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



```csharp
using System.Text.Json;
using System.IO;

var filePath = Path.Combine("..","..","..","Data","wikipedia_embeddings.json");

var options = new JsonSerializerOptions
{
    WriteIndented = true,
};

var jsonString = JsonSerializer.Serialize(pageBlocks, options);
await System.IO.File.WriteAllTextAsync(filePath, jsonString);
```
