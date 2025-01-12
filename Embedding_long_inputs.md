## Embedding texts that are longer than the model's maximum context length
OpenAI's embedding models cannot embed text that exceeds a maximum length. The maximum length varies by model, and is measured by _tokens_, not string length. If you are unfamiliar with tokenization, check out [How to count tokens with tiktoken](How_to_count_tokens_with_tiktoken.ipynb).

This notebook shows how to handle texts that are longer than a model's maximum context length. We'll demonstrate using embeddings from `text-embedding-ada-002`, but the same ideas can be applied to other models and tasks. To learn more about embeddings, check out the [OpenAI Embeddings Guide](https://beta.openai.com/docs/guides/embeddings).

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
using System.Collections.Generic;
```


```csharp
OpenAIClient client = new (new Uri(azureOpenAIEndpoint), new AzureKeyCredential(azureOpenAIKey.GetClearTextPassword()));
```


```csharp
var longText = string.Join(" ", Enumerable.Repeat("AGI", 5000));
```

## Run the following cell

It will display and error like:
```
Azure.RequestFailedException: This model's maximum context length is 8191 tokens, however you requested 10000 tokens (10000 in your prompt; 0 for the completion). Please reduce your prompt; or completion length.

Status: 400 (model_error)

Content:

{

  "error": {

    "message": "This model's maximum context length is 8191 tokens, however you requested 10000 tokens (10000 in your prompt; 0 for the completion). Please reduce your prompt; or completion length.",

    "type": "invalid_request_error",

    "param": null,

    "code": null

  }

}
```

This shows that we have crossed the limit of `8191` tokens.


```csharp
var embeddingResponse = await client.GetEmbeddingsAsync(new EmbeddingsOptions(deployment, new []{ longText }));
```


    Azure.RequestFailedException: This model's maximum context length is 8191 tokens, however you requested 10000 tokens (10000 in your prompt; 0 for the completion). Please reduce your prompt; or completion length.


    Status: 400 (model_error)


    


    Content:


    {
    

      "error": {
    

        "message": "This model's maximum context length is 8191 tokens, however you requested 10000 tokens (10000 in your prompt; 0 for the completion). Please reduce your prompt; or completion length.",
    

        "type": "invalid_request_error",
    

        "param": null,
    

        "code": null
    

      }
    

    }
    

    


    


    Headers:


    Access-Control-Allow-Origin: REDACTED


    X-Content-Type-Options: REDACTED


    apim-request-id: REDACTED


    X-Request-ID: REDACTED


    ms-azureml-model-error-reason: REDACTED


    ms-azureml-model-error-statuscode: REDACTED


    x-ms-client-request-id: 89940e48-7900-40f3-ba70-68eef1b6a149


    x-ms-region: REDACTED


    Strict-Transport-Security: REDACTED


    Date: Tue, 07 Nov 2023 12:16:57 GMT


    Content-Length: 294


    Content-Type: application/json


    


       at Azure.Core.HttpPipelineExtensions.ProcessMessageAsync(HttpPipeline pipeline, HttpMessage message, RequestContext requestContext, CancellationToken cancellationToken)


       at Azure.AI.OpenAI.OpenAIClient.GetEmbeddingsAsync(EmbeddingsOptions embeddingsOptions, CancellationToken cancellationToken)


       at Submission#8.<<Initialize>>d__0.MoveNext()


    --- End of stack trace from previous location ---


       at Microsoft.CodeAnalysis.Scripting.ScriptExecutionState.RunSubmissionsAsync[TResult](ImmutableArray`1 precedingExecutors, Func`2 currentExecutor, StrongBox`1 exceptionHolderOpt, Func`2 catchExceptionOpt, CancellationToken cancellationToken)


Clearly we want to avoid these errors, particularly when handling programmatically with a large number of embeddings. Yet, we still might be faced with texts that are longer than the maximum context length. Below we describe and provide recipes for the main approaches to handling these longer texts: (1) simply truncating the text to the maximum allowed length, and (2) chunking the text and embedding each chunk individually.

## 1. Truncating the input text
The simplest solution is to truncate the input text to the maximum allowed length. Because the context length is measured in tokens, we have to first tokenize the text before truncating it. The API accepts inputs both in the form of text or tokens, so as long as you are careful that you are using the appropriate encoding, there is no need to convert the tokens back into string form. Below is an example of such a truncation function.


```csharp
var tokenizer = await Tokenizer.CreateAsync(TokenizerModel.ada2);
var truncated = tokenizer.TruncateByTokenCount(longText, 8191);
longText.Length.Display();
truncated.Length.Display();
```


<div class="dni-plaintext"><pre>19999</pre></div><style>
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



<div class="dni-plaintext"><pre>16382</pre></div><style>
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


## 2. Chunking the input text
Though truncation works, discarding potentially relevant text is a clear drawback. Another approach is to divide the input text into chunks and then embed each chunk individually. Then, we can either use the chunk embeddings separately, or combine them in some way, such as averaging (weighted by the size of each chunk).

Now we define a function that encodes a string into tokens and then breaks it up into chunks.

Finally, we can write a function that safely handles embedding requests, even when the input text is longer than the maximum context length, by chunking the input tokens and embedding each chunk individually. The `average` flag can be set to `True` to return the weighted average of the chunk embeddings, or `False` to simply return the unmodified list of chunk embeddings.


```csharp
var textChunks = tokenizer.ChunkByTokenCount( longText, 2000, true).ToList();
textChunks.Count.Display();
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



```csharp
foreach(var chunk in textChunks.Chunk(16)){
    var embeddings = await client.GetEmbeddingsAsync(new EmbeddingsOptions(deployment, chunk));
    embeddings.Value.Data.Display();
}
```


<table><thead><tr><th><i>index</i></th><th>value</th></tr></thead><tbody><tr><td>0</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.016826738, 0.0069990805, -0.0065451926, -0.026996454, -0.010584136, -0.0030555197, -0.024088942, 0.0011922775, -0.0144586265, -0.027154328, 0.018589662, 0.014826999, -0.011722145, -0.011662941, 0.0026197217, 0.031680048, 0.016103148, 0.011202476, 0.00795948, -0.014445471 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>0</pre></div></td></tr></tbody></table></div></details></td></tr><tr><td>1</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.016608136, 0.0035173707, -0.0018321727, -0.025159389, -0.00280755, 0.0010914538, -0.017783932, 0.0016501246, -0.01102978, -0.029742327, 0.017824017, 0.0124126775, -0.015899986, -0.013060702, -0.00056702155, 0.03257493, 0.015405616, 0.009052303, 0.0072485227, -0.013321248 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>1</pre></div></td></tr></tbody></table></div></details></td></tr><tr><td>2</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.016608136, 0.0035173707, -0.0018321727, -0.025159389, -0.00280755, 0.0010914538, -0.017783932, 0.0016501246, -0.01102978, -0.029742327, 0.017824017, 0.0124126775, -0.015899986, -0.013060702, -0.00056702155, 0.03257493, 0.015405616, 0.009052303, 0.0072485227, -0.013321248 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>2</pre></div></td></tr></tbody></table></div></details></td></tr><tr><td>3</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.016608136, 0.0035173707, -0.0018321727, -0.025159389, -0.00280755, 0.0010914538, -0.017783932, 0.0016501246, -0.01102978, -0.029742327, 0.017824017, 0.0124126775, -0.015899986, -0.013060702, -0.00056702155, 0.03257493, 0.015405616, 0.009052303, 0.0072485227, -0.013321248 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>3</pre></div></td></tr></tbody></table></div></details></td></tr><tr><td>4</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.016608136, 0.0035173707, -0.0018321727, -0.025159389, -0.00280755, 0.0010914538, -0.017783932, 0.0016501246, -0.01102978, -0.029742327, 0.017824017, 0.0124126775, -0.015899986, -0.013060702, -0.00056702155, 0.03257493, 0.015405616, 0.009052303, 0.0072485227, -0.013321248 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>4</pre></div></td></tr></tbody></table></div></details></td></tr></tbody></table><style>
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


In some cases, it may make sense to split chunks on paragraph boundaries or sentence boundaries to help preserve the meaning of the text.
