## Get embeddings from dataset
This notebook gives an example on how to get embeddings text.

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
var sources = Enumerable.Range(0, 8).Select(i => $"This is the {i}th document.").ToList();
```

### Create text embeddings using the `deployment`


```csharp
foreach (var source in sources)
{
   var embeddings = await client.GetEmbeddingsAsync(new EmbeddingsOptions(deployment, new []{ source }));
    embeddings.Value.Data[0].Display();
}
```


<details open="open" class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ 0.003541318, -0.0041777287, -0.004222209, -0.012919823, 0.0069594597, 0.012919823, -0.010442611, 0.0038082, -0.020693615, -0.018709108, 0.017928991, 0.030492973, -0.0013814562, 0.013008784, -0.007452165, 0.025579607, 0.010832669, -0.018312206, -0.0061793434, 0.004324856 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>0</pre></div></td></tr></tbody></table></div></details><style>
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



<details open="open" class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.00032866732, 0.011613191, -0.017493287, -0.02527105, 0.016691456, -0.004166182, -0.011386005, -0.0013439028, -0.0010599208, -0.018402029, 0.009936027, 0.03001522, -0.0069358414, 0.019805234, -0.0065449486, 0.0130230775, 0.024081668, -0.026607437, 0.0024789954, 0.0012904473 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>0</pre></div></td></tr></tbody></table></div></details><style>
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



<details open="open" class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.0069043813, 0.011252084, -0.009558875, -0.019589959, 0.0072855223, 0.018146347, -0.019158226, -0.007690273, 0.0005830947, -0.0025954673, 0.0036596258, 0.019940743, -0.0064422903, 0.0098219635, 0.009133887, 0.011157642, 0.021101031, -0.00948467, 0.017134469, -0.0015852756 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>0</pre></div></td></tr></tbody></table></div></details><style>
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



<details open="open" class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.006491534, 0.0074753026, -0.0072143027, -0.024065522, 0.011959146, 0.011490684, -0.01856445, -0.01699845, -0.00617365, 0.00081018696, 0.013719222, 0.015258451, -0.006123457, 0.0006077448, -0.0014789989, 0.019180141, 0.0208666, -0.02022414, 0.008586225, -0.00839884 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>0</pre></div></td></tr></tbody></table></div></details><style>
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



<details open="open" class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.00646148, 0.006511466, -0.017621614, -0.033403754, 0.0056850365, -0.00899742, -0.018354736, -0.018328078, -0.0118032815, 0.0015145657, 0.010796904, 0.009730543, -0.020567436, 0.0149957, -0.007831087, 0.021940375, 0.022433566, -0.026245806, 0.012256485, -0.009423964 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>0</pre></div></td></tr></tbody></table></div></details><style>
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



<details open="open" class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.021144673, 0.009870633, -0.0070905495, -0.0359005, 0.0052961926, 0.00646904, -0.022307497, -0.02115804, -0.017776495, 0.0027934492, 0.004243637, 0.010224826, -0.014956314, 0.018377956, -0.017081475, 0.021425355, 0.021545647, -0.027506787, 0.009736975, -0.008921662 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>0</pre></div></td></tr></tbody></table></div></details><style>
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



<details open="open" class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.006183072, 0.0080613885, -0.006550714, -0.038528893, 0.008175024, 0.007386264, -0.020855334, -0.005504605, -0.008195077, 0.005287362, 0.011443696, 0.0071991007, -0.015708344, 0.015039904, -0.017352708, 0.019117389, 0.019932887, -0.024692181, 0.012680311, -0.0037098431 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>0</pre></div></td></tr></tbody></table></div></details><style>
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



<details open="open" class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.0034153257, 0.0040983907, -0.014244893, -0.020133842, 0.00043064603, -0.0029925548, -0.019895101, -0.0057795267, -0.01720263, -4.5178458E-05, 0.0121161165, 0.0169241, -0.009967445, 0.01240128, -0.0034518, 0.020054262, 0.014417317, -0.018820766, 0.0066118054, -0.001461461 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>0</pre></div></td></tr></tbody></table></div></details><style>
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


### Generate a single request with a batch of inputs, this is more efficient and saves api calls
Use `Linq` extension [`Chuck`](https://learn.microsoft.com/en-us/dotnet/api/system.linq.enumerable.chunk?view=net-7.0) to split the source collection into chunks of at most size 4 and perform back requests.


```csharp
foreach(var chunk in sources.Chunk(4))
{
   var embeddings = await client.GetEmbeddingsAsync(new EmbeddingsOptions(deployment, chunk));
    embeddings.Value.Data.Display();
}
```


<table><thead><tr><th><i>index</i></th><th>value</th></tr></thead><tbody><tr><td>0</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ 0.003541318, -0.0041777287, -0.004222209, -0.012919823, 0.0069594597, 0.012919823, -0.010442611, 0.0038082, -0.020693615, -0.018709108, 0.017928991, 0.030492973, -0.0013814562, 0.013008784, -0.007452165, 0.025579607, 0.010832669, -0.018312206, -0.0061793434, 0.004324856 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>0</pre></div></td></tr></tbody></table></div></details></td></tr><tr><td>1</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.00034386062, 0.011571421, -0.017504113, -0.025254026, 0.016635587, -0.004132173, -0.011324226, -0.0013829585, -0.0010205165, -0.018452808, 0.009947948, 0.030037591, -0.0069415164, 0.019829087, -0.0065506804, 0.013007827, 0.024078177, -0.026630303, 0.0024903275, 0.001331181 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>1</pre></div></td></tr></tbody></table></div></details></td></tr><tr><td>2</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.0069416147, 0.011252296, -0.009613023, -0.019549854, 0.007245184, 0.018092722, -0.019199062, -0.0076701804, 0.00056961377, -0.0026477976, 0.0036900516, 0.01994112, -0.006459277, 0.009835641, 0.009127312, 0.011157853, 0.021074446, -0.009525325, 0.017161775, -0.0016148193 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>2</pre></div></td></tr></tbody></table></div></details></td></tr><tr><td>3</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.0065574623, 0.0075181522, -0.0072101955, -0.024047375, 0.012043771, 0.011494806, -0.018584497, -0.017098272, -0.0060955277, 0.00092303223, 0.013724143, 0.015290703, -0.0061323484, 0.00064143626, -0.0015448028, 0.01916024, 0.020860696, -0.020284953, 0.008555831, -0.00835499 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>3</pre></div></td></tr></tbody></table></div></details></td></tr></tbody></table><style>
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



<table><thead><tr><th><i>index</i></th><th>value</th></tr></thead><tbody><tr><td>0</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.00646148, 0.006511466, -0.017621614, -0.033403754, 0.0056850365, -0.00899742, -0.018354736, -0.018328078, -0.0118032815, 0.0015145657, 0.010796904, 0.009730543, -0.020567436, 0.0149957, -0.007831087, 0.021940375, 0.022433566, -0.026245806, 0.012256485, -0.009423964 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>0</pre></div></td></tr></tbody></table></div></details></td></tr><tr><td>1</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.021176927, 0.009873658, -0.007128007, -0.03591393, 0.0053075925, 0.0064900266, -0.022285877, -0.021123484, -0.017743193, 0.0027707035, 0.0042353855, 0.010254443, -0.01500422, 0.018384513, -0.017061789, 0.021457504, 0.021537669, -0.02746988, 0.009726689, -0.008911679 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>1</pre></div></td></tr></tbody></table></div></details></td></tr><tr><td>2</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.006207697, 0.008083044, -0.006541983, -0.038509786, 0.0081432145, 0.0074010994, -0.020832725, -0.0054956675, -0.008243501, 0.005224895, 0.01145265, 0.0072339564, -0.015791686, 0.014989399, -0.017342774, 0.019027578, 0.019910093, -0.024683703, 0.012669452, -0.0036838353 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>2</pre></div></td></tr></tbody></table></div></details></td></tr><tr><td>3</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>Azure.AI.OpenAI.EmbeddingItem</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.0034319635, 0.0041415677, -0.014218609, -0.020107659, 0.0004928267, -0.0030340548, -0.01985565, -0.0057862573, -0.017189661, -3.0102217E-05, 0.0120765325, 0.016950916, -0.009974248, 0.012454546, -0.003524809, 0.020094395, 0.014417563, -0.018821087, 0.0065920227, -0.0014308138 ... (1516 more) ]</pre></div></td></tr><tr><td>Index</td><td><div class="dni-plaintext"><pre>3</pre></div></td></tr></tbody></table></div></details></td></tr></tbody></table><style>
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

