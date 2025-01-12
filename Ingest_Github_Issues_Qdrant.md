# Ingest GitHub Issues into Qdrant

This sample shows how to get started loading and querying dotnet/runtime repo GitHub issue embeddings using Azure OpenAI and the Qdrant SDK

## Install packages


```python
#r "nuget: Azure.AI.OpenAI, 1.0.0-beta.14"
#r "nuget: Qdrant.Client, 1.6.0-alpha.1"
```


<div><div></div><div></div><div><strong>Installed Packages</strong><ul><li><span>Azure.AI.OpenAI, 1.0.0-beta.14</span></li><li><span>Qdrant.Client, 1.6.0-alpha.1</span></li></ul></div></div>



```python
#r "nuget: Octokit, 9.0.0"
#r "nuget: Octokit.Reactive, 9.0.0"
```


<div><div></div><div></div><div><strong>Installed Packages</strong><ul><li><span>Octokit, 9.0.0</span></li><li><span>Octokit.Reactive, 9.0.0</span></li></ul></div></div>



```python
#r "nuget:Microsoft.DotNet.Interactive.AIUtilities, 1.0.0-beta.24129.1"
```

## Add using statements


```python
using Azure;
using Azure.AI.OpenAI;
using Microsoft.DotNet.Interactive;
using Microsoft.DotNet.Interactive.AIUtilities;
using Octokit;
```

## Configure Azure OpenAI credentials


```python
var azureOpenAIKey = await Kernel.GetPasswordAsync("Provide your OPEN_AI_KEY");
var azureOpenAIEndpoint = await Kernel.GetInputAsync("Provide the OPEN_AI_ENDPOINT");
var embeddingDeployment = await Kernel.GetInputAsync("Provide embedding name");
```

## Configure GitHub credentials 

You will need access token with rights to query and update issues.


```python
var githubKey = await Kernel.GetPasswordAsync("Provide your Github api key");
var repoName = await Kernel.GetInputAsync("Provide repo");
var org = await Kernel.GetInputAsync("Provide org");
```

## Configure OpenAI client


```python
OpenAIClient openAIClient = new (new System.Uri(azureOpenAIEndpoint), new AzureKeyCredential(azureOpenAIKey.GetClearTextPassword()));
```

## Configure GitHub client


```python
var options = new ApiOptions();
var gitHubClient = new GitHubClient(new ProductHeaderValue("notebook"));

if (!string.IsNullOrEmpty(githubKey.GetClearTextPassword())) {
    Console.WriteLine("Using GitHub API token");
    var tokenAuth = new Credentials(githubKey.GetClearTextPassword());
    gitHubClient.Credentials = tokenAuth;
} else {
    Console.WriteLine("Using anonymous GitHub API");
}
```

    Using GitHub API token


## Download data from GitHub

### Get labels from the repository


```python
var allLabels = await gitHubClient.Issue.Labels.GetAllForRepository(org, repoName);
```


```python
var areaLabels = allLabels.Where(label => label.Name.StartsWith("area-", StringComparison.OrdinalIgnoreCase)).ToList();
```

### Get all issues from the repository


```python
var allIssues = new List<Issue>();
```


```python
foreach(var label in areaLabels)
{
    var request = new RepositoryIssueRequest
    {
        Filter = IssueFilter.All
    };
    
    request.Labels.Add(label.Name);

    var apiOptions = new ApiOptions
    {
        PageSize = 50,
        PageCount = 1
    };

    var issues = await gitHubClient.Issue.GetAllForRepository(org, repoName, request, apiOptions);
    
    allIssues.AddRange(issues);
}
```


```python
allIssues.Count()
```


<div class="dni-plaintext"><pre>4138</pre></div><style>
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



```python
public record GitHubIssue(string Title, string Text, string Area, string Url);
```


```python
var dataCollection = 
    allIssues
        .DistinctBy(issue => issue.Id)
        .Select(issue => 
            new GitHubIssue(
                issue.Title,
                issue.Body,
                issue.Labels?.Where(l => 
                    l.Name.StartsWith("area-",StringComparison.OrdinalIgnoreCase))
                        .FirstOrDefault()?
                        .Name?
                        .Replace("area-",string.Empty)?
                        .Replace("-"," "),
                issue.HtmlUrl));
```

## Helper functions to save and load to disk


```python
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

public async Task SaveIssuesToFileAsync(IEnumerable<GitHubIssue> data, string fileName)
{ 
    var filePath = Path.Combine("..","..","..","Data",fileName);
    var issuesJson = JsonSerializer.Serialize(data,new JsonSerializerOptions(JsonSerializerOptions.Default){WriteIndented=true});
    await File.WriteAllTextAsync(filePath, issuesJson);
}

public async Task<GitHubIssue[]> LoadIssuesFromFileAsync(string fileName)
{
    var filePath = Path.Combine("..","..","..","Data",fileName);
    var text = await File.ReadAllTextAsync(filePath);
    return JsonSerializer.Deserialize<GitHubIssue[]>(text);
}
```


```python
await SaveIssuesToFileAsync(dataCollection, "issues.json");
```

## Naive Search


```python
var dataCollection = await LoadIssuesFromFileAsync("issues.json");
```


```python
dataCollection.Count()
```


<div class="dni-plaintext"><pre>4127</pre></div><style>
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



```python
public string[] NaiveSearch(string query, IEnumerable<GitHubIssue> data,int resultLimit = 1)
{
    return data
            .Where(d => d.Text?.Contains(query)==true)
            .Select(d => d.Text)
            .Take(resultLimit)
            .ToArray();
}
```


```python
NaiveSearch("What are the latest issues for AOT",dataCollection).Display();
```


<div class="dni-plaintext"><pre>[  ]</pre></div><style>
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


## Chunk issues


```python
var dataCollection = await LoadIssuesFromFileAsync("issues.json");
```


```python
dataCollection.Count()
```


<div class="dni-plaintext"><pre>4127</pre></div><style>
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


### Initialize collection of issues with chunks


```python
public record TextWithEmbedding(string Text, float[] Embedding);
public record IssueWithChunks(GitHubIssue Issue,List<TextWithEmbedding> Chunks);
```


```python
var issuesWithChunksCollection = 
    dataCollection
        .Select(issue => new IssueWithChunks(issue, new ()))
        .ToArray();
```

### Helper functions to save and load chunks to disk


```python
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

public async Task SaveIssuesWithChunksToFileAsync(IEnumerable<IssueWithChunks> data, string fileName)
{ 
    var filePath = Path.Combine("..","..","..","Data",fileName);
    var issuesJson = JsonSerializer.Serialize(data,new JsonSerializerOptions(JsonSerializerOptions.Default){WriteIndented=true});
    await File.WriteAllTextAsync(filePath, issuesJson);
}

public async Task<IssueWithChunks[]> LoadIssuesWithChunksFromFileAsync(string fileName)
{
    var filePath = Path.Combine("..","..","..","Data",fileName);
    var text = await File.ReadAllTextAsync(filePath);
    return JsonSerializer.Deserialize<IssueWithChunks[]>(text);
}
```

### Chunk data and generate embeddings


```python
var tokenizer = await Tokenizer.CreateAsync(TokenizerModel.ada2);

var counter = 0;

foreach(var item in issuesWithChunksCollection.Take(100))
{
    var fullText = item.Issue.Text;
    if(string.IsNullOrWhiteSpace(fullText))
        continue;

    var chunks = 
        tokenizer
            .ChunkByTokenCountWithOverlap(fullText, 3000, 50)
            .Select(t => 
$"""
Title: {item.Issue.Title}
Area: {item.Issue.Area}

{t}
""")
            .Chunk(16)
            .ToArray();

    foreach(var chunk in chunks)
    {
        
        var embeddingResponse = await openAIClient.GetEmbeddingsAsync(new EmbeddingsOptions(embeddingDeployment,chunk));
        item.Chunks.AddRange(
            embeddingResponse.Value.Data.Select(d => 
                new TextWithEmbedding(chunk[d.Index],d.Embedding.ToArray())));
    }

    if(counter % 50 == 0)
        await SaveIssuesWithChunksToFileAsync(issuesWithChunksCollection, "areaIssuesWithEmbeddingsSubset.json");
    counter++;
}

await SaveIssuesWithChunksToFileAsync(issuesWithChunksCollection, "areaIssuesWithEmbeddingsSubset.json");
```


```python
issuesWithChunksCollection.Take(5).DisplayTable();
```


<table><thead><tr><td><span>Issue</span></td><td><span>Chunks</span></td></tr></thead><tbody><tr><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>GitHubIssue { Title = DependencyContext RuntimeGraph expected behavior., Text = To my knowledge this is not a defect but I am finding a discrepancy in the behavior of DependencyContext&#39;s [RuntimeGraph](https://learn.microsoft.com/en-us/dotnet/api/microsoft.extensions.dependencymodel.dependencycontex...</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Title</td><td><div class="dni-plaintext"><pre>DependencyContext RuntimeGraph expected behavior.</pre></div></td></tr><tr><td>Text</td><td><div class="dni-plaintext"><pre>To my knowledge this is not a defect but I am finding a discrepancy in the behavior of DependencyContext&#39;s [RuntimeGraph](https://learn.microsoft.com/en-us/dotnet/api/microsoft.extensions.dependencymodel.dependencycontext.runtimegraph?view=dotnet-plat-ext-7.0) property.

I&#39;ve actually asked Bing&#39;s AI twice and received two different answers on this, but where does the information for RuntimeGraph come from? [Official documentation](https://learn.microsoft.com/en-us/dotnet/api/microsoft.extensions.dependencymodel.dependencycontext?view=dotnet-plat-ext-7.0) says &quot;ApplicationName.deps.json&quot; but AI also said ApplicationName.runtimeconfig.json. However, it can&#39;t strictly be this as I&#39;ve created an application that simply prints the RuntimeGraph to console and it differs depending on which platform I run it on. 

The application I&#39;ve created uses default csproj options for a dotnet 6.0 Exe. I am compiling it in WSL on an Oracle Linux 8.7 system as RuntimeGetter and launching it against RuntimeGetter.dll. I do not set the RuntimeIdentifier.

The code is simple, it loads the Assembly using `LoadFromAssemblyPath` using arg[0] and then does:
```
            string retVal = &quot;&quot;;

            foreach (var item in DependencyCtx.RuntimeGraph)
            {
                retVal += &quot;Runtime: &quot; + item.Runtime;
                retVal += &quot; | Fallbacks: &quot;;
                foreach (var item2 in item.Fallbacks)
                {
                    retVal += item2 + &quot; &quot;;
                }
            }

            return retVal;
```

When run on Oracle Linux 8.7 I see the following:

&gt;  Runtime: ol.8.7-x64 | Fallbacks: ol.8.7 ol.8.6-x64 rhel.8.7-x64 ol.8.6 rhel.8.7 ol.8.5-x64 rhel.8.6-x64 ol.8.5 rhel.8.6 ol.8.4-x64 rhel.8.5-x64 ol.8.4 rhel.8.5 ol.8.3-x64 rhel.8.4-x64 ol.8.3 rhel.8.4 ol.8.2-x64 rhel.8.3-x64 ol.8.2 rhel.8.3 ol.8.1-x64 rhel.8.2-x64 ol.8.1 rhel.8.2 ol.8.0-x64 rhel.8.1-x64 ol.8.0 rhel.8.1 ol.8-x64 rhel.8.0-x64 ol.8 rhel.8.0 ol-x64 rhel.8-x64 ol rhel.8 rhel-x64 rhel linux-x64 linux unix-x64 unix any base

To me, this makes sense as the application was published from an OL8.7 system and is running on an OL8.7 system. I then copy it to an Amazon Linux 2 system and get the following:

&gt; Runtime: alpine-x64 | Fallbacks: alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.10-x64 | Fallbacks: alpine.3.10 alpine.3.9-x64 alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.11-x64 | Fallbacks: alpine.3.11 alpine.3.10-x64 alpine.3.10 alpine.3.9-x64 alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.12-x64 | Fallbacks: alpine.3.12 alpine.3.11-x64 alpine.3.11 alpine.3.10-x64 alpine.3.10 alpine.3.9-x64 alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.13-x64 | Fallbacks: alpine.3.13 alpine.3.12-x64 alpine.3.12 alpine.3.11-x64 alpine.3.11 alpine.3.10-x64 alpine.3.10 alpine.3.9-x64 alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.14-x64 | Fallbacks: alpine.3.14 alpine.3.13-x64 alpine.3.13 alpine.3.12-x64 alpine.3.12 alpine.3.11-x64 alpine.3.11 alpine.3.10-x64 alpine.3.10 alpine.3.9-x64 alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.15-x64 | Fallbacks: alpine.3.15 alpine.3.14-x64 alpine.3.14 alpine.3.13-x64 alpine.3.13 alpine.3.12-x64 alpine.3.12 alpine.3.11-x64 alpine.3.11 alpine.3.10-x64 alpine.3.10 alpine.3.9-x64 alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.16-x64 | Fallbacks: alpine.3.16 alpine.3.15-x64 alpine.3.15 alpine.3.14-x64 alpine.3.14 alpine.3.13-x64 alpine.3.13 alpine.3.12-x64 alpine.3.12 alpine.3.11-x64 alpine.3.11 alpine.3.10-x64 alpine.3.10 alpine.3.9-x64 alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.17-x64 | Fallbacks: alpine.3.17 alpine.3.16-x64 alpine.3.16 alpine.3.15-x64 alpine.3.15 alpine.3.14-x64 alpine.3.14 alpine.3.13-x64 alpine.3.13 alpine.3.12-x64 alpine.3.12 alpine.3.11-x64 alpine.3.11 alpine.3.10-x64 alpine.3.10 alpine.3.9-x64 alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.18-x64 | Fallbacks: alpine.3.18 alpine.3.17-x64 alpine.3.17 alpine.3.16-x64 alpine.3.16 alpine.3.15-x64 alpine.3.15 alpine.3.14-x64 alpine.3.14 alpine.3.13-x64 alpine.3.13 alpine.3.12-x64 alpine.3.12 alpine.3.11-x64 alpine.3.11 alpine.3.10-x64 alpine.3.10 alpine.3.9-x64 alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.6-x64 | Fallbacks: alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.7-x64 | Fallbacks: alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.8-x64 | Fallbacks: alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.9-x64 | Fallbacks: alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: android-x64 | Fallbacks: android linux-x64 linux unix-x64 unix any base Runtime: android.21-x64 | Fallbacks: android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.22-x64 | Fallbacks: android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.23-x64 | Fallbacks: android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.24-x64 | Fallbacks: android.24 android.23-x64 android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.25-x64 | Fallbacks: android.25 android.24-x64 android.24 android.23-x64 android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.26-x64 | Fallbacks: android.26 android.25-x64 android.25 android.24-x64 android.24 android.23-x64 android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.27-x64 | Fallbacks: android.27 android.26-x64 android.26 android.25-x64 android.25 android.24-x64 android.24 android.23-x64 android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.28-x64 | Fallbacks: android.28 android.27-x64 android.27 android.26-x64 android.26 android.25-x64 android.25 android.24-x64 android.24 android.23-x64 android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.29-x64 | Fallbacks: android.29 android.28-x64 android.28 android.27-x64 android.27 android.26-x64 android.26 android.25-x64 android.25 android.24-x64 android.24 android.23-x64 android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.30-x64 | Fallbacks: android.30 android.29-x64 android.29 android.28-x64 android.28 android.27-x64 android.27 android.26-x64 android.26 android.25-x64 android.25 android.24-x64 android.24 android.23-x64 android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.31-x64 | Fallbacks: android.31 android.30-x64 android.30 android.29-x64 android.29 android.28-x64 android.28 android.27-x64 android.27 android.26-x64 android.26 android.25-x64 android.25 android.24-x64 android.24 android.23-x64 android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.32-x64 | Fallbacks: android.32 android.31-x64 android.31 android.30-x64 android.30 android.29-x64 android.29 android.28-x64 android.28 android.27-x64 android.27 android.26-x64 android.26 android.25-x64 android.25 android.24-x64 android.24 android.23-x64 android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: arch-x64 | Fallbacks: arch linux-x64 linux unix-x64 unix any base Runtime: centos-x64 | Fallbacks: centos rhel-x64 rhel linux-x64 linux unix-x64 unix any base Runtime: centos.7-x64 | Fallbacks: centos.7 centos-x64 rhel.7-x64 centos rhel.7 rhel-x64 rhel linux-x64 linux unix-x64 unix any base Runtime: centos.8-x64 | Fallbacks: centos.8 centos-x64 rhel.8-x64 centos rhel.8 rhel-x64 rhel linux-x64 linux unix-x64 unix any base Runtime: centos.9-x64 | Fallbacks: centos.9 centos-x64 rhel.9-x64 centos rhel.9 rhel-x64 rhel linux-x64 linux unix-x64 unix any base {SNIP} (you get the idea)...

On Rocky Linux 8.8, I also get a single entry stating `rocky.8-x64` (note: no minor build number, unlike OL) with fallbacks for Rocky.

It seems to be about every known linux distribution from https://github.com/dotnet/runtime/blob/main/src/libraries/Microsoft.NETCore.Platforms/src/runtime.json.

Which of these is the correct behavior?
Does something else need to be done during build/publish time to change this?</pre></div></td></tr><tr><td>Area</td><td><div class="dni-plaintext"><pre>AssemblyLoader coreclr</pre></div></td></tr><tr><td>Url</td><td><div class="dni-plaintext"><pre>https://github.com/dotnet/runtime/issues/94305</pre></div></td></tr></tbody></table></div></details></td><td><table><thead><tr><th><i>index</i></th><th>value</th></tr></thead><tbody><tr><td>0</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>TextWithEmbedding { Text = Title: DependencyContext RuntimeGraph expected behavior.\nArea: AssemblyLoader coreclr\n\nTo my knowledge this is not a defect but I am finding a discrepancy in the behavior of DependencyContext&#39;s [RuntimeGraph](https://learn.microsoft.com/en-us/dotnet/api/microsoft.extens...</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Text</td><td><div class="dni-plaintext"><pre>Title: DependencyContext RuntimeGraph expected behavior.
Area: AssemblyLoader coreclr

To my knowledge this is not a defect but I am finding a discrepancy in the behavior of DependencyContext&#39;s [RuntimeGraph](https://learn.microsoft.com/en-us/dotnet/api/microsoft.extensions.dependencymodel.dependencycontext.runtimegraph?view=dotnet-plat-ext-7.0) property.

I&#39;ve actually asked Bing&#39;s AI twice and received two different answers on this, but where does the information for RuntimeGraph come from? [Official documentation](https://learn.microsoft.com/en-us/dotnet/api/microsoft.extensions.dependencymodel.dependencycontext?view=dotnet-plat-ext-7.0) says &quot;ApplicationName.deps.json&quot; but AI also said ApplicationName.runtimeconfig.json. However, it can&#39;t strictly be this as I&#39;ve created an application that simply prints the RuntimeGraph to console and it differs depending on which platform I run it on. 

The application I&#39;ve created uses default csproj options for a dotnet 6.0 Exe. I am compiling it in WSL on an Oracle Linux 8.7 system as RuntimeGetter and launching it against RuntimeGetter.dll. I do not set the RuntimeIdentifier.

The code is simple, it loads the Assembly using `LoadFromAssemblyPath` using arg[0] and then does:
```
            string retVal = &quot;&quot;;

            foreach (var item in DependencyCtx.RuntimeGraph)
            {
                retVal += &quot;Runtime: &quot; + item.Runtime;
                retVal += &quot; | Fallbacks: &quot;;
                foreach (var item2 in item.Fallbacks)
                {
                    retVal += item2 + &quot; &quot;;
                }
            }

            return retVal;
```

When run on Oracle Linux 8.7 I see the following:

&gt;  Runtime: ol.8.7-x64 | Fallbacks: ol.8.7 ol.8.6-x64 rhel.8.7-x64 ol.8.6 rhel.8.7 ol.8.5-x64 rhel.8.6-x64 ol.8.5 rhel.8.6 ol.8.4-x64 rhel.8.5-x64 ol.8.4 rhel.8.5 ol.8.3-x64 rhel.8.4-x64 ol.8.3 rhel.8.4 ol.8.2-x64 rhel.8.3-x64 ol.8.2 rhel.8.3 ol.8.1-x64 rhel.8.2-x64 ol.8.1 rhel.8.2 ol.8.0-x64 rhel.8.1-x64 ol.8.0 rhel.8.1 ol.8-x64 rhel.8.0-x64 ol.8 rhel.8.0 ol-x64 rhel.8-x64 ol rhel.8 rhel-x64 rhel linux-x64 linux unix-x64 unix any base

To me, this makes sense as the application was published from an OL8.7 system and is running on an OL8.7 system. I then copy it to an Amazon Linux 2 system and get the following:

&gt; Runtime: alpine-x64 | Fallbacks: alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.10-x64 | Fallbacks: alpine.3.10 alpine.3.9-x64 alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.11-x64 | Fallbacks: alpine.3.11 alpine.3.10-x64 alpine.3.10 alpine.3.9-x64 alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.12-x64 | Fallbacks: alpine.3.12 alpine.3.11-x64 alpine.3.11 alpine.3.10-x64 alpine.3.10 alpine.3.9-x64 alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.13-x64 | Fallbacks: alpine.3.13 alpine.3.12-x64 alpine.3.12 alpine.3.11-x64 alpine.3.11 alpine.3.10-x64 alpine.3.10 alpine.3.9-x64 alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.14-x64 | Fallbacks: alpine.3.14 alpine.3.13-x64 alpine.3.13 alpine.3.12-x64 alpine.3.12 alpine.3.11-x64 alpine.3.11 alpine.3.10-x64 alpine.3.10 alpine.3.9-x64 alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.15-x64 | Fallbacks: alpine.3.15 alpine.3.14-x64 alpine.3.14 alpine.3.13-x64 alpine.3.13 alpine.3.12-x64 alpine.3.12 alpine.3.11-x64 alpine.3.11 alpine.3.10-x64 alpine.3.10 alpine.3.9-x64 alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.16-x64 | Fallbacks: alpine.3.16 alpine.3.15-x64 alpine.3.15 alpine.3.14-x64 alpine.3.14 alpine.3.13-x64 alpine.3.13 alpine.3.12-x64 alpine.3.12 alpine.3.11-x64 alpine.3.11 alpine.3.10-x64 alpine.3.10 alpine.3.9-x64 alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.17-x64 | Fallbacks: alpine.3.17 alpine.3.16-x64 alpine.3.16 alpine.3.15-x64 alpine.3.15 alpine.3.14-x64 alpine.3.14 alpine.3.13-x64 alpine.3.13 alpine.3.12-x64 alpine.3.12 alpine.3.11-x64 alpine.3.11 alpine.3.10-x64 alpine.3.10 alpine.3.9-x64 alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.18-x64 | Fallbacks: alpine.3.18 alpine.3.17-x64 alpine.3.17 alpine.3.16-x64 alpine.3.16 alpine.3.15-x64 alpine.3.15 alpine.3.14-x64 alpine.3.14 alpine.3.13-x64 alpine.3.13 alpine.3.12-x64 alpine.3.12 alpine.3.11-x64 alpine.3.11 alpine.3.10-x64 alpine.3.10 alpine.3.9-x64 alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.6-x64 | Fallbacks: alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.7-x64 | Fallbacks: alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.8-x64 | Fallbacks: alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: alpine.3.9-x64 | Fallbacks: alpine.3.9 alpine.3.8-x64 alpine.3.8 alpine.3.7-x64 alpine.3.7 alpine.3.6-x64 alpine.3.6 alpine-x64 alpine linux-musl-x64 linux-musl linux-x64 linux unix-x64 unix any base Runtime: android-x64 | Fallbacks: android linux-x64 linux unix-x64 unix any base Runtime: android.21-x64 | Fallbacks: android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.22-x64 | Fallbacks: android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.23-x64 | Fallbacks: android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.24-x64 | Fallbacks: android.24 android.23-x64 android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.25-x64 | Fallbacks: android.25 android.24-x64 android.24 android.23-x64 android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.26-x64 | Fallbacks: android.26 android.25-x64 android.25 android.24-x64 android.24 android.23-x64 android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.27-x64 | Fallbacks: android.27 android.26-x64 android.26 android.25-x64 android.25 android.24-x64 android.24 android.23-x64 android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.28-x64 | Fallbacks: android.28 android.27-x64 android.27 android.26-x64 android.26 android.25-x64 android.25 android.24-x64 android.24 android.23-x64 android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.29-x64 | Fallbacks: android.29 android.28-x64 android.28 android.27-x64 android.27 android.26-x64 android.26 android.25-x64 android.25 android.24-x64 android.24 android.23-x64 android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.30-x64 | Fallbacks: android.30 android.29-x64 android.29 android.28-x64 android.28 android.27-x64 android.27 android.26-x</pre></div></td></tr><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ 0.019324131, 0.011482557, 0.0019214335, -0.021945072, 0.0001695643, 0.022143414, -0.040064987, -0.035078116, -0.025940238, -0.031847984, 0.024424342, 0.005093127, -0.00896787, 0.01989082, -0.014429346, 0.012141335, 0.018785777, 0.009775404, 0.01469144, -0.010908783 ... (1516 more) ]</pre></div></td></tr></tbody></table></div></details></td></tr><tr><td>1</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>TextWithEmbedding { Text = Title: DependencyContext RuntimeGraph expected behavior.\nArea: AssemblyLoader coreclr\n\n linux unix-x64 unix any base Runtime: android.30-x64 | Fallbacks: android.30 android.29-x64 android.29 android.28-x64 android.28 android.27-x64 android.27 android.26-x64 android.26 a...</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Text</td><td><div class="dni-plaintext"><pre>Title: DependencyContext RuntimeGraph expected behavior.
Area: AssemblyLoader coreclr

 linux unix-x64 unix any base Runtime: android.30-x64 | Fallbacks: android.30 android.29-x64 android.29 android.28-x64 android.28 android.27-x64 android.27 android.26-x64 android.26 android.25-x64 android.25 android.24-x64 android.24 android.23-x64 android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.31-x64 | Fallbacks: android.31 android.30-x64 android.30 android.29-x64 android.29 android.28-x64 android.28 android.27-x64 android.27 android.26-x64 android.26 android.25-x64 android.25 android.24-x64 android.24 android.23-x64 android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: android.32-x64 | Fallbacks: android.32 android.31-x64 android.31 android.30-x64 android.30 android.29-x64 android.29 android.28-x64 android.28 android.27-x64 android.27 android.26-x64 android.26 android.25-x64 android.25 android.24-x64 android.24 android.23-x64 android.23 android.22-x64 android.22 android.21-x64 android.21 android-x64 android linux-x64 linux unix-x64 unix any base Runtime: arch-x64 | Fallbacks: arch linux-x64 linux unix-x64 unix any base Runtime: centos-x64 | Fallbacks: centos rhel-x64 rhel linux-x64 linux unix-x64 unix any base Runtime: centos.7-x64 | Fallbacks: centos.7 centos-x64 rhel.7-x64 centos rhel.7 rhel-x64 rhel linux-x64 linux unix-x64 unix any base Runtime: centos.8-x64 | Fallbacks: centos.8 centos-x64 rhel.8-x64 centos rhel.8 rhel-x64 rhel linux-x64 linux unix-x64 unix any base Runtime: centos.9-x64 | Fallbacks: centos.9 centos-x64 rhel.9-x64 centos rhel.9 rhel-x64 rhel linux-x64 linux unix-x64 unix any base {SNIP} (you get the idea)...

On Rocky Linux 8.8, I also get a single entry stating `rocky.8-x64` (note: no minor build number, unlike OL) with fallbacks for Rocky.

It seems to be about every known linux distribution from https://github.com/dotnet/runtime/blob/main/src/libraries/Microsoft.NETCore.Platforms/src/runtime.json.

Which of these is the correct behavior?
Does something else need to be done during build/publish time to change this?</pre></div></td></tr><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ 0.014448724, -0.010964074, 0.007867405, -0.01898236, -0.013112343, 0.025534939, -0.04779359, -0.032216847, -0.030779878, -0.011991506, 0.043597642, -0.008169169, -0.008772696, 0.006308294, -0.01902547, 0.014032003, 0.01717178, -0.002942194, 0.010518613, -0.010123447 ... (1516 more) ]</pre></div></td></tr></tbody></table></div></details></td></tr></tbody></table></td></tr><tr><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>GitHubIssue { Title = No way to set the .NET entry assembly from unmanaged host? Also, documentation seems to be wrong/unclear., Text = ### Description\r\n\r\nI have a rather complex scenario where I have to launch a WPF (.NET 7) application from another .NET application (let&#39;s call that one the .NE...</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Title</td><td><div class="dni-plaintext"><pre>No way to set the .NET entry assembly from unmanaged host? Also, documentation seems to be wrong/unclear.</pre></div></td></tr><tr><td>Text</td><td><div class="dni-plaintext"><pre>### Description

I have a rather complex scenario where I have to launch a WPF (.NET 7) application from another .NET application (let&#39;s call that one the .NET host application), in the same process as the .NET host application, which in turn is hosted in an unmanaged application using hostfxr&#39;s `hdt_load_assembly_and_get_function_pointer`.

In the  .NET host application, I load the WPF application assembly from bytes in memory, then invoke the `Assembly.EntryPoint` to launch it. It is here that the problem manifests itself. WPF depends on Assembly.GetEntryAssembly () returning the main WPF assembly for resource management.
In my case however, Assembly.GetEntryAssembly () returns `null`, with no apparent way to set it to anything else, making the WPF app crash.

The [documentation](https://learn.microsoft.com/en-us/dotnet/api/system.reflection.assembly.getentryassembly?view=net-7.0) for `Assembly.GetEntryAssembly` states it returns:

`The assembly that is the process executable in the default application domain, or the first executable that was executed by AppDomain.ExecuteAssembly(String). Can return null when called from unmanaged code.`

However, when I look at the runtime source code, all `AppDomain.ExecuteAssembly(String)` seems to do is call a private method that invokes the assembly entrypoint, just like I am doing:

```c#
private static int ExecuteAssembly(Assembly assembly, string?[]? args)
{
    MethodInfo? entry = assembly.EntryPoint;
    if (entry == null)
    {
        throw new MissingMethodException(SR.Arg_EntryPointNotFoundException);
    }

    object? result = entry.Invoke(
        obj: null,
        invokeAttr: BindingFlags.DoNotWrapExceptions,
        binder: null,
        parameters: entry.GetParameters().Length &gt; 0 ? new object?[] { args } : null,
        culture: null);

    return result != null ? (int)result : 0;
}
``` 

Needless to say, this does not make `Assembly.GetEntryAssembly ()` return a different value.

As far as I can tell, there&#39;s no way to change the behavior of `Assembly.GetEntryAssembly ()`, as all this function does is call `GetEntryAssemblyInternal`: [github](https://github.com/search?q=repo%3Adotnet%2Fruntime%20GetEntryAssemblyInternal&amp;type=code)

So how do I make this work? (assuming I can&#39;t modify the WPF application as it is not mine).



### Reproduction Steps

Create an unmanaged .NET host, use that to load a .NET assembly that loads another .NET assembly that is the main assembly of a WPF application.
Start the WPF application by invoking its `Assembly.EntryPoint`.

The application crashes on exceptions caused by `Assembly.GetEntryAssembly ()` returning `null`.

### Expected behavior

I expect there should be a way for an unmanaged host (or even another .NET assembly) to set the value returned by `Assembly.GetEntryAssembly()` to a specific assembly so that it does not return `null`.

### Actual behavior

`Assembly.GetEntryAssembly()` always returns `null`

### Regression?

_No response_

### Known Workarounds

None as far as I can tell

### Configuration

.NET 7, Windows 11, x64. Not relevant, I suspect.

### Other information

_No response_</pre></div></td></tr><tr><td>Area</td><td><div class="dni-plaintext"><pre>AssemblyLoader coreclr</pre></div></td></tr><tr><td>Url</td><td><div class="dni-plaintext"><pre>https://github.com/dotnet/runtime/issues/94139</pre></div></td></tr></tbody></table></div></details></td><td><table><thead><tr><th><i>index</i></th><th>value</th></tr></thead><tbody><tr><td>0</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>TextWithEmbedding { Text = Title: No way to set the .NET entry assembly from unmanaged host? Also, documentation seems to be wrong/unclear.\nArea: AssemblyLoader coreclr\n\n### Description\r\n\r\nI have a rather complex scenario where I have to launch a WPF (.NET 7) application from another .NET app...</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Text</td><td><div class="dni-plaintext"><pre>Title: No way to set the .NET entry assembly from unmanaged host? Also, documentation seems to be wrong/unclear.
Area: AssemblyLoader coreclr

### Description

I have a rather complex scenario where I have to launch a WPF (.NET 7) application from another .NET application (let&#39;s call that one the .NET host application), in the same process as the .NET host application, which in turn is hosted in an unmanaged application using hostfxr&#39;s `hdt_load_assembly_and_get_function_pointer`.

In the  .NET host application, I load the WPF application assembly from bytes in memory, then invoke the `Assembly.EntryPoint` to launch it. It is here that the problem manifests itself. WPF depends on Assembly.GetEntryAssembly () returning the main WPF assembly for resource management.
In my case however, Assembly.GetEntryAssembly () returns `null`, with no apparent way to set it to anything else, making the WPF app crash.

The [documentation](https://learn.microsoft.com/en-us/dotnet/api/system.reflection.assembly.getentryassembly?view=net-7.0) for `Assembly.GetEntryAssembly` states it returns:

`The assembly that is the process executable in the default application domain, or the first executable that was executed by AppDomain.ExecuteAssembly(String). Can return null when called from unmanaged code.`

However, when I look at the runtime source code, all `AppDomain.ExecuteAssembly(String)` seems to do is call a private method that invokes the assembly entrypoint, just like I am doing:

```c#
private static int ExecuteAssembly(Assembly assembly, string?[]? args)
{
    MethodInfo? entry = assembly.EntryPoint;
    if (entry == null)
    {
        throw new MissingMethodException(SR.Arg_EntryPointNotFoundException);
    }

    object? result = entry.Invoke(
        obj: null,
        invokeAttr: BindingFlags.DoNotWrapExceptions,
        binder: null,
        parameters: entry.GetParameters().Length &gt; 0 ? new object?[] { args } : null,
        culture: null);

    return result != null ? (int)result : 0;
}
``` 

Needless to say, this does not make `Assembly.GetEntryAssembly ()` return a different value.

As far as I can tell, there&#39;s no way to change the behavior of `Assembly.GetEntryAssembly ()`, as all this function does is call `GetEntryAssemblyInternal`: [github](https://github.com/search?q=repo%3Adotnet%2Fruntime%20GetEntryAssemblyInternal&amp;type=code)

So how do I make this work? (assuming I can&#39;t modify the WPF application as it is not mine).



### Reproduction Steps

Create an unmanaged .NET host, use that to load a .NET assembly that loads another .NET assembly that is the main assembly of a WPF application.
Start the WPF application by invoking its `Assembly.EntryPoint`.

The application crashes on exceptions caused by `Assembly.GetEntryAssembly ()` returning `null`.

### Expected behavior

I expect there should be a way for an unmanaged host (or even another .NET assembly) to set the value returned by `Assembly.GetEntryAssembly()` to a specific assembly so that it does not return `null`.

### Actual behavior

`Assembly.GetEntryAssembly()` always returns `null`

### Regression?

_No response_

### Known Workarounds

None as far as I can tell

### Configuration

.NET 7, Windows 11, x64. Not relevant, I suspect.

### Other information

_No response_</pre></div></td></tr><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.006688335, 0.022486106, 0.004288016, 0.0018721103, -0.026600938, 0.022652362, -0.015337103, -0.007820953, -0.04572036, 0.004790247, 0.027598472, 0.010529538, 0.0038100302, -0.0036541652, 0.0019933386, -0.0018253508, 0.011007523, -0.027252106, -0.0063454323, -0.01850982 ... (1516 more) ]</pre></div></td></tr></tbody></table></div></details></td></tr></tbody></table></td></tr><tr><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>GitHubIssue { Title = System.Text.Json 7.0.2 fails to load in a .net6 application., Text = ### Description\r\n\r\nWe are attempting to reference RestSharp 110.2 which references system.text.json 7.0.2.\r\nOur application has many integration scenarios (as a full wpf application, headless service, or...</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Title</td><td><div class="dni-plaintext"><pre>System.Text.Json 7.0.2 fails to load in a .net6 application.</pre></div></td></tr><tr><td>Text</td><td><div class="dni-plaintext"><pre>### Description

We are attempting to reference RestSharp 110.2 which references system.text.json 7.0.2.
Our application has many integration scenarios (as a full wpf application, headless service, or as a desktop plugin) and targets net6.
We do not currently use deps.json files as they caused our application to not find our own project dependencies during our migration to .net6.

When starting our application we try to use a RestSharp class, this attempts to resolve system.text.json which fails with an error that the assembly cannot be loaded. A reason is not given and the inner exception is null.

Using dotnet-trace I see the following:
```
HasStack=&quot;True&quot; ThreadID=&quot;19,208&quot; ProcessorNumber=&quot;0&quot; ClrInstanceID=&quot;6&quot; AssemblyName=&quot;System.Text.Json, Version=7.0.0.0, Culture=neutral, PublicKeyToken=cc7b13ffcd2ddd51&quot; Stage=&quot;ApplicationAssemblies&quot; AssemblyLoadContext=&quot;Default&quot; Result=&quot;MismatchedAssemblyName&quot; ResultAssemblyName=&quot;System.Text.Json, Version=6.0.0.0, Culture=neutral, PublicKeyToken=cc7b13ffcd2ddd51&quot; ResultAssemblyPath=&quot;C:\Program Files\dotnet\shared\Microsoft.NETCore.App\6.0.23\System.Text.Json.dll&quot; ErrorMessage=&quot;Requested version 7.0.0.0 is incompatible with found version 6.0.0.0&quot; ActivityID=&quot;/#15712/1/225/&quot; 
```

I have verified in the module window in visual studio that no version of system.text.json is loaded at this time.

In a small test project, we can recreate this issue by removing the deps.json file.







### Reproduction Steps

please see:
https://github.com/pinzart90/SytemTextDemo/tree/main

this test project sets `GenerateDependencyFile` to false at build time, setting it to true resolves the issue. But that causes other issues for our application and integration.

We&#39;d like to understand if the behavior we are seeing is a bug or as designed and if there are any workarounds.

### Expected behavior

We expect that STJ 7.0 will be loaded in net6 since we reference it and no other earlier version of STJ is loaded.
We expect that it will be loaded even if a `GenerateDependencyFile` is false at build time and we do not have deps.json files.

### Actual behavior

The assembly fails to load even when using an assembly resolver.

### Regression?

unknown

### Known Workarounds

Hoping you can provide some.

### Configuration

6.0.415
windows 11 pro
x64
no
na

### Other information

_No response_</pre></div></td></tr><tr><td>Area</td><td><div class="dni-plaintext"><pre>AssemblyLoader coreclr</pre></div></td></tr><tr><td>Url</td><td><div class="dni-plaintext"><pre>https://github.com/dotnet/runtime/issues/93780</pre></div></td></tr></tbody></table></div></details></td><td><table><thead><tr><th><i>index</i></th><th>value</th></tr></thead><tbody><tr><td>0</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>TextWithEmbedding { Text = Title: System.Text.Json 7.0.2 fails to load in a .net6 application.\nArea: AssemblyLoader coreclr\n\n### Description\r\n\r\nWe are attempting to reference RestSharp 110.2 which references system.text.json 7.0.2.\r\nOur application has many integration scenarios (as a full ...</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Text</td><td><div class="dni-plaintext"><pre>Title: System.Text.Json 7.0.2 fails to load in a .net6 application.
Area: AssemblyLoader coreclr

### Description

We are attempting to reference RestSharp 110.2 which references system.text.json 7.0.2.
Our application has many integration scenarios (as a full wpf application, headless service, or as a desktop plugin) and targets net6.
We do not currently use deps.json files as they caused our application to not find our own project dependencies during our migration to .net6.

When starting our application we try to use a RestSharp class, this attempts to resolve system.text.json which fails with an error that the assembly cannot be loaded. A reason is not given and the inner exception is null.

Using dotnet-trace I see the following:
```
HasStack=&quot;True&quot; ThreadID=&quot;19,208&quot; ProcessorNumber=&quot;0&quot; ClrInstanceID=&quot;6&quot; AssemblyName=&quot;System.Text.Json, Version=7.0.0.0, Culture=neutral, PublicKeyToken=cc7b13ffcd2ddd51&quot; Stage=&quot;ApplicationAssemblies&quot; AssemblyLoadContext=&quot;Default&quot; Result=&quot;MismatchedAssemblyName&quot; ResultAssemblyName=&quot;System.Text.Json, Version=6.0.0.0, Culture=neutral, PublicKeyToken=cc7b13ffcd2ddd51&quot; ResultAssemblyPath=&quot;C:\Program Files\dotnet\shared\Microsoft.NETCore.App\6.0.23\System.Text.Json.dll&quot; ErrorMessage=&quot;Requested version 7.0.0.0 is incompatible with found version 6.0.0.0&quot; ActivityID=&quot;/#15712/1/225/&quot; 
```

I have verified in the module window in visual studio that no version of system.text.json is loaded at this time.

In a small test project, we can recreate this issue by removing the deps.json file.







### Reproduction Steps

please see:
https://github.com/pinzart90/SytemTextDemo/tree/main

this test project sets `GenerateDependencyFile` to false at build time, setting it to true resolves the issue. But that causes other issues for our application and integration.

We&#39;d like to understand if the behavior we are seeing is a bug or as designed and if there are any workarounds.

### Expected behavior

We expect that STJ 7.0 will be loaded in net6 since we reference it and no other earlier version of STJ is loaded.
We expect that it will be loaded even if a `GenerateDependencyFile` is false at build time and we do not have deps.json files.

### Actual behavior

The assembly fails to load even when using an assembly resolver.

### Regression?

unknown

### Known Workarounds

Hoping you can provide some.

### Configuration

6.0.415
windows 11 pro
x64
no
na

### Other information

_No response_</pre></div></td></tr><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.00927887, 0.018907337, -0.015265708, 0.02027659, 0.0013765356, 0.046671115, -0.021864338, -0.019985259, -0.018543173, -0.012417953, 0.014624781, -0.0073779398, 0.008332047, -0.013721657, 0.0011325466, 0.01654756, 0.01696999, -0.006893603, 0.027588978, -0.02980309 ... (1516 more) ]</pre></div></td></tr></tbody></table></div></details></td></tr></tbody></table></td></tr><tr><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>GitHubIssue { Title = [API Proposal]: Add constructor to AssemblyLoadContext that&#39;s accepting custom path, Text = ### Background and motivation\r\n\r\n# Current Situation\r\n\r\nIt’s such a common case to load a number of assemblies from a different location – in fact, this is the default case for u...</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Title</td><td><div class="dni-plaintext"><pre>[API Proposal]: Add constructor to AssemblyLoadContext that&#39;s accepting custom path</pre></div></td></tr><tr><td>Text</td><td><div class="dni-plaintext"><pre>### Background and motivation

# Current Situation

It’s such a common case to load a number of assemblies from a different location – in fact, this is the default case for using [`AssemblyLoadContext`](https://learn.microsoft.com/en-us/dotnet/api/system.runtime.loader.assemblyloadcontext).

Yet, for implementing this simple and regular use case, it&#39;s required to manually derive from [`AssemblyLoadContext`](https://learn.microsoft.com/en-us/dotnet/api/system.runtime.loader.assemblyloadcontext) and to create your own implementation that’s doing nothing else than just providing [`AssemblyDependencyResolver`](https://learn.microsoft.com/en-us/dotnet/api/system.runtime.loader.assemblydependencyresolver) with that path, making a mountain out of a molehill.

See [Create a .NET Core application with plugins](https://learn.microsoft.com/en-us/dotnet/core/tutorials/creating-app-with-plugin-support#load-plugins) for reference.

# Desired Situation

I propose to add a constructor to [`AssemblyLoadContext`](https://learn.microsoft.com/en-us/dotnet/api/system.runtime.loader.assemblyloadcontext) that’s accepting a file system directory path, so we can just use this constructor instead of being required to create a blatantly dispensable new class for this standard use case.

### API Proposal

```csharp
public AssemblyLoadContext(string filePath, string? name = null, bool isCollectible = false) {}
```

### API Usage

```csharp
new AssemblyLoadContext(@&quot;.\plugIns&quot;).LoadFromAssemblyPath(@&quot;.\plugIns\PlugIn1.dll&quot;)
```</pre></div></td></tr><tr><td>Area</td><td><div class="dni-plaintext"><pre>AssemblyLoader coreclr</pre></div></td></tr><tr><td>Url</td><td><div class="dni-plaintext"><pre>https://github.com/dotnet/runtime/issues/92074</pre></div></td></tr></tbody></table></div></details></td><td><table><thead><tr><th><i>index</i></th><th>value</th></tr></thead><tbody><tr><td>0</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>TextWithEmbedding { Text = Title: [API Proposal]: Add constructor to AssemblyLoadContext that&#39;s accepting custom path\nArea: AssemblyLoader coreclr\n\n### Background and motivation\r\n\r\n# Current Situation\r\n\r\nIt’s such a common case to load a number of assemblies from a different location – in...</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Text</td><td><div class="dni-plaintext"><pre>Title: [API Proposal]: Add constructor to AssemblyLoadContext that&#39;s accepting custom path
Area: AssemblyLoader coreclr

### Background and motivation

# Current Situation

It’s such a common case to load a number of assemblies from a different location – in fact, this is the default case for using [`AssemblyLoadContext`](https://learn.microsoft.com/en-us/dotnet/api/system.runtime.loader.assemblyloadcontext).

Yet, for implementing this simple and regular use case, it&#39;s required to manually derive from [`AssemblyLoadContext`](https://learn.microsoft.com/en-us/dotnet/api/system.runtime.loader.assemblyloadcontext) and to create your own implementation that’s doing nothing else than just providing [`AssemblyDependencyResolver`](https://learn.microsoft.com/en-us/dotnet/api/system.runtime.loader.assemblydependencyresolver) with that path, making a mountain out of a molehill.

See [Create a .NET Core application with plugins](https://learn.microsoft.com/en-us/dotnet/core/tutorials/creating-app-with-plugin-support#load-plugins) for reference.

# Desired Situation

I propose to add a constructor to [`AssemblyLoadContext`](https://learn.microsoft.com/en-us/dotnet/api/system.runtime.loader.assemblyloadcontext) that’s accepting a file system directory path, so we can just use this constructor instead of being required to create a blatantly dispensable new class for this standard use case.

### API Proposal

```csharp
public AssemblyLoadContext(string filePath, string? name = null, bool isCollectible = false) {}
```

### API Usage

```csharp
new AssemblyLoadContext(@&quot;.\plugIns&quot;).LoadFromAssemblyPath(@&quot;.\plugIns\PlugIn1.dll&quot;)
```</pre></div></td></tr><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ 0.030751444, 0.020277333, -0.0030771983, -0.0032569012, -0.0203321, 0.034420807, -0.030285928, -0.021564348, -0.0012801692, -0.013506812, 0.036885303, -0.0117405895, 0.010638412, 0.009193943, 0.00024431036, 0.005315783, 0.030477611, 0.0010653815, 0.01677227, 0.025028335 ... (1516 more) ]</pre></div></td></tr></tbody></table></div></details></td></tr></tbody></table></td></tr><tr><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>GitHubIssue { Title = Assembly Resolution finding a mismatched version corrupts state and makes loading good version impossible, Text = ### Description\r\n\r\nWhile trying to resolve an assembly for loading, if one of the stages finds a mismatched version, later resolution stages fail to load good v...</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Title</td><td><div class="dni-plaintext"><pre>Assembly Resolution finding a mismatched version corrupts state and makes loading good version impossible</pre></div></td></tr><tr><td>Text</td><td><div class="dni-plaintext"><pre>### Description

While trying to resolve an assembly for loading, if one of the stages finds a mismatched version, later resolution stages fail to load good version.

### Reproduction Steps

Put an old version of the required assembly in the application process .exe folder, then subscribe an assembly loader that will pick the right version from elsewhere.

Alternatively, I suspect that attempting to call Assembly.Load(name) with the wrong version beside the .exe and then execute LoadFrom(pathToGoodVersion) will fail as well.

### Expected behavior

Good version should always succeed to load.

Here are traces of what happens if the assembly is not found at stage:ApplicationAssemblies, stage:AppDomainAssemblyResolveEvent succeeds.

&lt;HTML&gt;
&lt;BODY&gt;
&lt;!--StartFragment--&gt;&lt;TABLE&gt;&lt;TR&gt;&lt;TD&gt;Rest&lt;/TD&gt;&lt;/TR&gt;&lt;TR&gt;&lt;TD&gt;HasStack=&amp;quot;True&amp;quot; ThreadID=&amp;quot;55,320&amp;quot; ProcessorNumber=&amp;quot;0&amp;quot; ClrInstanceID=&amp;quot;10&amp;quot; AssemblyName=&amp;quot;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; Stage=&amp;quot;FindInLoadContext&amp;quot; AssemblyLoadContext=&amp;quot;Default&amp;quot; Result=&amp;quot;AssemblyNotFound&amp;quot; ResultAssemblyName=&amp;quot;&amp;quot; ResultAssemblyPath=&amp;quot;&amp;quot; ErrorMessage=&amp;quot;Could not locate assembly&amp;quot; ActivityID=&amp;quot;/#7520/1/206/&amp;quot; &lt;/TD&gt;&lt;/TR&gt;&lt;TR&gt;&lt;TD&gt;HasStack=&amp;quot;True&amp;quot; ThreadID=&amp;quot;55,320&amp;quot; ProcessorNumber=&amp;quot;0&amp;quot; ClrInstanceID=&amp;quot;10&amp;quot; AssemblyName=&amp;quot;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; Stage=&amp;quot;ApplicationAssemblies&amp;quot; AssemblyLoadContext=&amp;quot;Default&amp;quot; Result=&amp;quot;AssemblyNotFound&amp;quot; ResultAssemblyName=&amp;quot;&amp;quot; ResultAssemblyPath=&amp;quot;&amp;quot; ErrorMessage=&amp;quot;Could not locate assembly&amp;quot; ActivityID=&amp;quot;/#7520/1/206/&amp;quot; &lt;/TD&gt;&lt;/TR&gt;&lt;TR&gt;&lt;TD&gt;HasStack=&amp;quot;True&amp;quot; ThreadID=&amp;quot;55,320&amp;quot; ProcessorNumber=&amp;quot;0&amp;quot; ClrInstanceID=&amp;quot;10&amp;quot; AssemblyName=&amp;quot;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; Stage=&amp;quot;AssemblyLoadContextResolvingEvent&amp;quot; AssemblyLoadContext=&amp;quot;Default&amp;quot; Result=&amp;quot;AssemblyNotFound&amp;quot; ResultAssemblyName=&amp;quot;&amp;quot; ResultAssemblyPath=&amp;quot;&amp;quot; ErrorMessage=&amp;quot;Could not locate assembly&amp;quot; ActivityID=&amp;quot;/#7520/1/206/&amp;quot; &lt;/TD&gt;&lt;/TR&gt;&lt;TR&gt;&lt;TD&gt;HasStack=&amp;quot;True&amp;quot; ThreadID=&amp;quot;55,320&amp;quot; ProcessorNumber=&amp;quot;0&amp;quot; ClrInstanceID=&amp;quot;10&amp;quot; AssemblyName=&amp;quot;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; Stage=&amp;quot;AppDomainAssemblyResolveEvent&amp;quot; AssemblyLoadContext=&amp;quot;Default&amp;quot; Result=&amp;quot;Success&amp;quot; ResultAssemblyName=&amp;quot;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; ResultAssemblyPath=&amp;quot;D:\src\subs\target\test\managedstore\Internal.Exchange.Test.ManagedStore.QTests.Common\bin\Debug\net6.0\FluentAssertions.dll&amp;quot; ErrorMessage=&amp;quot;&amp;quot; ActivityID=&amp;quot;/#7520/1/206/&amp;quot; &lt;/TD&gt;&lt;/TR&gt;&lt;/TABLE&gt;
&lt;!--EndFragment--&gt;
&lt;/BODY&gt;
&lt;/HTML&gt;

### Actual behavior

If the mismatched version is ever rejected, no good version can be loaded.

Here are traces of what happens if the mismatched version is found at stage:ApplicationAssemblies, stage:AppDomainAssemblyResolveEvent succeeds.

&lt;HTML&gt;
&lt;BODY&gt;
&lt;!--StartFragment--&gt;&lt;TABLE&gt;&lt;TR&gt;&lt;TD&gt;Rest&lt;/TD&gt;&lt;/TR&gt;&lt;TR&gt;&lt;TD&gt;HasStack=&amp;quot;True&amp;quot; ThreadID=&amp;quot;60,068&amp;quot; ProcessorNumber=&amp;quot;0&amp;quot; ClrInstanceID=&amp;quot;9&amp;quot; AssemblyName=&amp;quot;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; Stage=&amp;quot;FindInLoadContext&amp;quot; AssemblyLoadContext=&amp;quot;Default&amp;quot; Result=&amp;quot;AssemblyNotFound&amp;quot; ResultAssemblyName=&amp;quot;&amp;quot; ResultAssemblyPath=&amp;quot;&amp;quot; ErrorMessage=&amp;quot;Could not locate assembly&amp;quot; ActivityID=&amp;quot;/#57240/1/206/&amp;quot; &lt;/TD&gt;&lt;/TR&gt;&lt;TR&gt;&lt;TD&gt;HasStack=&amp;quot;True&amp;quot; ThreadID=&amp;quot;60,068&amp;quot; ProcessorNumber=&amp;quot;0&amp;quot; ClrInstanceID=&amp;quot;9&amp;quot; AssemblyName=&amp;quot;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; Stage=&amp;quot;ApplicationAssemblies&amp;quot; AssemblyLoadContext=&amp;quot;Default&amp;quot; Result=&amp;quot;MismatchedAssemblyName&amp;quot; ResultAssemblyName=&amp;quot;FluentAssertions, Version=5.6.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; ResultAssemblyPath=&amp;quot;D:\src\subs\target\test\tools\UniTP\bin\Debug\net6.0\FluentAssertions.dll&amp;quot; ErrorMessage=&amp;quot;Requested version 6.7.0.0 is incompatible with found version 5.6.0.0&amp;quot; ActivityID=&amp;quot;/#57240/1/206/&amp;quot; &lt;/TD&gt;&lt;/TR&gt;&lt;TR&gt;&lt;TD&gt;HasStack=&amp;quot;True&amp;quot; ThreadID=&amp;quot;60,068&amp;quot; ProcessorNumber=&amp;quot;0&amp;quot; ClrInstanceID=&amp;quot;9&amp;quot; AssemblyName=&amp;quot;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; Stage=&amp;quot;AssemblyLoadContextResolvingEvent&amp;quot; AssemblyLoadContext=&amp;quot;Default&amp;quot; Result=&amp;quot;AssemblyNotFound&amp;quot; ResultAssemblyName=&amp;quot;&amp;quot; ResultAssemblyPath=&amp;quot;&amp;quot; ErrorMessage=&amp;quot;Could not locate assembly&amp;quot; ActivityID=&amp;quot;/#57240/1/206/&amp;quot; &lt;/TD&gt;&lt;/TR&gt;&lt;TR&gt;&lt;TD&gt;HasStack=&amp;quot;True&amp;quot; ThreadID=&amp;quot;60,068&amp;quot; ProcessorNumber=&amp;quot;0&amp;quot; ClrInstanceID=&amp;quot;9&amp;quot; AssemblyName=&amp;quot;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; Stage=&amp;quot;AppDomainAssemblyResolveEvent&amp;quot; AssemblyLoadContext=&amp;quot;Default&amp;quot; Result=&amp;quot;Exception&amp;quot; ResultAssemblyName=&amp;quot;&amp;quot; ResultAssemblyPath=&amp;quot;&amp;quot; ErrorMessage=&amp;quot;Could not load file or assembly &#39;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&#39;.&amp;quot; ActivityID=&amp;quot;/#57240/1/206/&amp;quot; &lt;/TD&gt;&lt;/TR&gt;&lt;/TABLE&gt;
&lt;!--EndFragment--&gt;
&lt;/BODY&gt;
&lt;/HTML&gt;

### Regression?

Yes,
This did not happen in net472, but happens in net6.0

### Known Workarounds

Avoid bad versions.

### Configuration

net6.0
Windows
x64


### Other information

_No response_</pre></div></td></tr><tr><td>Area</td><td><div class="dni-plaintext"><pre>AssemblyLoader coreclr</pre></div></td></tr><tr><td>Url</td><td><div class="dni-plaintext"><pre>https://github.com/dotnet/runtime/issues/91952</pre></div></td></tr></tbody></table></div></details></td><td><table><thead><tr><th><i>index</i></th><th>value</th></tr></thead><tbody><tr><td>0</td><td><details class="dni-treeview"><summary><span class="dni-code-hint"><code>TextWithEmbedding { Text = Title: Assembly Resolution finding a mismatched version corrupts state and makes loading good version impossible\nArea: AssemblyLoader coreclr\n\n### Description\r\n\r\nWhile trying to resolve an assembly for loading, if one of the stages finds a mismatched version, later ...</code></span></summary><div><table><thead><tr></tr></thead><tbody><tr><td>Text</td><td><div class="dni-plaintext"><pre>Title: Assembly Resolution finding a mismatched version corrupts state and makes loading good version impossible
Area: AssemblyLoader coreclr

### Description

While trying to resolve an assembly for loading, if one of the stages finds a mismatched version, later resolution stages fail to load good version.

### Reproduction Steps

Put an old version of the required assembly in the application process .exe folder, then subscribe an assembly loader that will pick the right version from elsewhere.

Alternatively, I suspect that attempting to call Assembly.Load(name) with the wrong version beside the .exe and then execute LoadFrom(pathToGoodVersion) will fail as well.

### Expected behavior

Good version should always succeed to load.

Here are traces of what happens if the assembly is not found at stage:ApplicationAssemblies, stage:AppDomainAssemblyResolveEvent succeeds.

&lt;HTML&gt;
&lt;BODY&gt;
&lt;!--StartFragment--&gt;&lt;TABLE&gt;&lt;TR&gt;&lt;TD&gt;Rest&lt;/TD&gt;&lt;/TR&gt;&lt;TR&gt;&lt;TD&gt;HasStack=&amp;quot;True&amp;quot; ThreadID=&amp;quot;55,320&amp;quot; ProcessorNumber=&amp;quot;0&amp;quot; ClrInstanceID=&amp;quot;10&amp;quot; AssemblyName=&amp;quot;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; Stage=&amp;quot;FindInLoadContext&amp;quot; AssemblyLoadContext=&amp;quot;Default&amp;quot; Result=&amp;quot;AssemblyNotFound&amp;quot; ResultAssemblyName=&amp;quot;&amp;quot; ResultAssemblyPath=&amp;quot;&amp;quot; ErrorMessage=&amp;quot;Could not locate assembly&amp;quot; ActivityID=&amp;quot;/#7520/1/206/&amp;quot; &lt;/TD&gt;&lt;/TR&gt;&lt;TR&gt;&lt;TD&gt;HasStack=&amp;quot;True&amp;quot; ThreadID=&amp;quot;55,320&amp;quot; ProcessorNumber=&amp;quot;0&amp;quot; ClrInstanceID=&amp;quot;10&amp;quot; AssemblyName=&amp;quot;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; Stage=&amp;quot;ApplicationAssemblies&amp;quot; AssemblyLoadContext=&amp;quot;Default&amp;quot; Result=&amp;quot;AssemblyNotFound&amp;quot; ResultAssemblyName=&amp;quot;&amp;quot; ResultAssemblyPath=&amp;quot;&amp;quot; ErrorMessage=&amp;quot;Could not locate assembly&amp;quot; ActivityID=&amp;quot;/#7520/1/206/&amp;quot; &lt;/TD&gt;&lt;/TR&gt;&lt;TR&gt;&lt;TD&gt;HasStack=&amp;quot;True&amp;quot; ThreadID=&amp;quot;55,320&amp;quot; ProcessorNumber=&amp;quot;0&amp;quot; ClrInstanceID=&amp;quot;10&amp;quot; AssemblyName=&amp;quot;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; Stage=&amp;quot;AssemblyLoadContextResolvingEvent&amp;quot; AssemblyLoadContext=&amp;quot;Default&amp;quot; Result=&amp;quot;AssemblyNotFound&amp;quot; ResultAssemblyName=&amp;quot;&amp;quot; ResultAssemblyPath=&amp;quot;&amp;quot; ErrorMessage=&amp;quot;Could not locate assembly&amp;quot; ActivityID=&amp;quot;/#7520/1/206/&amp;quot; &lt;/TD&gt;&lt;/TR&gt;&lt;TR&gt;&lt;TD&gt;HasStack=&amp;quot;True&amp;quot; ThreadID=&amp;quot;55,320&amp;quot; ProcessorNumber=&amp;quot;0&amp;quot; ClrInstanceID=&amp;quot;10&amp;quot; AssemblyName=&amp;quot;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; Stage=&amp;quot;AppDomainAssemblyResolveEvent&amp;quot; AssemblyLoadContext=&amp;quot;Default&amp;quot; Result=&amp;quot;Success&amp;quot; ResultAssemblyName=&amp;quot;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; ResultAssemblyPath=&amp;quot;D:\src\subs\target\test\managedstore\Internal.Exchange.Test.ManagedStore.QTests.Common\bin\Debug\net6.0\FluentAssertions.dll&amp;quot; ErrorMessage=&amp;quot;&amp;quot; ActivityID=&amp;quot;/#7520/1/206/&amp;quot; &lt;/TD&gt;&lt;/TR&gt;&lt;/TABLE&gt;
&lt;!--EndFragment--&gt;
&lt;/BODY&gt;
&lt;/HTML&gt;

### Actual behavior

If the mismatched version is ever rejected, no good version can be loaded.

Here are traces of what happens if the mismatched version is found at stage:ApplicationAssemblies, stage:AppDomainAssemblyResolveEvent succeeds.

&lt;HTML&gt;
&lt;BODY&gt;
&lt;!--StartFragment--&gt;&lt;TABLE&gt;&lt;TR&gt;&lt;TD&gt;Rest&lt;/TD&gt;&lt;/TR&gt;&lt;TR&gt;&lt;TD&gt;HasStack=&amp;quot;True&amp;quot; ThreadID=&amp;quot;60,068&amp;quot; ProcessorNumber=&amp;quot;0&amp;quot; ClrInstanceID=&amp;quot;9&amp;quot; AssemblyName=&amp;quot;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; Stage=&amp;quot;FindInLoadContext&amp;quot; AssemblyLoadContext=&amp;quot;Default&amp;quot; Result=&amp;quot;AssemblyNotFound&amp;quot; ResultAssemblyName=&amp;quot;&amp;quot; ResultAssemblyPath=&amp;quot;&amp;quot; ErrorMessage=&amp;quot;Could not locate assembly&amp;quot; ActivityID=&amp;quot;/#57240/1/206/&amp;quot; &lt;/TD&gt;&lt;/TR&gt;&lt;TR&gt;&lt;TD&gt;HasStack=&amp;quot;True&amp;quot; ThreadID=&amp;quot;60,068&amp;quot; ProcessorNumber=&amp;quot;0&amp;quot; ClrInstanceID=&amp;quot;9&amp;quot; AssemblyName=&amp;quot;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; Stage=&amp;quot;ApplicationAssemblies&amp;quot; AssemblyLoadContext=&amp;quot;Default&amp;quot; Result=&amp;quot;MismatchedAssemblyName&amp;quot; ResultAssemblyName=&amp;quot;FluentAssertions, Version=5.6.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; ResultAssemblyPath=&amp;quot;D:\src\subs\target\test\tools\UniTP\bin\Debug\net6.0\FluentAssertions.dll&amp;quot; ErrorMessage=&amp;quot;Requested version 6.7.0.0 is incompatible with found version 5.6.0.0&amp;quot; ActivityID=&amp;quot;/#57240/1/206/&amp;quot; &lt;/TD&gt;&lt;/TR&gt;&lt;TR&gt;&lt;TD&gt;HasStack=&amp;quot;True&amp;quot; ThreadID=&amp;quot;60,068&amp;quot; ProcessorNumber=&amp;quot;0&amp;quot; ClrInstanceID=&amp;quot;9&amp;quot; AssemblyName=&amp;quot;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; Stage=&amp;quot;AssemblyLoadContextResolvingEvent&amp;quot; AssemblyLoadContext=&amp;quot;Default&amp;quot; Result=&amp;quot;AssemblyNotFound&amp;quot; ResultAssemblyName=&amp;quot;&amp;quot; ResultAssemblyPath=&amp;quot;&amp;quot; ErrorMessage=&amp;quot;Could not locate assembly&amp;quot; ActivityID=&amp;quot;/#57240/1/206/&amp;quot; &lt;/TD&gt;&lt;/TR&gt;&lt;TR&gt;&lt;TD&gt;HasStack=&amp;quot;True&amp;quot; ThreadID=&amp;quot;60,068&amp;quot; ProcessorNumber=&amp;quot;0&amp;quot; ClrInstanceID=&amp;quot;9&amp;quot; AssemblyName=&amp;quot;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&amp;quot; Stage=&amp;quot;AppDomainAssemblyResolveEvent&amp;quot; AssemblyLoadContext=&amp;quot;Default&amp;quot; Result=&amp;quot;Exception&amp;quot; ResultAssemblyName=&amp;quot;&amp;quot; ResultAssemblyPath=&amp;quot;&amp;quot; ErrorMessage=&amp;quot;Could not load file or assembly &#39;FluentAssertions, Version=6.7.0.0, Culture=neutral, PublicKeyToken=33f2691a05b67b6a&#39;.&amp;quot; ActivityID=&amp;quot;/#57240/1/206/&amp;quot; &lt;/TD&gt;&lt;/TR&gt;&lt;/TABLE&gt;
&lt;!--EndFragment--&gt;
&lt;/BODY&gt;
&lt;/HTML&gt;

### Regression?

Yes,
This did not happen in net472, but happens in net6.0

### Known Workarounds

Avoid bad versions.

### Configuration

net6.0
Windows
x64


### Other information

_No response_</pre></div></td></tr><tr><td>Embedding</td><td><div class="dni-plaintext"><pre>[ -0.006019099, 0.002385238, -0.010495782, -0.00064542814, -0.014777825, 0.024972469, -0.0423063, -0.012860822, -0.039955948, 0.00021483668, 0.02395888, -0.005093649, 0.014263687, 0.0122952685, -0.0051524076, 0.008145432, 0.02769006, 0.006059496, -0.00018385061, -0.027484404 ... (1516 more) ]</pre></div></td></tr></tbody></table></div></details></td></tr></tbody></table></td></tr></tbody></table><style>
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



```python
await SaveIssuesWithChunksToFileAsync(issuesWithChunksCollection, "areaIssuesWithEmbeddingsSubset.json");
```

## Embedding Search


```python
#r "nuget: System.Numerics.Tensors, 8.0.0-rc.2.23479.6"
```


<div><div></div><div></div><div><strong>Installed Packages</strong><ul><li><span>System.Numerics.Tensors, 8.0.0-rc.2.23479.6</span></li></ul></div></div>



```python
using System.Numerics.Tensors;
```


```python
public class SimilarityComparer : ISimilarityComparer<float[]>
{
    public float Score(float[] a, float[] b)
    {
        return TensorPrimitives.CosineSimilarity(a,b);
    }
}
```


```python
public async Task<string[]> EmbeddingSearchAsync(string query, IEnumerable<IssueWithChunks> data,int resultLimit = 1)
{
    var embeddingResponse = await openAIClient.GetEmbeddingsAsync(new EmbeddingsOptions(embeddingDeployment,new [] {query}));
    var embeddingVector = embeddingResponse.Value.Data[0].Embedding.ToArray();

    var searchResults = 
        data
        .SelectMany(d => d.Chunks)
        .ScoreBySimilarityTo(embeddingVector,new SimilarityComparer(),c => c.Embedding)
        .OrderByDescending(e => e.Score)
        .Where(e => e.Score > 0.5)
        .Take(resultLimit)
        .Select(e => e.Value.Text)
        .ToArray();
    
    return searchResults;
}
```


```python
(await EmbeddingSearchAsync("What are the latest issues for AOT", issuesWithChunksCollection, 3)).Display();
```


<div class="dni-plaintext"><pre>[ Title: [wasi] `System.Globalization.Tests` AOT build fails with `LLVM ERROR: out of memory` on Windows
Area: Codegen AOT mono

## Build Information
Build: https://dev.azure.com/dnceng-public/cbb18261-c48f-4abb-8651-8cdcb5474649/_build/results?buildId=481587
Build error leg or test failing: System.Globalization.Tests.WorkItemExecution
Pull request: https://github.com/dotnet/runtime/pull/95146
&lt;!-- Error message template  --&gt;
## Error Message

Fill the error message using [step by step known issues guidance](https://github.com/dotnet/arcade/blob/main/Documentation/Projects/Build%20Analysis/KnownIssues.md#how-to-fill-out-a-known-issue-error-section).

&lt;!-- Use ErrorMessage for String.Contains matches. Use ErrorPattern for regex matches (single line/no backtracking). Set BuildRetry to `true` to retry builds with this error. Set ExcludeConsoleLog to `true` to skip helix logs analysis. --&gt;

```json
{
  &quot;ErrorMessage&quot;: &quot;&quot;,
  &quot;ErrorPattern&quot;: &quot;LLVM ERROR: out of memory.*C:.*ProxyProjectForAOTOnHelix.proj&quot;,
  &quot;BuildRetry&quot;: false,
  &quot;ExcludeConsoleLog&quot;: false
}
```

From the console log:
```
error : Failed to compile C:\helix\work\workitem\e\wasm_build\obj\wasm\for-build\aot-instances.dll.bc -&gt; C:\helix\work\workitem\e\wasm_build\obj\wasm\for-build\aot-instances.dll.o [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : C:\helix\work\workitem\e\publish&gt;C:\Windows\System32\chcp.com 65001 1&gt;nul  [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : C:\helix\work\workitem\e\publish&gt;setlocal [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : C:\helix\work\workitem\e\publish&gt;set errorlevel=dummy  [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : C:\helix\work\workitem\e\publish&gt;set errorlevel=  [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : C:\helix\work\workitem\e\publish&gt;C:\helix\work\correlation\build\\wasi-sdk\\bin\clang.exe &quot;@C:\helix\work\correlation\build\microsoft.netcore.app.runtime.wasi-wasm\runtimes\wasi-wasm\native\src\wasi-default.rsp&quot;  &quot;@C:\helix\work\workitem\e\wasm_build\obj\wasm\for-build\wasi-compile-bc.rsp&quot; -c -o &quot;C:\Users\ContainerAdministrator\AppData\Local\Temp\tmpmr5st5.tmp&quot; &quot;C:\helix\work\workitem\e\wasm_build\obj\wasm\for-build\aot-instances.dll.bc&quot;  [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : clang version 16.0.0 [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : Target: wasm32-unknown-wasi [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : Thread model: posix [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : InstalledDir: C:/helix/work/correlation/build//wasi-sdk//bin [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error :  (in-process) [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error :  &quot;C:/helix/work/correlation/build/wasi-sdk/bin/clang.exe&quot; -cc1 -triple wasm32-unknown-wasi -emit-obj -disable-free -clear-ast-before-backend -disable-llvm-verifier -discard-value-names -main-file-name aot-instances.dll.bc -mrelocation-model static -mframe-pointer=none -ffp-contract=on -fno-rounding-math -mconstructor-aliases -target-cpu generic -fvisibility=hidden -mllvm -treat-scalable-fixed-error-as-warning -debug-info-kind=constructor -dwarf-version=4 -debugger-tuning=gdb -v -fcoverage-compilation-dir=C:/helix/work/workitem/e/publish -resource-dir C:/helix/work/correlation/build/wasi-sdk/lib/clang/16 -Oz -fdebug-compilation-dir=C:/helix/work/workitem/e/publish -ferror-limit 19 -fgnuc-version=4.2.1 -vectorize-slp -o &quot;C:\\Users\\ContainerAdministrator\\AppData\\Local\\Temp\\tmpmr5st5.tmp&quot; -x ir &quot;C:\\helix\\work\\workitem\\e\\wasm_build\\obj\\wasm\\for-build\\aot-instances.dll.bc&quot; [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : clang -cc1 version 16.0.0 based upon LLVM 16.0.0 default target wasm32-wasi [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : LLVM ERROR: out of memory [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : Allocation failed [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace, preprocessed source, and associated run script. [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : Stack dump: [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : 0.	Program arguments: C:\\helix\\work\\correlation\\build\\\\wasi-sdk\\\\bin\\clang.exe @C:\\helix\\work\\correlation\\build\\microsoft.netcore.app.runtime.wasi-wasm\\runtimes\\wasi-wasm\\native\\src\\wasi-default.rsp @C:\\helix\\work\\workitem\\e\\wasm_build\\obj\\wasm\\for-build\\wasi-compile-bc.rsp -c -o C:\\Users\\ContainerAdministrator\\AppData\\Local\\Temp\\tmpmr5st5.tmp C:\\helix\\work\\workitem\\e\\wasm_build\\obj\\wasm\\for-build\\aot-instances.dll.bc [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : 1.	Code generation [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : Exception Code: 0xC000001D [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : 0x00A1FB73 &lt;unknown module&gt; [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : 0x7133C902 &lt;unknown module&gt; [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : 0x00A023A1 &lt;unknown module&gt; [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : 0x00E7FD3C &lt;unknown module&gt; [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : 0x00E7FFBC &lt;unknown module&gt; [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : 0x00E80133 &lt;unknown module&gt; [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : 0x008DE5CF &lt;unknown module&gt; [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : clang: error: clang frontend command failed due to signal (use -v to see invocation) [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : clang version 16.0.0 [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : Target: wasm32-unknown-wasi [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : Thread model: posix [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : InstalledDir: C:/helix/work/correlation/build//wasi-sdk//bin [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
error : clang: note: diagnostic msg: Error generating preprocessed source(s) - no preprocessable inputs. [took 93.46s] [C:\helix\work\workitem\e\publish\ProxyProjectForAOTOnHelix.proj]
```
&lt;!-- Known issue validation start --&gt;
 ### Known issue validation
**Build: :mag_right:** https://dev.azure.com/dnceng-public/public/_build/results?buildId=481587
**Error message validated:** `LLVM ERROR: out of memory.*C:.*ProxyProjectForAOTOnHelix.proj`
**Result validation: :white_check_mark:** Known issue matched with the provided build.
**Validation performed at:** 11/28/2023 11:22:33 PM UTC
&lt;!-- Known issue validation end --&gt;
&lt;!--Known issue error report start --&gt;
### Report

|Build|Definition|Test|Pull Request|
|---|---|---|---|
|[481587](https://dev.azure.com/dnceng-public/public/_build/results?buildId=481587)|dotnet/runtime|[System.Globalization.Tests.WorkItemExecution](https://dev.azure.com/dnceng-public/public/_build/results?buildId=481587&amp;view=ms.vss-test-web.build-test-results-tab&amp;runId=11065910&amp;resultId=100710)|dotnet/runtime#95146|
#### Summary
|24-Hour Hit Count|7-Day Hit Count|1-Month Count|
|---|---|---|
|0|0|1|
&lt;!--Known issue error report end --&gt;, Title: [wasm] Some AOT configuration combinations to disallow
Area: Codegen AOT mono

Some other cases:
- Debug+aot is not very usable either, aot doesn&#39;t support managed debugging.
- it might be possible to use profiled aot without il linking.


_Originally posted by @vargaz in https://github.com/dotnet/runtime/issues/94064#issuecomment-1781792993_
            , Title: [browser] run dotnet in service worker
Area: Build mono

Do we want the same wasm memory for multiple tabs ? 
Do we care that they would share user&#39;s security tokens inside the dotnet memory ? 
Do we have vision how &quot;upgrade&quot; should work when one of the tabs receives new version of the app ? 
What 3rd party solutions already exist ?
 ]</pre></div><style>
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


## Store in DB

### Start DB locally


```python
docker run -d -p 6333:6333 -p 6334:6334 -v "$pwd/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
```

    [31;1mdocker: error during connect: this error may indicate that the docker daemon is not running: Post "http://%2F%2F.%2Fpipe%2Fdocker_engine/v1.24/containers/create": open //./pipe/docker_engine: The system cannot find the file specified.[0m
    [31;1mSee 'docker run --help'.[0m
    


    Command failed: SubmitCode: docker run -d -p 6333:6333 -p 6334:6334 -v "$pwd/q ...



```python
using Qdrant.Client;
using Qdrant.Client.Grpc;
```

### Initialize Qdrant client


```python
var qdrantClient = new QdrantClient(host: "localhost",port: 6334,https:false);
```

### Create collection


```python
var collectionName = "gh_issues";
```


```python
var collections = await qdrantClient.ListCollectionsAsync();
if(collections.Where(x => x.Contains(collectionName)).Count() > 0) 
    await qdrantClient.DeleteCollectionAsync(collectionName);
```


```python
await qdrantClient.CreateCollectionAsync(collectionName, new VectorParams { Size=1536, Distance=Distance.Cosine})
```

### Map issue embeddings to points


```python
var vectors = 
    issuesWithChunksCollection
        .Where(d => d.Chunks.Count > 0)
        .SelectMany(d => 
            d.Chunks.Select(c => new {
                Embedding=c.Embedding,
                Text=$"<issueTitle>{d.Issue.Title}</issueTitle>\n<issueUrl>{d.Issue.Url}</issueUrl>\n<issueArea>{d.Issue.Area}</issueArea>\n<issueSnippet>{c.Text}</issueSnippet>"
                }))
        .ToList();
  
```


```python
var points = vectors.Select(vector => 
{
    var point = new PointStruct
    {
        Id = new PointId { Uuid = Guid.NewGuid().ToString() },
        Vectors = vector.Embedding,
        Payload = 
            {
                ["text"] = vector.Text
            }
    };
    return point;
}).ToList();

```

### Insert data into Qdrant collection


```python
await qdrantClient.UpsertAsync(collectionName,points);
```

## Get Count


```python
await qdrantClient.CountAsync(collectionName)
```


<div class="dni-plaintext"><pre>95</pre></div><style>
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


## Search with Qdrant


```python
public async Task<string[]> SearchWithQdrantAsync(string query, string collectionName, int resultLimit = 1)
{
    var embeddingResponse = await openAIClient.GetEmbeddingsAsync(new EmbeddingsOptions(embeddingDeployment,new [] {query}));
    var embeddingVector = embeddingResponse.Value.Data[0].Embedding.ToArray();

    var results = await qdrantClient.SearchAsync(collectionName,embeddingVector, limit:(ulong)resultLimit);
    return results.Select(r => r.Payload["text"].StringValue).ToArray();
}
```


```python
(await SearchWithQdrantAsync("What are the latest issues for AOT", collectionName, 3)).Display();
```


<div class="dni-plaintext"><pre>[ &lt;issueTitle&gt;[docs] Add Mono Library Mode and update Mono Profiled AOT&lt;/issueTitle&gt;
&lt;issueUrl&gt;https://github.com/dotnet/runtime/pull/94578&lt;/issueUrl&gt;
&lt;issueArea&gt;Build mono&lt;/issueArea&gt;
&lt;issueSnippet&gt;Title: [docs] Add Mono Library Mode and update Mono Profiled AOT
Area: Build mono

This PR looks to add/update documentation on [Mono&#39;s Library Mode](https://github.com/dotnet/runtime/issues/79377) and [Mono&#39;s Profiled AOT](https://github.com/dotnet/runtime/issues/69512) workflows.&lt;/issueSnippet&gt;, &lt;issueTitle&gt;[browser][AOT] Space in path to published project on unix&lt;/issueTitle&gt;
&lt;issueUrl&gt;https://github.com/dotnet/runtime/pull/94306&lt;/issueUrl&gt;
&lt;issueArea&gt;Build mono&lt;/issueArea&gt;
&lt;issueSnippet&gt;Title: [browser][AOT] Space in path to published project on unix
Area: Build mono

Theoretically, enclosing included directories in quotations should be enough to escape spaces in Unix.

Reason: wbt use `artifacts/bin/dotnet-latest/packs/Microsoft.NET.Runtime.Emscripten.3.1.34.Sdk.linux-x64/9.0.0-alpha.1.23528.4/tools/bin/clang` that is `16.0.5`. Version `17.0.0` is already working fine on Unix, should not need any code edition, only unification of `wbt` -&gt; `wbt artifacts` for all the platforms.
The issue reported clang `15.0.0` https://github.com/dotnet/runtime/issues/92335&lt;/issueSnippet&gt;, &lt;issueTitle&gt;[wasm] WBT `SatelliteAssembliesTests.CheckThatSatelliteAssembliesAreNotAOTed` failing&lt;/issueTitle&gt;
&lt;issueUrl&gt;https://github.com/dotnet/runtime/issues/90458&lt;/issueUrl&gt;
&lt;issueArea&gt;Build mono&lt;/issueArea&gt;
&lt;issueSnippet&gt;Title: [wasm] WBT `SatelliteAssembliesTests.CheckThatSatelliteAssembliesAreNotAOTed` failing
Area: Build mono

Test failing with [rolling build](https://dev.azure.com/dnceng-public/public/_build/results?buildId=371479&amp;view=results):

```
  [4/5] System.Private.CoreLib.dll.bc -&gt; System.Private.CoreLib.dll.o 

WasmApp.Native.targets(379,5): error : Failed to compile /root/helix/work/workitem/e/wbt/ddpnbu30_we1/obj/Release/net8.0/browser-wasm/wasm/for-publish/aot-instances.dll.bc -&gt; /root/helix/work/workitem/e/wbt/ddpnbu30_we1/obj/Release/net8.0/browser-wasm/wasm/for-publish/aot-instances.dll.o 
WasmApp.Native.targets(379,5): error : emcc: warning: linker setting ignored during compilation: &#39;EXPORT_ES6&#39; [-Wunused-command-line-argument] 
WasmApp.Native.targets(379,5): error : Killed
```

`WasmApp.Native.targets(379,5): error : emcc: error: &#39;/root/helix/work/workitem/e/dotnet-latest/packs/Microsoft.NET.Runtime.Emscripten.3.1.34.Sdk.linux-x64/8.0.0-rc.1.23411.1/tools/bin/clang++ -target wasm32-unknown-emscripten -fvisibility=default -mllvm -combiner-global-alias-analysis=false -mllvm -wasm-enable-sjlj -mllvm -disable-lsr -DEMSCRIPTEN --sysroot=/root/helix/work/workitem/e/dotnet-latest/packs/Microsoft.NET.Runtime.Emscripten.3.1.34.Cache.linux-x64/8.0.0-rc.1.23411.1/tools/emscripten/cache/sysroot -Xclang -iwithsysroot/include/fakesdl -Xclang -iwithsysroot/include/compat -msimd128 -O0 -g3 -fwasm-exceptions -c /root/helix/work/workitem/e/wbt/ddpnbu30_we1/obj/Release/net8.0/browser-wasm/wasm/for-publish/aot-instances.dll.bc -o /tmp/tmpeo8pzM.tmp&#39; failed (returned 137) [took 128.818s] `

[Changes since last passing rolling build](https://github.com/dotnet/runtime/compare/64a67710163...55828b9aa91).

This includes commits from @pavelsavara @ivanpovazan @vargaz @radekdoulik 
&lt;!-- Error message template  --&gt;
### Known Issue Error Message

Fill the error message using [step by step known issues guidance](https://github.com/dotnet/arcade/blob/main/Documentation/Projects/Build%20Analysis/KnownIssues.md#how-to-fill-out-a-known-issue-error-section).

&lt;!-- Use ErrorMessage for String.Contains matches. Use ErrorPattern for regex matches (single line/no backtracking). Set BuildRetry to `true` to retry builds with this error. Set ExcludeConsoleLog to `true` to skip helix logs analysis. --&gt;

```json
{
  &quot;ErrorMessage&quot;: &quot;error : Failed to compile &quot;,
  &quot;ErrorPattern&quot;: &quot;&quot;,
  &quot;BuildRetry&quot;: false,
  &quot;ExcludeConsoleLog&quot;: false
}
```


&lt;!--Known issue error report start --&gt;
### Report

|Build|Definition|Test|Pull Request|
|---|---|---|---|
|[2242043](https://dev.azure.com/dnceng/internal/_build/results?buildId=2242043)|dotnet-runtime|[wasmaot.x64.micro.net8.0.Partition27.WorkItemExecution](https://dev.azure.com/dnceng//internal/_build/results?buildId=2242043&amp;view=ms.vss-test-web.build-test-results-tab&amp;runId=51894143&amp;resultId=100058)||
|[371479](https://dev.azure.com/dnceng-public/public/_build/results?buildId=371479)|dotnet/runtime|[Workloads-Wasm.Build.Tests.SatelliteAssembliesTests.WorkItemExecution](https://dev.azure.com/dnceng-public/public/_build/results?buildId=371479&amp;view=ms.vss-test-web.build-test-results-tab&amp;runId=7882660&amp;resultId=100143)||
#### Summary
|24-Hour Hit Count|7-Day Hit Count|1-Month Count|
|---|---|---|
|0|2|2|
&lt;!--Known issue error report end --&gt;
&lt;!-- Known issue validation start --&gt;
 ### Known issue validation
**Build: :mag_right:** https://dev.azure.com/dnceng-public/public/_build/results?buildId=371479
**Error message validated:** `error : Failed to compile `
**Result validation: :white_check_mark:** Known issue matched with the provided build.
**Validation performed at:** 8/12/2023 3:57:22 PM UTC
&lt;!-- Known issue validation end --&gt;&lt;/issueSnippet&gt; ]</pre></div><style>
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

