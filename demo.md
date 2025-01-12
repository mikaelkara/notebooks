```
# Set the provider configuration
Import-Module ./ -Force
Set-TuneProvider @splat
```


```
# Grab a tuning file
Get-ChildItem -Recurse totbot-tee-tune.jsonl -OutVariable totbot
```


```
# Check the validity of the training file
$totbot | Test-TuneFile -OutVariable testFile
```


```
# Upload the file
$totbot | Send-TuneFile -OutVariable upload
```


```
# check out the uploaded file
Get-TuneFile -Id $upload.id -OutVariable file
```


```
# Start a tuning job
Start-TuneJob -FileId $upload.id -OutVariable startjob
```


```
# Wait for it to complete
Wait-TuneJob -Id $startjob.id -OutVariable job
```


```
# Get that job
Get-TuneJob | Select-Object -First 1 -OutVariable job
```


```
# Retrieve events for a specific fine-tuning job
$job | Get-TuneJobEvent -OutVariable tunevent
```


```
# Retrieve a custom model
Get-TuneModel -Custom
```


```
# Delete a custom model
Get-TuneModel -Custom | Select-Object -First 1 |
Remove-TuneModel -Confirm:$false
```


```
# Retrieve your preferred default model
Get-TuneModelDefault -OutVariable defaultModel
```


```
# Measure token count for a given text
Measure-TuneToken -InputObject "Have some finetuna" -Model cl100k_base
```


```
# Get the current provider configuration
Get-TuneProvider
```
