# Introduction
The purpose of this cookbook is to show you how to properly benchmark TGI. For more background details and explanation, please check out this [popular blog](https://huggingface.co/blog/tgi-benchmarking) first.

## Setup
Make sure you have an environment with TGI installed; docker is a great choice.The commands here can be easily copied/pasted into a terminal, which might be even easier. Don't feel compelled to use Jupyter. If you just want to test this out, you can duplicate and use [derek-thomas/tgi-benchmark-space](https://huggingface.co/spaces/derek-thomas/tgi-benchmark-space). 

# TGI Launcher


```python
!text-generation-launcher --version
```

    text-generation-launcher 2.2.1-dev0
    

Below we can see the different settings for TGI. Be sure to read through them and decide which settings are most 
important for your use-case.

Here are some of the most important ones:
- `--model-id`
- `--quantize` Quantization saves memory, but does not always improve speed
- `--max-input-tokens` This allows TGI to optimize the prefilling operation
- `--max-total-tokens` In combination with the above TGI now knows what the max input and output tokens are
- `--max-batch-size` This lets TGI know how many requests it can process at once.

The last 3 together provide the necessary restrictions to optimize for your use-case. You can find a lot of performance improvements by setting these as appropriately as possible.


```python
!text-generation-launcher -h
```

    Text Generation Launcher
    
    [1m[4mUsage:[0m [1mtext-generation-launcher[0m [OPTIONS]
    
    [1m[4mOptions:[0m
          [1m--model-id[0m <MODEL_ID>
              The name of the model to load. Can be a MODEL_ID as listed on <https://hf.co/models> like `gpt2` or `OpenAssistant/oasst-sft-1-pythia-12b`. Or it can be a local directory containing the necessary files as saved by `save_pretrained(...)` methods of transformers [env: MODEL_ID=] [default: bigscience/bloom-560m]
          [1m--revision[0m <REVISION>
              The actual revision of the model if you're referring to a model on the hub. You can use a specific commit id or a branch like `refs/pr/2` [env: REVISION=]
          [1m--validation-workers[0m <VALIDATION_WORKERS>
              The number of tokenizer workers used for payload validation and truncation inside the router [env: VALIDATION_WORKERS=] [default: 2]
          [1m--sharded[0m <SHARDED>
              Whether to shard the model across multiple GPUs By default text-generation-inference will use all available GPUs to run the model. Setting it to `false` deactivates `num_shard` [env: SHARDED=] [possible values: true, false]
          [1m--num-shard[0m <NUM_SHARD>
              The number of shards to use if you don't want to use all GPUs on a given machine. You can use `CUDA_VISIBLE_DEVICES=0,1 text-generation-launcher... --num_shard 2` and `CUDA_VISIBLE_DEVICES=2,3 text-generation-launcher... --num_shard 2` to launch 2 copies with 2 shard each on a given machine with 4 GPUs for instance [env: NUM_SHARD=]
          [1m--quantize[0m <QUANTIZE>
              Whether you want the model to be quantized [env: QUANTIZE=] [possible values: awq, eetq, exl2, gptq, marlin, bitsandbytes, bitsandbytes-nf4, bitsandbytes-fp4, fp8]
          [1m--speculate[0m <SPECULATE>
              The number of input_ids to speculate on If using a medusa model, the heads will be picked up automatically Other wise, it will use n-gram speculation which is relatively free in terms of compute, but the speedup heavily depends on the task [env: SPECULATE=]
          [1m--dtype[0m <DTYPE>
              The dtype to be forced upon the model. This option cannot be used with `--quantize` [env: DTYPE=] [possible values: float16, bfloat16]
          [1m--trust-remote-code[0m
              Whether you want to execute hub modelling code. Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure no malicious code has been contributed in a newer revision [env: TRUST_REMOTE_CODE=]
          [1m--max-concurrent-requests[0m <MAX_CONCURRENT_REQUESTS>
              The maximum amount of concurrent requests for this particular deployment. Having a low limit will refuse clients requests instead of having them wait for too long and is usually good to handle backpressure correctly [env: MAX_CONCURRENT_REQUESTS=] [default: 128]
          [1m--max-best-of[0m <MAX_BEST_OF>
              This is the maximum allowed value for clients to set `best_of`. Best of makes `n` generations at the same time, and return the best in terms of overall log probability over the entire generated sequence [env: MAX_BEST_OF=] [default: 2]
          [1m--max-stop-sequences[0m <MAX_STOP_SEQUENCES>
              This is the maximum allowed value for clients to set `stop_sequences`. Stop sequences are used to allow the model to stop on more than just the EOS token, and enable more complex "prompting" where users can preprompt the model in a specific way and define their "own" stop token aligned with their prompt [env: MAX_STOP_SEQUENCES=] [default: 4]
          [1m--max-top-n-tokens[0m <MAX_TOP_N_TOKENS>
              This is the maximum allowed value for clients to set `top_n_tokens`. `top_n_tokens` is used to return information about the the `n` most likely tokens at each generation step, instead of just the sampled token. This information can be used for downstream tasks like for classification or ranking [env: MAX_TOP_N_TOKENS=] [default: 5]
          [1m--max-input-tokens[0m <MAX_INPUT_TOKENS>
              This is the maximum allowed input length (expressed in number of tokens) for users. The larger this value, the longer prompt users can send which can impact the overall memory required to handle the load. Please note that some models have a finite range of sequence they can handle. Default to min(max_position_embeddings - 1, 4095) [env: MAX_INPUT_TOKENS=]
          [1m--max-input-length[0m <MAX_INPUT_LENGTH>
              Legacy version of [`Args::max_input_tokens`] [env: MAX_INPUT_LENGTH=]
          [1m--max-total-tokens[0m <MAX_TOTAL_TOKENS>
              This is the most important value to set as it defines the "memory budget" of running clients requests. Clients will send input sequences and ask to generate `max_new_tokens` on top. with a value of `1512` users can send either a prompt of `1000` and ask for `512` new tokens, or send a prompt of `1` and ask for `1511` max_new_tokens. The larger this value, the larger amount each request will be in your RAM and the less effective batching can be. Default to min(max_position_embeddings, 4096) [env: MAX_TOTAL_TOKENS=]
          [1m--waiting-served-ratio[0m <WAITING_SERVED_RATIO>
              This represents the ratio of waiting queries vs running queries where you want to start considering pausing the running queries to include the waiting ones into the same batch. `waiting_served_ratio=1.2` Means when 12 queries are waiting and there's only 10 queries left in the current batch we check if we can fit those 12 waiting queries into the batching strategy, and if yes, then batching happens delaying the 10 running queries by a `prefill` run [env: WAITING_SERVED_RATIO=] [default: 0.3]
          [1m--max-batch-prefill-tokens[0m <MAX_BATCH_PREFILL_TOKENS>
              Limits the number of tokens for the prefill operation. Since this operation take the most memory and is compute bound, it is interesting to limit the number of requests that can be sent. Default to `max_input_tokens + 50` to give a bit of room [env: MAX_BATCH_PREFILL_TOKENS=]
          [1m--max-batch-total-tokens[0m <MAX_BATCH_TOTAL_TOKENS>
              **IMPORTANT** This is one critical control to allow maximum usage of the available hardware [env: MAX_BATCH_TOTAL_TOKENS=]
          [1m--max-waiting-tokens[0m <MAX_WAITING_TOKENS>
              This setting defines how many tokens can be passed before forcing the waiting queries to be put on the batch (if the size of the batch allows for it). New queries require 1 `prefill` forward, which is different from `decode` and therefore you need to pause the running batch in order to run `prefill` to create the correct values for the waiting queries to be able to join the batch [env: MAX_WAITING_TOKENS=] [default: 20]
          [1m--max-batch-size[0m <MAX_BATCH_SIZE>
              Enforce a maximum number of requests per batch Specific flag for hardware targets that do not support unpadded inference [env: MAX_BATCH_SIZE=]
          [1m--cuda-graphs[0m <CUDA_GRAPHS>
              Specify the batch sizes to compute cuda graphs for. Use "0" to disable. Default = "1,2,4,8,16,32" [env: CUDA_GRAPHS=]
          [1m--hostname[0m <HOSTNAME>
              The IP address to listen on [env: HOSTNAME=r-derek-thomas-tgi-benchmark-space-geij6846-b385a-lont4] [default: 0.0.0.0]
      [1m-p[0m, [1m--port[0m <PORT>
              The port to listen on [env: PORT=80] [default: 3000]
          [1m--shard-uds-path[0m <SHARD_UDS_PATH>
              The name of the socket for gRPC communication between the webserver and the shards [env: SHARD_UDS_PATH=] [default: /tmp/text-generation-server]
          [1m--master-addr[0m <MASTER_ADDR>
              The address the master shard will listen on. (setting used by torch distributed) [env: MASTER_ADDR=] [default: localhost]
          [1m--master-port[0m <MASTER_PORT>
              The address the master port will listen on. (setting used by torch distributed) [env: MASTER_PORT=] [default: 29500]
          [1m--huggingface-hub-cache[0m <HUGGINGFACE_HUB_CACHE>
              The location of the huggingface hub cache. Used to override the location if you want to provide a mounted disk for instance [env: HUGGINGFACE_HUB_CACHE=]
          [1m--weights-cache-override[0m <WEIGHTS_CACHE_OVERRIDE>
              The location of the huggingface hub cache. Used to override the location if you want to provide a mounted disk for instance [env: WEIGHTS_CACHE_OVERRIDE=]
          [1m--disable-custom-kernels[0m
              For some models (like bloom), text-generation-inference implemented custom cuda kernels to speed up inference. Those kernels were only tested on A100. Use this flag to disable them if you're running on different hardware and encounter issues [env: DISABLE_CUSTOM_KERNELS=]
          [1m--cuda-memory-fraction[0m <CUDA_MEMORY_FRACTION>
              Limit the CUDA available memory. The allowed value equals the total visible memory multiplied by cuda-memory-fraction [env: CUDA_MEMORY_FRACTION=] [default: 1.0]
          [1m--rope-scaling[0m <ROPE_SCALING>
              Rope scaling will only be used for RoPE models and allow rescaling the position rotary to accomodate for larger prompts [env: ROPE_SCALING=] [possible values: linear, dynamic]
          [1m--rope-factor[0m <ROPE_FACTOR>
              Rope scaling will only be used for RoPE models See `rope_scaling` [env: ROPE_FACTOR=]
          [1m--json-output[0m
              Outputs the logs in JSON format (useful for telemetry) [env: JSON_OUTPUT=]
          [1m--otlp-endpoint[0m <OTLP_ENDPOINT>
              [env: OTLP_ENDPOINT=]
          [1m--otlp-service-name[0m <OTLP_SERVICE_NAME>
              [env: OTLP_SERVICE_NAME=] [default: text-generation-inference.router]
          [1m--cors-allow-origin[0m <CORS_ALLOW_ORIGIN>
              [env: CORS_ALLOW_ORIGIN=]
          [1m--api-key[0m <API_KEY>
              [env: API_KEY=]
          [1m--watermark-gamma[0m <WATERMARK_GAMMA>
              [env: WATERMARK_GAMMA=]
          [1m--watermark-delta[0m <WATERMARK_DELTA>
              [env: WATERMARK_DELTA=]
          [1m--ngrok[0m
              Enable ngrok tunneling [env: NGROK=]
          [1m--ngrok-authtoken[0m <NGROK_AUTHTOKEN>
              ngrok authentication token [env: NGROK_AUTHTOKEN=]
          [1m--ngrok-edge[0m <NGROK_EDGE>
              ngrok edge [env: NGROK_EDGE=]
          [1m--tokenizer-config-path[0m <TOKENIZER_CONFIG_PATH>
              The path to the tokenizer config file. This path is used to load the tokenizer configuration which may include a `chat_template`. If not provided, the default config will be used from the model hub [env: TOKENIZER_CONFIG_PATH=]
          [1m--disable-grammar-support[0m
              Disable outlines grammar constrained generation. This is a feature that allows you to generate text that follows a specific grammar [env: DISABLE_GRAMMAR_SUPPORT=]
      [1m-e[0m, [1m--env[0m
              Display a lot of information about your runtime environment
          [1m--max-client-batch-size[0m <MAX_CLIENT_BATCH_SIZE>
              Control the maximum number of inputs that a client can send in a single request [env: MAX_CLIENT_BATCH_SIZE=] [default: 4]
          [1m--lora-adapters[0m <LORA_ADAPTERS>
              Lora Adapters a list of adapter ids i.e. `repo/adapter1,repo/adapter2` to load during startup that will be available to callers via the `adapter_id` field in a request [env: LORA_ADAPTERS=]
          [1m--usage-stats[0m <USAGE_STATS>
              Control if anonymous usage stats are collected. Options are "on", "off" and "no-stack" Defaul is on [env: USAGE_STATS=] [default: on] [possible values: on, off, no-stack]
      [1m-h[0m, [1m--help[0m
              Print help (see more with '--help')
      [1m-V[0m, [1m--version[0m
              Print version
    

We can launch directly from the cookbook since we dont need the command to be interactive.

We will just be using defaults in this cookbook as the intent is to understand the benchmark tool.

These parameters were changed if you're running on a Space because we don't want to conflict with the Spaces server:
- `--hostname`
- `--port`

Feel free to change or remove them based on your requirements.


```python
!RUST_BACKTRACE=1 \
text-generation-launcher \
--model-id astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit \
--quantize gptq \
--hostname 0.0.0.0 \
--port 1337
```

    [2m2024-08-16T12:07:56.411768Z[0m [32m INFO[0m [2mtext_generation_launcher[0m[2m:[0m Args {
        model_id: "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit",
        revision: None,
        validation_workers: 2,
        sharded: None,
        num_shard: None,
        quantize: Some(
            Gptq,
        ),
        speculate: None,
        dtype: None,
        trust_remote_code: false,
        max_concurrent_requests: 128,
        max_best_of: 2,
        max_stop_sequences: 4,
        max_top_n_tokens: 5,
        max_input_tokens: None,
        max_input_length: None,
        max_total_tokens: None,
        waiting_served_ratio: 0.3,
        max_batch_prefill_tokens: None,
        max_batch_total_tokens: None,
        max_waiting_tokens: 20,
        max_batch_size: None,
        cuda_graphs: None,
        hostname: "0.0.0.0",
        port: 1337,
        shard_uds_path: "/tmp/text-generation-server",
        master_addr: "localhost",
        master_port: 29500,
        huggingface_hub_cache: None,
        weights_cache_override: None,
        disable_custom_kernels: false,
        cuda_memory_fraction: 1.0,
        rope_scaling: None,
        rope_factor: None,
        json_output: false,
        otlp_endpoint: None,
        otlp_service_name: "text-generation-inference.router",
        cors_allow_origin: [],
        api_key: None,
        watermark_gamma: None,
        watermark_delta: None,
        ngrok: false,
        ngrok_authtoken: None,
        ngrok_edge: None,
        tokenizer_config_path: None,
        disable_grammar_support: false,
        env: false,
        max_client_batch_size: 4,
        lora_adapters: None,
        usage_stats: On,
    }
    [2m2024-08-16T12:07:56.411941Z[0m [32m INFO[0m [2mhf_hub[0m[2m:[0m Token file not found "/data/token"    
    [2Kconfig.json [00:00:00] [████████████████████████] 1021 B/1021 B 50.70 KiB/s (0s)[2m2024-08-16T12:07:56.458451Z[0m [32m INFO[0m [2mtext_generation_launcher[0m[2m:[0m Model supports up to 8192 but tgi will now set its default to 4096 instead. This is to save VRAM by refusing large prompts in order to allow more users on the same hardware. You can increase that size using `--max-batch-prefill-tokens=8242 --max-total-tokens=8192 --max-input-tokens=8191`.
    [2m2024-08-16T12:07:56.458473Z[0m [32m INFO[0m [2mtext_generation_launcher[0m[2m:[0m Default `max_input_tokens` to 4095
    [2m2024-08-16T12:07:56.458480Z[0m [32m INFO[0m [2mtext_generation_launcher[0m[2m:[0m Default `max_total_tokens` to 4096
    [2m2024-08-16T12:07:56.458487Z[0m [32m INFO[0m [2mtext_generation_launcher[0m[2m:[0m Default `max_batch_prefill_tokens` to 4145
    [2m2024-08-16T12:07:56.458494Z[0m [32m INFO[0m [2mtext_generation_launcher[0m[2m:[0m Using default cuda graphs [1, 2, 4, 8, 16, 32]
    [2m2024-08-16T12:07:56.458606Z[0m [32m INFO[0m [1mdownload[0m: [2mtext_generation_launcher[0m[2m:[0m Starting check and download process for astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit
    [2m2024-08-16T12:07:59.750101Z[0m [32m INFO[0m [2mtext_generation_launcher[0m[2m:[0m Download file: model.safetensors
    ^C
    [2m2024-08-16T12:08:09.101893Z[0m [32m INFO[0m [1mdownload[0m: [2mtext_generation_launcher[0m[2m:[0m Terminating download
    [2m2024-08-16T12:08:09.102368Z[0m [32m INFO[0m [1mdownload[0m: [2mtext_generation_launcher[0m[2m:[0m Waiting for download to gracefully shutdown
    

# TGI Benchmark
Now lets learn how to launch the benchmark tool!

Here we can see the different settings for TGI Benchmark.

Here are some of the more important TGI Benchmark settings:

- `--tokenizer-name` This is required so the tool knows what tokenizer to use
- `--batch-size` This is important for load testing. We should use enough values to see what happens to throughput and latency. Do note that batch-size in the context of the benchmarking tool is number of virtual users. 
- `--sequence-length` AKA input tokens, it is important to match your use-case needs
- `--decode-length` AKA output tokens, it is important to match your use-case needs
- `--runs` 10 is the default

<blockquote style="border-left: 5px solid #80CBC4; background: #263238; color: #CFD8DC; padding: 0.5em 1em; margin: 1em 0;">
  <strong>💡 Tip:</strong> Use a low number for <code style="background: #37474F; color: #FFFFFF; padding: 2px 4px; border-radius: 4px;">--runs</code> when you are exploring but a higher number as you finalize to get more precise statistics
</blockquote>



```python
!text-generation-benchmark -h
```

    Text Generation Benchmarking tool
    
    [1m[4mUsage:[0m [1mtext-generation-benchmark[0m [OPTIONS] [1m--tokenizer-name[0m <TOKENIZER_NAME>
    
    [1m[4mOptions:[0m
      [1m-t[0m, [1m--tokenizer-name[0m <TOKENIZER_NAME>
              The name of the tokenizer (as in model_id on the huggingface hub, or local path) [env: TOKENIZER_NAME=]
          [1m--revision[0m <REVISION>
              The revision to use for the tokenizer if on the hub [env: REVISION=] [default: main]
      [1m-b[0m, [1m--batch-size[0m <BATCH_SIZE>
              The various batch sizes to benchmark for, the idea is to get enough batching to start seeing increased latency, this usually means you're moving from memory bound (usual as BS=1) to compute bound, and this is a sweet spot for the maximum batch size for the model under test
      [1m-s[0m, [1m--sequence-length[0m <SEQUENCE_LENGTH>
              This is the initial prompt sent to the text-generation-server length in token. Longer prompt will slow down the benchmark. Usually the latency grows somewhat linearly with this for the prefill step [env: SEQUENCE_LENGTH=] [default: 10]
      [1m-d[0m, [1m--decode-length[0m <DECODE_LENGTH>
              This is how many tokens will be generated by the server and averaged out to give the `decode` latency. This is the *critical* number you want to optimize for LLM spend most of their time doing decoding [env: DECODE_LENGTH=] [default: 8]
      [1m-r[0m, [1m--runs[0m <RUNS>
              How many runs should we average from [env: RUNS=] [default: 10]
      [1m-w[0m, [1m--warmups[0m <WARMUPS>
              Number of warmup cycles [env: WARMUPS=] [default: 1]
      [1m-m[0m, [1m--master-shard-uds-path[0m <MASTER_SHARD_UDS_PATH>
              The location of the grpc socket. This benchmark tool bypasses the router completely and directly talks to the gRPC processes [env: MASTER_SHARD_UDS_PATH=] [default: /tmp/text-generation-server-0]
          [1m--temperature[0m <TEMPERATURE>
              Generation parameter in case you want to specifically test/debug particular decoding strategies, for full doc refer to the `text-generation-server` [env: TEMPERATURE=]
          [1m--top-k[0m <TOP_K>
              Generation parameter in case you want to specifically test/debug particular decoding strategies, for full doc refer to the `text-generation-server` [env: TOP_K=]
          [1m--top-p[0m <TOP_P>
              Generation parameter in case you want to specifically test/debug particular decoding strategies, for full doc refer to the `text-generation-server` [env: TOP_P=]
          [1m--typical-p[0m <TYPICAL_P>
              Generation parameter in case you want to specifically test/debug particular decoding strategies, for full doc refer to the `text-generation-server` [env: TYPICAL_P=]
          [1m--repetition-penalty[0m <REPETITION_PENALTY>
              Generation parameter in case you want to specifically test/debug particular decoding strategies, for full doc refer to the `text-generation-server` [env: REPETITION_PENALTY=]
          [1m--frequency-penalty[0m <FREQUENCY_PENALTY>
              Generation parameter in case you want to specifically test/debug particular decoding strategies, for full doc refer to the `text-generation-server` [env: FREQUENCY_PENALTY=]
          [1m--watermark[0m
              Generation parameter in case you want to specifically test/debug particular decoding strategies, for full doc refer to the `text-generation-server` [env: WATERMARK=]
          [1m--do-sample[0m
              Generation parameter in case you want to specifically test/debug particular decoding strategies, for full doc refer to the `text-generation-server` [env: DO_SAMPLE=]
          [1m--top-n-tokens[0m <TOP_N_TOKENS>
              Generation parameter in case you want to specifically test/debug particular decoding strategies, for full doc refer to the `text-generation-server` [env: TOP_N_TOKENS=]
      [1m-h[0m, [1m--help[0m
              Print help (see more with '--help')
      [1m-V[0m, [1m--version[0m
              Print version
    

Here is an example command. Notice that I add the batch sizes of interest repeatedly to make sure all of them are used 
by the benchmark tool. I'm also considering which batch sizes are important based on estimated user activity.

<blockquote style="border-left: 5px solid #FFAB91; background: #37474F; color: #FFCCBC; padding: 0.5em 1em; margin: 1em 0;">
  <strong>⚠️ Warning:</strong> Please note that the TGI Benchmark tool is designed to work in a terminal, not a jupyter notebook. This means you will need to copy/paste the command in a jupyter terminal tab. I am putting it here for convenience.
</blockquote>



```python
!text-generation-benchmark \
--tokenizer-name astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit \
--sequence-length 70 \
--decode-length 50 \
--batch-size 1 \
--batch-size 2 \
--batch-size 4 \
--batch-size 8 \
--batch-size 16 \
--batch-size 32 \
--batch-size 64 \
--batch-size 128 
```


```python

```
