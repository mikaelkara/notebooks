# AINetwork Toolkit

>[AI Network](https://www.ainetwork.ai/build-on-ain) is a layer 1 blockchain designed to accommodate large-scale AI models, utilizing a decentralized GPU network powered by the [$AIN token](https://www.ainetwork.ai/token), enriching AI-driven `NFTs` (`AINFTs`).
>
>The `AINetwork Toolkit` is a set of tools for interacting with the [AINetwork Blockchain](https://www.ainetwork.ai/public/whitepaper.pdf). These tools allow you to transfer `AIN`, read and write values, create apps, and set permissions for specific paths within the blockchain database.

## Installing dependencies

Before using the AINetwork Toolkit, you need to install the ain-py package. You can install it with pip:



```python
%pip install --upgrade --quiet  ain-py langchain-community
```

## Set environmental variables

You need to set the `AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY` environmental variable to your AIN Blockchain Account Private Key.


```python
import os

os.environ["AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY"] = ""
```

### Get AIN Blockchain private key


```python
import os

from ain.account import Account

if os.environ.get("AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY", None):
    account = Account(os.environ["AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY"])
else:
    account = Account.create()
    os.environ["AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY"] = account.private_key
    print(
        f"""
address: {account.address}
private_key: {account.private_key}
"""
    )
# IMPORTANT: If you plan to use this account in the future, make sure to save the
#  private key in a secure place. Losing access to your private key means losing
#  access to your account.
```

    
    address: 0x5BEB4Defa2ccc274498416Fd7Cb34235DbC122Ac
    private_key: f5e2f359bb6b7836a2ac70815473d1a290c517f847d096f5effe818de8c2cf14
    
    

## Initialize the AINetwork Toolkit

You can initialize the AINetwork Toolkit like this:


```python
from langchain_community.agent_toolkits.ainetwork.toolkit import AINetworkToolkit

toolkit = AINetworkToolkit()
tools = toolkit.get_tools()
address = tools[0].interface.wallet.defaultAccount.address
```

## Initialize the Agent with the AINetwork Toolkit

You can initialize the agent with the AINetwork Toolkit like this:


```python
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
)
```

## Example Usage

Here are some examples of how you can use the agent with the AINetwork Toolkit:

### Define App name to test


```python
appName = f"langchain_demo_{address.lower()}"
```

### Create an app in the AINetwork Blockchain database


```python
print(
    agent.run(
        f"Create an app in the AINetwork Blockchain database with the name {appName}"
    )
)
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `AINappOps` with `{'type': 'SET_ADMIN', 'appName': 'langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac'}`
    
    
    [0m[36;1m[1;3m{"tx_hash": "0x018846d6a9fc111edb1a2246ae2484ef05573bd2c584f3d0da155fa4b4936a9e", "result": {"gas_amount_total": {"bandwidth": {"service": 4002, "app": {"langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac": 2}}, "state": {"service": 1640}}, "gas_cost_total": 0, "func_results": {"_createApp": {"op_results": {"0": {"path": "/apps/langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac", "result": {"code": 0, "bandwidth_gas_amount": 1}}, "1": {"path": "/apps/langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac", "result": {"code": 0, "bandwidth_gas_amount": 1}}, "2": {"path": "/manage_app/langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac/config/admin", "result": {"code": 0, "bandwidth_gas_amount": 1}}}, "code": 0, "bandwidth_gas_amount": 2000}}, "code": 0, "bandwidth_gas_amount": 2001, "gas_amount_charged": 5642}}[0m[32;1m[1;3mThe app with the name "langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac" has been created in the AINetwork Blockchain database.[0m
    
    [1m> Finished chain.[0m
    The app with the name "langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac" has been created in the AINetwork Blockchain database.
    

### Set a value at a given path in the AINetwork Blockchain database


```python
print(
    agent.run(f"Set the value {{1: 2, '34': 56}} at the path /apps/{appName}/object .")
)
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `AINvalueOps` with `{'type': 'SET', 'path': '/apps/langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac/object', 'value': {'1': 2, '34': 56}}`
    
    
    [0m[33;1m[1;3m{"tx_hash": "0x3d1a16d9808830088cdf4d37f90f4b1fa1242e2d5f6f983829064f45107b5279", "result": {"gas_amount_total": {"bandwidth": {"service": 0, "app": {"langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac": 1}}, "state": {"service": 0, "app": {"langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac": 674}}}, "gas_cost_total": 0, "code": 0, "bandwidth_gas_amount": 1, "gas_amount_charged": 0}}[0m[32;1m[1;3mThe value {1: 2, '34': 56} has been set at the path /apps/langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac/object.[0m
    
    [1m> Finished chain.[0m
    The value {1: 2, '34': 56} has been set at the path /apps/langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac/object.
    

### Set permissions for a path in the AINetwork Blockchain database


```python
print(
    agent.run(
        f"Set the write permissions for the path /apps/{appName}/user/$from with the"
        " eval string auth.addr===$from ."
    )
)
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `AINruleOps` with `{'type': 'SET', 'path': '/apps/langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac/user/$from', 'eval': 'auth.addr===$from'}`
    
    
    [0m[38;5;200m[1;3m{"tx_hash": "0x37d5264e580f6a217a347059a735bfa9eb5aad85ff28a95531c6dc09252664d2", "result": {"gas_amount_total": {"bandwidth": {"service": 0, "app": {"langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac": 1}}, "state": {"service": 0, "app": {"langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac": 712}}}, "gas_cost_total": 0, "code": 0, "bandwidth_gas_amount": 1, "gas_amount_charged": 0}}[0m[32;1m[1;3mThe write permissions for the path `/apps/langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac/user/$from` have been set with the eval string `auth.addr===$from`.[0m
    
    [1m> Finished chain.[0m
    The write permissions for the path `/apps/langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac/user/$from` have been set with the eval string `auth.addr===$from`.
    

### Retrieve the permissions for a path in the AINetwork Blockchain database


```python
print(agent.run(f"Retrieve the permissions for the path /apps/{appName}."))
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `AINownerOps` with `{'type': 'GET', 'path': '/apps/langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac'}`
    
    
    [0m[33;1m[1;3m{".owner": {"owners": {"0x5BEB4Defa2ccc274498416Fd7Cb34235DbC122Ac": {"branch_owner": true, "write_function": true, "write_owner": true, "write_rule": true}}}}[0m[32;1m[1;3mThe permissions for the path /apps/langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac are as follows:
    
    - Address: 0x5BEB4Defa2ccc274498416Fd7Cb34235DbC122Ac
      - branch_owner: true
      - write_function: true
      - write_owner: true
      - write_rule: true[0m
    
    [1m> Finished chain.[0m
    The permissions for the path /apps/langchain_demo_0x5beb4defa2ccc274498416fd7cb34235dbc122ac are as follows:
    
    - Address: 0x5BEB4Defa2ccc274498416Fd7Cb34235DbC122Ac
      - branch_owner: true
      - write_function: true
      - write_owner: true
      - write_rule: true
    

### Get AIN from faucet


```python
!curl http://faucet.ainetwork.ai/api/test/{address}/
```

    {"result":"0x0eb07b67b7d0a702cb60e865d3deafff3070d8508077ef793d69d6819fd92ea3","time":1692348112376}

### Get AIN Balance


```python
print(agent.run(f"Check AIN balance of {address}"))
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `AINvalueOps` with `{'type': 'GET', 'path': '/accounts/0x5BEB4Defa2ccc274498416Fd7Cb34235DbC122Ac/balance'}`
    
    
    [0m[33;1m[1;3m100[0m[32;1m[1;3mThe AIN balance of address 0x5BEB4Defa2ccc274498416Fd7Cb34235DbC122Ac is 100 AIN.[0m
    
    [1m> Finished chain.[0m
    The AIN balance of address 0x5BEB4Defa2ccc274498416Fd7Cb34235DbC122Ac is 100 AIN.
    

### Transfer AIN


```python
print(
    agent.run(
        "Transfer 100 AIN to the address 0x19937b227b1b13f29e7ab18676a89ea3bdea9c5b"
    )
)
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `AINtransfer` with `{'address': '0x19937b227b1b13f29e7ab18676a89ea3bdea9c5b', 'amount': 100}`
    
    
    [0m[36;1m[1;3m{"tx_hash": "0xa59d15d23373bcc00e413ac8ba18cb016bb3bdd54058d62606aec688c6ad3d2e", "result": {"gas_amount_total": {"bandwidth": {"service": 3}, "state": {"service": 866}}, "gas_cost_total": 0, "func_results": {"_transfer": {"op_results": {"0": {"path": "/accounts/0x5BEB4Defa2ccc274498416Fd7Cb34235DbC122Ac/balance", "result": {"code": 0, "bandwidth_gas_amount": 1}}, "1": {"path": "/accounts/0x19937B227b1b13f29e7AB18676a89EA3BDEA9C5b/balance", "result": {"code": 0, "bandwidth_gas_amount": 1}}}, "code": 0, "bandwidth_gas_amount": 0}}, "code": 0, "bandwidth_gas_amount": 1, "gas_amount_charged": 869}}[0m[32;1m[1;3mThe transfer of 100 AIN to the address 0x19937b227b1b13f29e7ab18676a89ea3bdea9c5b was successful. The transaction hash is 0xa59d15d23373bcc00e413ac8ba18cb016bb3bdd54058d62606aec688c6ad3d2e.[0m
    
    [1m> Finished chain.[0m
    The transfer of 100 AIN to the address 0x19937b227b1b13f29e7ab18676a89ea3bdea9c5b was successful. The transaction hash is 0xa59d15d23373bcc00e413ac8ba18cb016bb3bdd54058d62606aec688c6ad3d2e.
    
