## Training cloud pi0


## 0. Build the docker image

The dockerfile is in the root of the repository - `openpi/Dockerfile`.

You should copy it in your pi0 repo locally with the following command:
```bash
wget https://raw.githubusercontent.com/helper2424/pi0_train/7f81b1d76fe119237fe2f75b5a7086dbebadab57/openpi/Dockerfile
```

Command to build it

```bash
docker build  --platform linux/amd64 -t helper2424/openpi:latest --target development .
```

U can use your own dockerhub account, mine is `helper2424`.

Push it to your dockerhub account:

```bash
docker push helper2424/openpi:latest
```

## 1. Run the docker image

To run the pi0 I will use Vast AI. Before that you need install the Vast AI CLI.

```bash
pip install vastai
```

### 1.1. Create a Vast AI account

Go to https://vast.ai/ and create an account.

### 1.2. Login in Vast AI

To make it you need to create a token in your account settings. And setup the token locally with the following command:

```bash
vastai set api-key <your key>
```

### 2. Run the training

### 2.1. Create a Vast AI instance

Search for a suitable instance with the following command:

```bash
vastai search offers 'gpu_name=RTX_4090 num_gpus=1' --limit 5 --order 'dph_total,reliability-total' 
```

If u want using H100, u can use the following command:

```bash
vastai search offers 'gpu_name=H100_SXM num_gpus=1' --limit 5 --order 'dph_total,reliability-total' 
```

The response example is the following:

```json
ID        CUDA   N  Model     PCIE  cpu_ghz  vCPUs    RAM  Disk  $/hr    DLP   DLP/$   score  NV Driver  Net_up  Net_down  R     Max_Days  mach_id  status    host_id  ports  country        
19115553  12.6  1x  RTX_4090  11.4  3.3      14.0    32.1  606   0.2281  97.8  428.78  324.3  560.35.03  445.5   781.3     99.6  26.9      9585     verified  58023    49     South_Korea,_KR      
```

The first column is the ID of the offer. We can now rent it with the following command:

```bash
vastai create instance 19688417 --image helper2424/openpi:latest --env '-p 5678:5678 -p 5679:5679/udp -p 5680:5680' --disk 100 --jupyter --ssh --jupyter-lab --direct
```

The result will be like a following:
```bash
Started. {'success': True, 'new_contract': 19115553}
```

Check the instance status with the following command:

```bash
vastai show instance 20700028
```

Whenever the status is `ready` u can connect to the instance with the following command:

```bash
ssh $(vastai ssh-url 20700028)
```

### 2.2. Run the training

Firstly calculate the normalization statistics for the training data.

```bash
uv run scripts/compute_norm_stats.py --config-name sam_frames4_fast
```

Run the training with the following command:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py sam_frames4_fast --exp-name=my_experiment --overwrite

```



### 4. Clean up

To stop the instance u can use the following command:

```bash
vastai stop instance 20695198
```

To delete the instance u can use the following command:

```bash
vastai destroy instance 20695198
```