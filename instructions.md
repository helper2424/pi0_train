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
vastai search offers 'gpu_name=H100_SXM num_gpus=4' --limit 5 --order 'dph_total,reliability-total' 
```

The response example is the following:

```json
ID        CUDA   N  Model     PCIE  cpu_ghz  vCPUs    RAM  Disk  $/hr    DLP   DLP/$   score  NV Driver  Net_up  Net_down  R     Max_Days  mach_id  status    host_id  ports  country        
19115553  12.6  1x  RTX_4090  11.4  3.3      14.0    32.1  606   0.2281  97.8  428.78  324.3  560.35.03  445.5   781.3     99.6  26.9      9585     verified  58023    49     South_Korea,_KR      
```

The first column is the ID of the offer. We can now rent it with the following command:

```bash
vastai create instance 19449554 --image helper2424/openpi:latest --env '-p 5678:5678 -p 5679:5679/udp -p 5680:5680' --disk 100 --jupyter --ssh --jupyter-lab --direct
```

The result will be like a following:
```bash
Started. {'success': True, 'new_contract': 20745916}
```

Check the instance status with the following command:

```bash
vastai show instance 20812830
```

Whenever the status is `ready` u can connect to the instance with the following command:

```bash
ssh $(vastai ssh-url 20812830)
```

### 2.2. Go to the app dir

```bash
cd /app
```

### 2.3. Run the training

Firstly calculate the normalization statistics for the training data.

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name demo3_frames_grab3
```

The `CUDA_VISIBLE_DEVICES=0` is important to use only one GPU for the normalization statistics computation. In other case the script fails in machiens with several GPUS with the following error:

```bash
 raise ValueError(f"One of {what_aval}{name_str} was given the sharding "
ValueError: One of device_put args was given the sharding of NamedSharding(mesh=Mesh('B': 4, axis_types=(Auto,)), spec=PartitionSpec('B',), memory_kind=device), which implies that the global size of its dimension 0 should be divisible by 4, but it is equal to 1 (full shape: (1, 10, 7))
```

Run the training with the following command:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py demo3_frames_grab3 --exp-name=my_experiment --overwrite --fsdp_devices=4 --save_interval=5000 --log_interval=100 --batch_size=128 --num_workers=$(nproc)
```



### 4. Clean up

To stop the instance u can use the following command:

```bash
vastai stop instance 20695198
```

To delete the instance u can use the following command:

```bash
vastai destroy instance 20795024
```