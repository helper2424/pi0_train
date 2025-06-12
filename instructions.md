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

Last image
```
docker build  --platform linux/amd64 -t helper2424/openpi_master_with_config:latest --target development -f Dockerfile .
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
vastai search offers 'gpu_name=H100_NVL num_gpus=8' --limit 10 --order 'dph_total,reliability-total' 
```

If u want using H100, u can use the following command:

```bash
vastai search offers 'gpu_name=H100_SXM num_gpus=8' --limit 10 --order 'dph_total,reliability-total' 
```

The response example is the following:

```json
ID        CUDA   N  Model     PCIE  cpu_ghz  vCPUs    RAM  Disk  $/hr    DLP   DLP/$   score  NV Driver  Net_up  Net_down  R     Max_Days  mach_id  status    host_id  ports  country        
19115553  12.6  1x  RTX_4090  11.4  3.3      14.0    32.1  606   0.2281  97.8  428.78  324.3  560.35.03  445.5   781.3     99.6  26.9      9585     verified  58023    49     South_Korea,_KR      
```

The first column is the ID of the offer. We can now rent it with the following command:

```bash
vastai create instance 20756105 --image helper2424/openpi_workable:latest --env '-p 5678:5678 -p 5679:5679/udp -p 5680:5680' --disk 200 --ssh --jupyter --jupyter-lab --direct
```

The result will be like a following:
```bash
Started. {'success': True, 'new_contract': 20745916}
```

Check the instance status with the following command:

```bash
vastai show instance 21062054
```

Whenever the status is `ready` u can connect to the instance with the following command:

```bash
ssh $(vastai ssh-url 21062054)
```

### 2.2. Not Vastai

I you use cloud provider that doen't support docker images, you can install docker instantly in the instance and use preparred by me docker image.

How to install docker in limux instance:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
```

Srart the docker daemon:

```bash
sudo systemctl start docker
```

Run the docker container with bash, also don't forget to provide enoguht resources and give accces to all GPUS:

```bash
docker run -it --gpus all --rm --name openpi -p 5678:5678 -p 5679:5679/udp -p 5680:5680 helper2424/openpi:latest bash
```

```bash
docker run -it --gpus all --rm --name openpi -p 5678:5678 -p 5679:5679/udp -p 5680:5680 -v /app:/app helper2424/openpi_master_with_config:latest bash
```
### 2.3. Go to the app dir

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
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py demo3_frames_grab3 --exp-name=my_experiment --overwrite --save_interval=2 --log_interval=100 --batch_size=32 --num_workers=4
```

Too debug openpi - use the following command:
```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 uv run scripts/train.py debug  --exp-name=my_experiment --overwrite --save_interval=2 --log_interval=10   
```

To train on the real robot - use the following command:

Train wiht 1 card

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name demo3_frames_grab3
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py demo3_frames_grab3 --exp-name=check_pi0 --overwrite --save_interval=2 --log_interval=100 --batch_size=2 --num_workers=4
```

Train with 8 H100

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name demo3_frames_grab3
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py demo3_frames_grab3 --exp-name=speed_up_pi0_8gpu_256_batch --overwrite --save_interval=1000 --log_interval=100 --batch_size=64 --num_workers=4 --fsdp_devices=8

uv run scripts/train.py demo3_frames_grab3 --exp-name=check_a1000 --overwrite --save_interval=2 --log_interval=100 --checkpoint_base_dir=/checkpoints --batch_size=1
```


### 3.4. Commands to run training

```bash
cd /
rm -rf openpi
rm -rf app
git clone https://github.com/helper2424/openpi.git
cd openpi
git fetch origin workable_commit_with_sam_policies_and_config
git checkout d42caa9144b63506a4a0a2e4974a38531b355840
 vi /.venv/lib64/python3.11/site-packages/lerobot/common/datasets/utils.py 
GIT_LFS_SKIP_SMUDGE=1  uv sync
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name demo3_frames_grab3
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py demo3_frames_grab3 --exp-name=speed_up_pi0_8gpu_128_batch --overwrite --save_interval=1000 --log_interval=100 --batch_size=128 --num_workers=8 --fsdp_devices=8 --num_train_steps=7500
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


### Cloud lambda flow

To start the tmux session, u need to run the following command:

```bash
tmux
```

If u join the second time, u need to run the following command to attach the session:

```bash
tmux attach -t 0
```

Run the docker image:

```bash
sudo docker run -it --gpus all --rm --name openpi -p 5678:5678 -p 5679:5679/udp -p 5680:5680 -v /checkpoints:/checkpoints helper2424/openpi_master_with_config:latest bash
```


For 8 gpu with 256 batch size
```bash
cd /app
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name demo3_frames_grab3
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py demo3_frames_grab3 --exp-name=speed_up_pi0_8gpu_256_batch --overwrite --save_interval=1000 --log_interval=100 --batch_size=256 --num_workers=8 --fsdp_devices=8 --num_train_steps=7500 --checkpoint_base_dir=/checkpoints
```

For 1 gpu with 32 batch size
```bash
cd /app
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name demo3_frames_grab3
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py demo3_frames_grab3 --exp-name=speed_up_pi0_1gpu_32_batch --overwrite --save_interval=1000 --log_interval=100 --batch_size=32 --num_workers=8 --fsdp_devices=1 --num_train_steps=30000 --checkpoint_base_dir=/checkpoints
```