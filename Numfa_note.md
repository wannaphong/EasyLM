# Numfa training note

```shell
gcloud alpha compute tpus queued-resources create node-1 \
--node-id nodel_id \
--project project \
--zone us-central2-b \
--accelerator-type v4-64 \
--runtime-version tpu-ubuntu2204-base

```
Wait until have the resource

```
gcloud compute tpus tpu-vm ssh nodel_id \
--zone us-central2-b \
--worker=all \
--command='sudo apt-get update && sudo apt-get install -y \
    build-essential \
    python-is-python3 \
    tmux \
    htop \
    git \
    nodejs \
    bmon \
    p7zip-full \
    nfs-common'

gcloud compute tpus tpu-vm ssh nodel_id \
--zone us-central2-b \
--worker=all \
--command='sudo apt-get update && sudo apt-get install -y \
    git-lfs'

gcloud compute tpus tpu-vm ssh nodel_id \
--zone us-central2-b \
--worker=all \
--command='pip install -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
jax[tpu]==0.4.13 \
tensorflow==2.11.0 \
flax==0.7.0 \
optax==0.1.7 \
distrax==0.1.3 \
chex==0.1.7 \
einops \
--extra-index-url https://download.pytorch.org/whl/cpu \
torch==2.0.1 \
transformers \
datasets \
huggingface_hub \
tqdm \
h5py \
ml_collections \
wandb \
gcsfs==2022.11.0 \
requests \
typing-extensions \
lm-eval==0.3.0 \
mlxu==0.1.11 \
sentencepiece \
pydantic \
uvicorn \
scipy \
scipy-stack'


gcloud compute tpus tpu-vm ssh nodel_id \
  --zone=us-central2-b --worker=all --command='pip install scipy==1.10'
```

We found datasets can error if HF id down, so we fixed the datasets and huggingface_cli.

```
gcloud compute tpus tpu-vm ssh nodel_id \
  --zone=us-central2-b --worker=all --command="pip uninstall -y datasets"
gcloud compute tpus tpu-vm ssh nodel_id \
  --zone=us-central2-b --worker=all --command="pip install https://github.com/wannaphong/datasets/archive/refs/heads/ok.zip"
gcloud compute tpus tpu-vm ssh nodel_id \
  --zone=us-central2-b --worker=all --command="pip uninstall -y huggingface_hub"
gcloud compute tpus tpu-vm ssh nodel_id \
  --zone=us-central2-b --worker=all --command="pip install https://github.com/wannaphong/huggingface_hub/archive/refs/heads/ok.zip"
```

Upload all file to TPU.

```
gcloud compute tpus tpu-vm scp --recurse train nodel_id: \
  --worker=all \
  --zone=us-central2-b

gcloud compute tpus tpu-vm ssh nodel_id \
--zone us-central2-b \
--worker=all \
--command='chmod +x ./train_3b.sh'
```

Running traini model

```
gcloud compute tpus tpu-vm ssh nodel_id \
  --zone=us-central2-b --worker=all --command='tmux new-session -d -s "launch" "~/train_3b.sh |& tee -a log1.txt"'
```

can view training by

```
gcloud compute tpus tpu-vm ssh nodel_id \
  --zone=us-central2-b --worker=all --command="tmux capture-pane -pt launch -S -2000"
```

If you want to kill the training model, you can use

```
gcloud compute tpus tpu-vm ssh nodel_id \
  --zone=us-central2-b --worker=all --command="tmux kill-session -t launch ; pkill -9 python3"
gcloud compute tpus tpu-vm ssh nodel_id \
  --zone=us-central2-b --worker=all --command="pkill -9 python3"
```

The mode will save in logger.output_dir.

You can read more at [EasyLM Docs](https://github.com/young-geng/EasyLM).