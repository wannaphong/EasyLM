#! /bin/bash
# Put your WANDB API key here to enable logging to wandb.
export WANDB_API_KEY='wandb key'

export HF_TOKEN='hf_xxx'

# TPU specific flags to improve training throughput
# TPU specific flags to improve training throughput
export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'


python3 -m EasyLM.models.llama.llama_train \
    --jax_distributed.initialize_jax_distributed=True \
    --mesh_dim='1,-1,1' \
    --dtype='bf16' \
    --total_steps=505300 \
    --log_freq=25 \
    --save_model_freq=126325 \
    --save_milestone_freq=252650 \
    --load_llama_config='3b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='' \
    --tokenizer.pretrained_model_name_or_path='scb10x/typhoon-7b' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=1.0 \
    --optimizer.adamw_optimizer.lr=1e-4 \
    --optimizer.adamw_optimizer.end_lr=1e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=2000 \
    --optimizer.adamw_optimizer.lr_decay_steps=505300 \
    --optimizer.adamw_optimizer.bf16_momentum=True \
    --train_dataset.type='huggingface' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.text_processor.add_eos_token=True \
    --train_dataset.text_processor.add_bos_token=True \
    --train_dataset.huggingface_dataset.path='wannaphong/mark13' \
    --train_dataset.huggingface_dataset.split='train' \
    --train_dataset.huggingface_dataset.seq_length=2048 \
    --train_dataset.huggingface_dataset.streaming=True \
    --train_dataset.huggingface_dataset.batch_size=64 \
    --checkpointer.save_optimizer_state=True \
    --logger.online=True \
    --logger.prefix='EasyLM' \
    --logger.project="mark13-3b" \
    --logger.output_dir="./mark13" \
    --logger.wandb_dir="./"

