


# Run fast chat

This require 14GB of GPU memory for our model.
```bash
python3 -m fastchat.serve.cli --model-path /models/finetune_model --num-gpus num_of_gpus
```

Run on the CPU only, and it requires 30GB of CPU memory.

```bash
python3 -m fastchat.serve.cli --model-path /models/finetune_model --device cpu
```

No Enough Memory
If you do not have enough memory, you can enable 8-bit compression by adding `--load-8bit` to commands above. 
This can reduce memory usage by around half with slightly degraded model quality. 

```bash
python3 -m fastchat.serve.cli --model-path /models/finetune_model --load-8bit
```

In addition to that, you can add `--cpu-offloading` to commands above to offload weights that don't fit on your GPU onto the CPU memory. This requires 8-bit compression to be enabled and the bitsandbytes package to be installed, which is only available on linux operating systems.