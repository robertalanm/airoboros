python -m airoboros.lmoe.vllm \
>   --model ../llama-2-7b-hf \
>   --lmoe ../lilith-lmoe-7b-v120rev2  \
>   --router-max-samples 100 \
>   --router-k 25 \
>   --port 8000 \
>   --host 0.0.0.0 \
>   --tensor-parallel-size 4