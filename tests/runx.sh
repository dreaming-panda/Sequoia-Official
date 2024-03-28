#CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset c4 >> x1.log
#CUDA_VISIBLE_DEVICES=2 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset c4 >> x1.log

CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset openwebtext >> x1.log
CUDA_VISIBLE_DEVICES=2 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset openwebtext >> x1.log

CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset cnn >> x1.log
CUDA_VISIBLE_DEVICES=2 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset cnn >> x1.log

#CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset c4 >> x1.log
#CUDA_VISIBLE_DEVICES=2 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset c4 >> x1.log

CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset openwebtext >> x1.log
CUDA_VISIBLE_DEVICES=2 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset openwebtext >> x1.log

CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset cnn >> x1.log
CUDA_VISIBLE_DEVICES=2 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset cnn >> x1.log

#CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset c4 >> x1.log
#CUDA_VISIBLE_DEVICES=2 python testbed.py --model  JackFram/llama-160m    --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset c4 >> x1.log

CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  JackFram/llama-160m    --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset openwebtext >> x1.log
CUDA_VISIBLE_DEVICES=2 python testbed.py --model  JackFram/llama-160m    --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset openwebtext >> x1.log

CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  JackFram/llama-160m    --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset cnn >> x1.log
CUDA_VISIBLE_DEVICES=2 python testbed.py --model  JackFram/llama-160m    --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset cnn >> x1.log

#CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset c4 >> x2.log
#CUDA_VISIBLE_DEVICES=2 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset c4 >> x2.log

CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B    --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset openwebtext >> x1.log
CUDA_VISIBLE_DEVICES=2 python testbed.py --model  princeton-nlp/Sheared-LLaMA-1.3B    --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset openwebtext >> x1.log

CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B    --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset cnn >> x1.log
CUDA_VISIBLE_DEVICES=2 python testbed.py --model  princeton-nlp/Sheared-LLaMA-1.3B    --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset cnn >> x1.log

CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset openwebtext >> x2.log
CUDA_VISIBLE_DEVICES=2 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset openwebtext >> x2.log

CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset cnn >> x2.log
CUDA_VISIBLE_DEVICES=2 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset cnn >> x2.log

#CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset c4 >> x2.log
#CUDA_VISIBLE_DEVICES=2 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset c4 >> x2.log

CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset openwebtext >> x2.log
CUDA_VISIBLE_DEVICES=2 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset openwebtext >> x2.log

CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset cnn >> x2.log
CUDA_VISIBLE_DEVICES=2 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset cnn >> x2.log

#CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset c4 >> x2.log
#CUDA_VISIBLE_DEVICES=2 python testbed.py --model  JackFram/llama-160m    --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset c4 >> x2.log

CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  JackFram/llama-160m    --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset openwebtext >> x2.log
CUDA_VISIBLE_DEVICES=2 python testbed.py --model  JackFram/llama-160m    --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset openwebtext >> x2.log

CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  JackFram/llama-160m    --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset cnn >> x2.log
CUDA_VISIBLE_DEVICES=2 python testbed.py --model  JackFram/llama-160m    --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset cnn >> x2.log

CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B    --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset openwebtext >> x2.log
CUDA_VISIBLE_DEVICES=2 python testbed.py --model  princeton-nlp/Sheared-LLaMA-1.3B    --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset openwebtext >> x2.log

CUDA_VISIBLE_DEVICES=2 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B    --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset cnn >> x2.log
CUDA_VISIBLE_DEVICES=2 python testbed.py --model  princeton-nlp/Sheared-LLaMA-1.3B    --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/8-chain.pt  --Mode greedy --dataset cnn >> x2.log
