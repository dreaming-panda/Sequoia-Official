# CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.4 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> X.log
# CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> X.log
# CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.8 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> X.log
# CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 1.0 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> X.log


# CUDA_VISIBLE_DEVICES=0 python test_specinfer.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.01 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> X.log
# CUDA_VISIBLE_DEVICES=0 python test_specinfer.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.4 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> X.log
# CUDA_VISIBLE_DEVICES=0 python test_specinfer.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> X.log
# CUDA_VISIBLE_DEVICES=0 python test_specinfer.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.8 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> X.log
# CUDA_VISIBLE_DEVICES=0 python test_specinfer.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 1.0 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> X.log

CUDA_VISIBLE_DEVICES=0 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.01 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> X.log
CUDA_VISIBLE_DEVICES=0 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.4 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> X.log
CUDA_VISIBLE_DEVICES=0 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> X.log
CUDA_VISIBLE_DEVICES=0 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.8 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> X.log
CUDA_VISIBLE_DEVICES=0 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 1.0 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> X.log

