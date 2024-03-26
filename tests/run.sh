CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 1.5 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> X.log
CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 2.0 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> X.log
CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 5.0 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> X.log

CUDA_VISIBLE_DEVICES=0 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 1.5 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> A.log
CUDA_VISIBLE_DEVICES=0 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 2.0 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> A.log
CUDA_VISIBLE_DEVICES=0 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 5.0 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/5x8-tree.pt  --Mode greedy --dataset cnn >> A.log

CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 1.5 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt  --Mode greedy --dataset cnn >> Y.log
CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 2.0 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt  --Mode greedy --dataset cnn >> Y.log
CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 5.0 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt  --Mode greedy --dataset cnn >> Y.log

CUDA_VISIBLE_DEVICES=0 python test_specinfer.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 1.5 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt  --Mode greedy --dataset cnn >> Z.log
CUDA_VISIBLE_DEVICES=0 python test_specinfer.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 2.0 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt  --Mode greedy --dataset cnn >> Z.log
CUDA_VISIBLE_DEVICES=0 python test_specinfer.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 5.0 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt --Mode greedy --dataset cnn >> Z.log

CUDA_VISIBLE_DEVICES=0 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 1.5 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt  --Mode greedy --dataset cnn >> W.log
CUDA_VISIBLE_DEVICES=0 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 2.0 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt  --Mode greedy --dataset cnn >> W.log
CUDA_VISIBLE_DEVICES=0 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 5.0 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt  --Mode greedy --dataset cnn >> W.log


