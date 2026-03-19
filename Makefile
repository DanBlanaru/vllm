OUTPUT_PATH := /scratch/results
WEIGHTS_PATH := /scratch/mistral-medium-2507

TP_SIZE := 4

SHARED_ARGS := --no-enable-prefix-caching --served-model-name default_model
MISTRAL_ARGS := --config-format mistral --load-format mistral --tokenizer-mode mistral

export CXXFLAGS="-I/usr/local/lib/python3.12/dist-packages/nvidia/nvtx/include"
export CFLAGS="-I/usr/local/lib/python3.12/dist-packages/nvidia/nvtx/include"
export CUDAFLAGS="-I/usr/local/lib/python3.12/dist-packages/nvidia/nvtx/include"
HF_HOME=../hf_cache
export HF_TOKEN := $(shell cat ../hf_token)


install-nsys:
	@. /etc/lsb-release && \
	UBUNTU_RELEASE=$$(echo "$$DISTRIB_RELEASE" | tr -d .) && \
	UBUNTU_RELEASE_TO_GET_OLD_KEY=1804 && \
	KERNEL_ARCH=$$(uname -m) && \
	DPKG_ARCH=$$(dpkg --print-architecture) && \
	OLD_KEY=7fa2af80 && \
	OLD_KEY_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$${UBUNTU_RELEASE_TO_GET_OLD_KEY}/$${KERNEL_ARCH}/$${OLD_KEY}.pub" && \
	echo "Updating and installing prerequisites..." && \
	sudo apt update && \
	sudo apt install -y --no-install-recommends gnupg ca-certificates wget && \
	echo "Adding NVIDIA DevTools repository..." && \
	echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$${UBUNTU_RELEASE}/$${DPKG_ARCH} /" | sudo tee /etc/apt/sources.list.d/nvidia-devtools.list && \
	echo "Adding NVIDIA public key..." && \
	sudo wget -O - "$$OLD_KEY_URL" | sudo apt-key add - && \
	echo "Installing nsight-systems-cli..." && \
	sudo apt update && \
	sudo apt install -y nsight-systems-cli



cuda-deps:
	sudo apt-get update && sudo apt-get install -y \
		cuda-nvrtc-dev-13-0 \
		libcublas-dev-13-0 \
		libcusparse-dev-13-0 \
		libcusolver-dev-13-0 \
		ccache

install:
	python3 -m pip uninstall -y vllm
	python3 -m venv .venv
	. .venv/bin/activate && pip install uv
	. .venv/bin/activate && uv pip install -r requirements/build.txt
	. .venv/bin/activate && uv pip install -r requirements/cuda.txt \
		--prerelease=allow \
		--index-strategy unsafe-best-match \
		--extra-index-url https://download.pytorch.org/whl/cu129 \
		--force-reinstall
	. .venv/bin/activate && CCACHE_NOHASHDIR="true" uv pip install --no-build-isolation -e . -v \
		--prerelease=allow \
		--index-strategy unsafe-best-match \
		--extra-index-url https://download.pytorch.org/whl/cu129
	. .venv/bin/activate && uv pip install nvidia-lm-eval math_verify nvtx
	# TORCH_CUDA_ARCH_LIST="9.0" bash vllm/tools/ep_kernels/install_python_libraries.sh

# Fast install target that skips flash attention (for dev iteration)
install-fast:
	. .venv/bin/activate && CCACHE_NOHASHDIR="true" CMAKE_ARGS="-DVLLM_SKIP_FLASH_ATTN=ON" \
		uv pip install --no-build-isolation -e . -v \
		--prerelease=allow \
		--index-strategy unsafe-best-match \
		--extra-index-url https://download.pytorch.org/whl/cu129


bench_quick_reduce_cuda:
	mkdir -p $(RUN_NAME)
	. .venv/bin/activate && torchrun --nproc_per_node=2 tests/distributed/benchmark_quick_reduce_cuda.py > $(RUN_NAME)/quick_reduce_cuda_tp2.txt 2>&1
	. .venv/bin/activate && torchrun --nproc_per_node=4 tests/distributed/benchmark_quick_reduce_cuda.py > $(RUN_NAME)/quick_reduce_cuda_tp4.txt 2>&1
	. .venv/bin/activate && torchrun --nproc_per_node=8 tests/distributed/benchmark_quick_reduce_cuda.py > $(RUN_NAME)/quick_reduce_cuda_tp8.txt 2>&1

serve:
	vllm serve $(WEIGHTS_PATH) \
	--tensor-parallel-size $(TP_SIZE) \
	$(MISTRAL_ARGS) \
	$(SHARED_ARGS)

serve-fp8:
	vllm serve $(WEIGHTS_PATH) \
	--tensor-parallel-size $(TP_SIZE) \
	$(MISTRAL_ARGS) \
	$(SHARED_ARGS) \
	--kv-cache-dtype fp8

serve-tke:
	vllm serve $(WEIGHTS_PATH) \
	--attention-config.backend TKE \
	--tensor-parallel-size $(TP_SIZE) \
	$(MISTRAL_ARGS) \
	$(SHARED_ARGS)

serve-tke-fp8:
	vllm serve $(WEIGHTS_PATH) \
	--attention-config.backend TKE \
	--tensor-parallel-size $(TP_SIZE) \
	$(MISTRAL_ARGS) \
	$(SHARED_ARGS) \
	--kv-cache-dtype fp8

EVAL_OUTPUT_DIR := $(CURDIR)/nemo_eval_results

eval-gsm8k:
	rm -rf $(EVAL_OUTPUT_DIR)
	mkdir -p $(EVAL_OUTPUT_DIR)
	HF_TOKEN=$(HF_TOKEN) HF_HOME=../hf_cache nemo-evaluator run_eval \
		--eval_type gsm8k \
		--model_id Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 \
		--model_url http://localhost:8080/v1/completions \
		--model_type completions \
		--output_dir ./nemo_eval_results \
		--override config.params.parallelism=1

lmeval:
	rm -rf $(EVAL_OUTPUT_DIR)
	mkdir -p $(EVAL_OUTPUT_DIR)
	cp $(CURDIR)/gsm8k_custom_cot.yaml .venv/lib/python3.12/site-packages/lm_eval/tasks/
	HF_TOKEN=$(HF_TOKEN) HF_HOME=../hf_cache lm_eval \
		--model local-completions \
		--model_args model=Qwen/Qwen3-235B-A22B-Thinking-2507-FP8,base_url=http://localhost:8080/v1/completions,tokenized_requests=False,trust_remote_code=True,num_concurrent=64 \
		--tasks gsm8k_custom_cot \
		--batch_size auto \
		--log_samples \
		--limit 5 \
		--output_path $(EVAL_OUTPUT_DIR)

query-completion:
	curl http://localhost:8080/v1/completions \
	    -H "Content-Type: application/json" \
	    -d '{"model": "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8", "prompt": "What is the capital of France?", "max_tokens": 128, "temperature": 0}'

query-chat:
	curl -X POST http://localhost:8080/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d '{"messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 256, "model": "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"}'