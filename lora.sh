## 帳號套件下載
iam=$(whoami)
container_dir=/work/${iam}/container_demo
home_dir=${container_dir}/home
tmp_dir=${container_dir}/tmp
output_dir=${container_dir}/output
mkdir -p ${home_dir} ${tmp_dir} ${output_dir}
rsync -avHS ${PWD}/ ${container_dir}/app

# 映像檔
IMAGE="/work/u00cjz00/nvidia/alpaca-lora_latest.sif"

# 模型
MODEL_ID="/work/u00cjz00/slurm_jobs/github/models/Llama-2-7b-chat-hf"

# 訓練資料及輸出結果目錄
input_data="/work/u00cjz00/slurm_jobs/github/dataset/school_math_30000.json";
output_data="${output_dir}/lora_school_math_30000";


# 安裝套件
ml libs/singularity/3.10.2
singularity exec --nv -B /work -B ${container_dir}/home:/home/$(whoami) -B ${container_dir}/tmp:/tmp -B ${container_dir}/app:/app ${IMAGE} bash -c "cd /app;  pip install -r requirements.txt -q; pip install protobuf install accelerate==0.24.1 bitsandbytes transformers accelerate bitsandbytes bitsandbytes==0.41 scipy -q"

# 執行訓練
singularity exec --nv -B /work -B ${container_dir}/home:/home/$(whoami) -B ${container_dir}/tmp:/tmp -B ${container_dir}/app:/app ${IMAGE} bash -c "cd /app;  python3 finetune.py --base_model \'${MODEL_ID}\' --data_path \'${input_data}\' \'${output_data}\'"
