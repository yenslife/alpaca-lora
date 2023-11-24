#!/work/u00cjz00/binary/bash5.0/bin/bash
#SBATCH -A MST110386                                                    ### project number, Example MST109178
#SBATCH -J _t2lora_							                                        ### Job name, Exmaple jupyterlab
#SBATCH -p gp4d                                                         ### Partition Name, Example ngs1gpu
#SBATCH --nodes=1                                                       ### Nodes, Default 1, node number
#SBATCH --ntasks-per-node=1                                             ### Tasks, Default 1, per node tasks
#SBATCH -c 4                                                            ### Cores assigned to each task, Example 4
#SBATCH --gres=gpu:1                                                    ### GPU number, Example gpu:1
#SBATCH --time=0-1:00:00                                                ### Runnung time, days-hours:minutes:seconds or hours:minutes:seconds
#SBATCH -o genai_%j.out     											### Log folder, Here %j is job ID
#SBATCH -e genai_%j.err     											### Log folder, Here %j is job ID


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
singularity exec --nv -B /work -B ${container_dir}/home:/home/$(whoami) -B ${container_dir}/tmp:/tmp -B ${container_dir}/app:/app ${IMAGE} bash -c "cd /app;  python3 finetune.py --base_model \'${MODEL_ID}\' --data_path \'${input_data}\' \'${output_data}\'" &
pid0=$!

# NODE
noed_hostname=$(hostname -s)
noed_port=$(python3 /work/u00cjz00/binary/availablePort.py)
node_ip=$(cat /etc/hosts |grep "$(hostname -a)" | awk '{print $1}')

# message
echo ""
echo ""
echo "****************************** 請輸入下方指令 ******************************"
echo ""
echo "# STEP1: Execute cmd in your client below to login login_node"
echo ssh ${iam}@ln01.twcc.ai
echo ""
echo "# STEP2: Login  computing_node "
echo ssh ${noed_hostname}
echo ""
echo ""
echo "***********************************************************************************************"

## Get pid
pid0=$!
sleep 2
pid1=$(pgrep -P ${pid0})
pid=${pid1}
wait $pid0
