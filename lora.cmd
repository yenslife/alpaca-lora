CURRENT_DIR=${PWD}
mkdir ~/logs/
cd ~/logs/
sbatch -A MST110386 --time=0-1:00:00 ${CURRENT_DIR}/lora.slurm
