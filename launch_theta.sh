. /etc/profile

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job started at: {$TSTAMP}"

# Figure out training environment
if [[ -z "${COBALT_NODEFILE}" ]]; then
    RANKS=$HOSTNAME
    NNODES=1
else
    MASTER_RANK=$(head -n 1 $COBALT_NODEFILE)
    RANKS=$(tr '\n' ' ' < $COBALT_NODEFILE)
    NNODES=$(< $COBALT_NODEFILE wc -l)
fi

#NNODES=1
#RANKS=$HOSTNAME
echo $NNODES
# Commands to run prior to the Python script for setting up the environment
PRELOAD="source /etc/profile ; "
# PRELOAD+="module load conda/pytorch ; "
# PRELOAD+="conda activate /lus/theta-fs0/projects/SuperBERT/jgpaul/envs/pytorch-1.9.1-cu11.3 ; "
PRELOAD+="ml conda;"
PRELOAD+="conda activate climate;"
PRELOAD+="export OMP_NUM_THREADS=4 ; "
PRELOAD+="export NODES=1; "

# time python process to ensure timely job exit
TIMER="timeout 718m "

# torchrun launch configuration
LAUNCHER="python3 -m torch.distributed.run "
LAUNCHER+="--nnodes=$NNODES --nproc_per_node=auto --max_restarts 0 "
if [[ "$NNODES" -eq 1 ]]; then
    LAUNCHER+="--standalone "
else
    LAUNCHER+="--rdzv_backend=c10d --rdzv_endpoint=$MASTER_RANK "
fi

# Training script and parameters

# CMD="scripts/run_cmip6_continuous.py --config scripts/configs/config_cmip6_mask2former_stage1.yaml"
# CMD="scripts/run_cmip6_continuous.py --config scripts/configs/config_cmip6_swinv2_stage1.yaml"
# CMD="scripts/run_cmip6_continuous.py --config scripts/configs/config_cmip6_vit_pretrained_stage1.yaml"
# CMD="scripts/run_cmip6_continuous.py --config scripts/configs/config_cmip6_mask2former_stage2.yaml --nodes 2 --gpus 4"
# CMD="scripts/run_cmip6_continuous.py --config scripts/configs/config_cmip6_swinv2_stage2.yaml --nodes 2 --gpus 4"
CMD="scripts/run_cmip6_continuous.py --config scripts/configs/config_cmip6_vit_pretrained_stage2.yaml --nodes 4 --gpus 8"

FULL_CMD=" $PRELOAD $TIMER $LAUNCHER $CMD $@ "
echo "Training Command: $FULL_CMD"


# Launch the pytorch processes on each worker (use ssh for remote nodes)
RANK=0
for NODE in $RANKS; do #${RANKS[*]:0:21}; do #$RANKS; do
    if [[ "$NODE" == "$HOSTNAME" ]]; then
        echo "Launching rank $RANK on local node $NODE"
        eval $FULL_CMD &
    else
        echo "Launching rank $RANK on remote node $NODE"
        ssh $NODE "cd $PWD; $FULL_CMD" &
    fi
    RANK=$((RANK+1))
done

wait