. /etc/profile

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job started at: {$TSTAMP}"

# Figure out training environment
if [[ -z "${PBS_NODEFILE}" ]]; then
    RANKS=$HOSTNAME
    NNODES=1
else
    MASTER_RANK=$(head -n 1 $PBS_NODEFILE)
    RANKS=$(tr '\n' ' ' < $PBS_NODEFILE)
    NNODES=$(< $PBS_NODEFILE wc -l)
fi


echo $NNODES
# Commands to run prior to the Python script for setting up the environment
PRELOAD="source /etc/profile ; "
PRELOAD+="module load conda/2023-01-10-unstable;"
PRELOAD+="conda activate climate;"
PRELOAD+="export OMP_NUM_THREADS=4 ; "
PRELOAD+="export NODES=$NNODES; "


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

# CMD="scripts/run_cmip6_multisource_continuous.py --config scripts/configs/config_cmip6_multisource_mask2former_stage1.yaml --nodes 10 --gpus 4"
CMD="scripts/run_cmip6_multisource_continuous.py --config scripts/configs/config_cmip6_multisource_mask2former_stage2.yaml --nodes 10 --gpus 4"

FULL_CMD=" $PRELOAD $TIMER $LAUNCHER $CMD $@ "
echo "Training Command: $FULL_CMD"


# Launch the pytorch processes on each worker (use ssh for remote nodes)
RANK=0
for NODE in $RANKS; do #${RANKS[*]:0:21}; do #$RANKS; do
    if [[ "$NODE" == "$HOSTNAME" ]]; then
        echo "Launching rank $RANK on local node $NODE"
        FULL_CMD_THIS_NODE=" export NODE_RANK=$RANK; $FULL_CMD "
        eval $FULL_CMD_THIS_NODE &
    else
        echo "Launching rank $RANK on remote node $NODE"
        FULL_CMD_THIS_NODE=" export NODE_RANK=$RANK; $FULL_CMD "
        ssh $NODE "cd $PWD; $FULL_CMD_THIS_NODE" &
    fi
    RANK=$((RANK+1))
done

wait