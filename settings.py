# Sync Period
UPDATE_GLOBAL_ITER = 4

# Discounting factor of rewards
GAMMA = 0.9

# Maximum episodes of RL algorithm
MAX_EP = 2000

# Maximum step size for each episodes (epochs for the model)
MAX_EP_STEP = 16

# The number of dropping worker
N_DROP = 0

# The number of total processes (workers)
N_WORKERS = 1

DIR_RESULT = "./result_test"

DIR_CHECKPOINT = "./checkpoint"

GPU_MAP = [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

def setup() -> list:
    from pathlib import Path
    import os
    Path(DIR_RESULT).mkdir(parents=True, exist_ok=True) # Result csv dir
    Path(DIR_CHECKPOINT).mkdir(parents=True, exist_ok=True) # Checkpoint file dir
    filelist = [ f for f in os.listdir(DIR_RESULT) ]
    if len(filelist) > 0:
        if "Y" == input(f"Would you like to delete log files? DIR='{DIR_RESULT}' [Y/N] "):
            for f in filelist:
                os.remove(os.path.join(DIR_RESULT, f))
    filelist = [ f for f in os.listdir(DIR_CHECKPOINT) ]
    if len(filelist) > 0:
        if "Y" == input(f"Do you want to load checkpoint? DIR='{DIR_CHECKPOINT}' [Y/N] "):
            # Load Checkpoint
            return filelist
        else:
            if "Y" == input(f"Are you sure that you backed up data? DIR='{DIR_CHECKPOINT}' [Y/N] "):
                for f in filelist:
                    os.remove(os.path.join(DIR_CHECKPOINT, f))
            else:
                raise
    return None
