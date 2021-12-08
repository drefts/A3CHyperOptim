# Sync Period
UPDATE_GLOBAL_ITER = 4

# Discounting factor of rewards
GAMMA = 0.9

# Maximum episodes of RL algorithm
MAX_EP = 3000

# Maximum step size for each episodes (epochs for the model)
MAX_EP_STEP = 12

# The number of dropping worker
N_DROP = 1

# The number of total processes (workers)
N_WORKERS = 6

DIR_RESULT = "./result"

DIR_CHECKPOINT = "./checkpoint"

def setup() -> list:
    from pathlib import Path
    import os
    Path(DIR_RESULT).mkdir(parents=True, exist_ok=True) # Result csv dir
    Path(DIR_CHECKPOINT).mkdir(parents=True, exist_ok=True) # Checkpoint file dir
    filelist = [ f for f in os.listdir(DIR_RESULT) ]
    if len(filelist) > 0:
        if "Y" == input(f"Are you sure that you backed up data? DIR='{DIR_RESULT}' [Y/N] "):
            for f in filelist:
                os.remove(os.path.join(DIR_RESULT, f))
        else:
            raise
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
