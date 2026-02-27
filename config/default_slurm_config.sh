
if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "'${BASH_SOURCE[0]}' is a config file, it should not be executed. Source it to populate the variables."
    exit 1
fi

# Values used to build some of the default configurations.
scripts_dir=$(readlink -f -n $(dirname ${BASH_SOURCE[0]})/../scripts)
default_config=$(readlink -f -n $(dirname ${BASH_SOURCE[0]}))/slurm_config.sh

# The SLURM configuration file that will be loaded if it exists. It should be 
# used to override any values provided here. By default, it will check if a 
# file "slurm_config.sh" exists in the same directory as this default config 
# file. Use it to override any of the variables described in this file except 
# COMP597_SLURM_CONFIG. 
#
# Of course, you might want to have multiple config files. To load a different 
# one than the default, set the COMP597_SLURM_CONFIG environment variable to 
# the path to your config file. When this variable is already set, it will not 
# be set to use the "slurm_config.sh" file.
#
# See "example_config.sh" for an example of how to make 
# a configuration.
export COMP597_SLURM_CONFIG=${COMP597_SLURM_CONFIG:-${default_config}}

################################################################################
########################### Possible configurations ############################
################################################################################

# NOTE: The variables are sorted alphabetically. 

# The SLURM account to use.
# See --account in srun --help
export COMP597_SLURM_ACCOUNT="energy-efficiency-comp597"
# Whether or not to print the configuration on launch.
export COMP597_SLURM_CONFIG_LOG=true
# The number of CPUs required per tasks.
# See --cpus-per-task in srun --help
export COMP597_SLURM_CPUS_PER_TASK=4
# The jobs script that will be provided to srun. 
# It defaults to the script "scripts/job.sh" in this repository.
export COMP597_SLURM_JOB_SCRIPT=${scripts_dir}/job.sh
# The minimum amount of memory to request. 
# See --mem in srun --help
export COMP597_SLURM_MIN_MEM="4GB"
# The SLURM nodes requested on which the job will execute.
# See --nodelist in srun --help
export COMP597_SLURM_NODELIST="gpu-grad-01"
# The number of tasks to run.
# See --ntasks in srun --help
export COMP597_SLURM_NTASKS=1
# The number of GPUs required for the job.
# See --gpus in srun --help
export COMP597_SLURM_NUM_GPUS=1
# The SLURM partition to use.
# See --partition in srun --help
export COMP597_SLURM_PARTITION="all"
# The SLURM QOS to use.
# See --qos in srun --help
export COMP597_SLURM_QOS="comp597"
# The maximum amount of time the job is allowed to run for. 
# See --time in srun --help
export COMP597_SLURM_TIME_LIMIT="5:00"

################################################################################
################################################################################
################################################################################

# Unset the variables used locally.
unset default_config
unset scripts_dir
