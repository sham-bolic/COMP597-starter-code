
if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "'${BASH_SOURCE[0]}' is a config file, it should not be executed. Source it to populate the variables."
    exit 1
fi

# Values used to build some of the default configurations.
scripts_dir=$(readlink -f -n $(dirname ${BASH_SOURCE[0]})/../scripts)
default_config=$(readlink -f -n $(dirname ${BASH_SOURCE[0]}))/job_config.sh

# The configuration file that will be loaded if it exists. It should be used to 
# override any values provided here. By default, it will check if a file 
# "job_config.sh" exists in the same directory as this default config file. Use 
# it to override any of the variables described in this file except 
# COMP597_JOB_CONFIG.
#
# Of course, you might want to have multiple config files. To load a different
# one than the default, set the COMP597_JOB_CONFIG environment variable to the 
# path to your config file. When this variable is already set, it will not be 
# set to use the "job_config.sh" file.
#
# See "example_config.sh" for an example of how to make a configuration. The 
# file was made for the SLURM configurations, but it works the same.
export COMP597_JOB_CONFIG=${COMP597_JOB_CONFIG:-${default_config}}

################################################################################
########################### Possible configurations ############################
################################################################################

# Whether or not to print the configuration on launch.
export COMP597_JOB_CONFIG_LOG=true
# This a scratch storage local to each SLURM node. It is a space with no disk 
# quota, but it is fully deleted at the end of each semester or when it gets 
# full. It is the perfect place to store files which it is nice when reused, 
# but can easily be generated again. For example, it is a perfect space to 
# place the Hugging Face home directory, the pip cache directory or even your 
# own conda environment. 
#
# There should be no reason to override this path with your configuration.
export COMP597_JOB_SCRATCH_STORAGE="/mnt/teaching/slurm"
# Path to the storage partition for the course. It is mounted on 
# each SLURM node at this path. It is the correct place to store your training 
# dataset or any other large file that does not reside in your code repo. 
#
# Please refrain from storing any Conda environment in any of the 
# subdirectories of this path. Conda environments can be very large, and this 
# partition has a finite amount of space. 
#
# There should be no reason to override this path with your configuration.
export COMP597_JOB_STORAGE_PARTITION="/home/slurm/comp597"
# Base path to the Conda content managed by the administrators of 
# this course.
#
# There should be no reason to override this path with your configuration.
export COMP597_JOB_CONDA_DIR=${COMP597_JOB_STORAGE_PARTITION}/conda
# Path to the Conda environments managed by the administrators of 
# this course.
#
# There should be no reason to override this path with your configuration.
export COMP597_JOB_CONDA_ENV_DIR=${COMP597_JOB_CONDA_DIR}/envs
# Name of the Conda environment managed by the administrators of this course. 
# It is creaeted from the requirements.txt file in this repository.
export COMP597_JOB_CONDA_ENV_NAME="comp597"
# Path to the Conda environment managed by the administrators of this course. 
export COMP597_JOB_CONDA_ENV_PREFIX=${COMP597_JOB_CONDA_ENV_DIR}/${COMP597_JOB_CONDA_ENV_NAME}
# Path to the directory where students are allowed to create their own files 
# and directories on the storage partition. 
#
# There should be no reason to override this path with your configuration.
export COMP597_JOB_STUDENTS_BASE_DIR=${COMP597_JOB_STORAGE_PARTITION}/students
# Path to the student's own storage on the partition. By default, it assumes 
# the directory is named after the user running this script. 
export COMP597_JOB_STUDENT_STORAGE_DIR=${COMP597_JOB_STUDENTS_BASE_DIR}/${USER}
# Path to the student's own storage on the scratch storage provided on each 
# node. By default, it assumes the directory is named after the user running 
# this script. 
export COMP597_JOB_STUDENT_SCRATCH_STORAGE_DIR=${COMP597_JOB_SCRATCH_STORAGE}/${USER}
# Path used to cache files. By default, it uses the scratch storage on the 
# local node the job is executing on.
export COMP597_JOB_CACHE_DIR=${COMP597_JOB_STUDENT_SCRATCH_STORAGE_DIR}/.cache
# Path to allow pip to cache files outside the home directory. It avoids 
# exceeding the disk quota applied to the home directory.
export PIP_CACHE_DIR=${COMP597_JOB_CACHE_DIR}/pip
# Path to allow Hugging Face's libraries to cache files outside the home 
# directory. It avoid exceeding the disk quota applied to the home directory.
export HF_HOME=${COMP597_JOB_CACHE_DIR}/huggingface
# If non-empty and a valid path, changes the working directory before running 
# the job command.
export COMP597_JOB_WORKING_DIRECTORY=""
# Command to run (preserve env override, e.g. COMP597_JOB_COMMAND="bash -lc" ./scripts/srun.sh '…').
export COMP597_JOB_COMMAND="${COMP597_JOB_COMMAND:-${scripts_dir}/launch.sh}"

################################################################################
################################################################################
################################################################################

# Unset the variables used locally.
unset default_config
