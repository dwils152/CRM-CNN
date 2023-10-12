#!/bin/bash
#SBATCH --job-name=my_sql
#SBATCH --output=my_sql.out
#SBATCH --error=my_sql.err
#SBATCH --time=500:00:00
#SBATCH --partition=Orion

module load singularity

# Local directories to bind to the container
mkdir -p ${PWD}/mysql/var/lib/mysql ${PWD}/mysql/run/mysqld

# Start the container 
singularity instance.start --bind ${HOME} \
    --bind ${PWD}/mysql/var/lib/mysql/:/var/lib/mysql \
    --bind ${PWD}/mysql/run/mysqld:/run/mysqld \
    ./mysql.simg mysql

# Start the mysql server
singularity run instance://mysql