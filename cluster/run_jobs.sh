#!/bin/bash
for i in {1..8}; do
    sbatch job.slurm $1 $i
done
