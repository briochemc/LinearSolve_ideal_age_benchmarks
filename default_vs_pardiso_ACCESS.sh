#!/bin/bash

#PBS -P xv83
#PBS -N LinearSolveBenchmarks
#PBS -l ncpus=48
#PBS -l mem=180GB
#PBS -l jobfs=4GB
#PBS -l walltime=2:00:00
#PBS -l storage=gdata/gh0+scratch/xv83
#PBS -l wd
#PBS -o output/PBS/
#PBS -j oe

echo "Going into LinearSolve_ideal_age_benchmarks"
cd ~/Projects/LinearSolve_ideal_age_benchmarks

echo "Running script"
julia src/default_vs_pardiso_ACCESS.jl &> output/$PBS_JOBID.default_vs_pardiso_ACCESS.out
