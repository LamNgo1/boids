#!/bin/sh

conda init bash
conda activate boids

for objective in ackley branin500 hartmann500 mopta lasso-dna half-cheetah; do
   for seed in {1..10}; do
      python test-boids.py -f $objective -d 100 -n 1000 --seed $seed --output results_from_paper
   done
done