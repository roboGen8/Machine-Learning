import sys
import os
import time

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
from array import array



"""
Commandline parameter(s):
   none
"""
sys.stdout = open("p4_ga.txt", "w")

print 'N, fitness, func_calls, time'

perf = {}

N_iter = [10, 20, 30, 40, 50, 60]

for N in N_iter:

	
	nexp = 20
	

	# Random number generator */
	random = Random()
	# The number of items
	NUM_ITEMS = N
	# The number of copies each
	COPIES_EACH = 4
	# The maximum weight for a single element
	MAX_WEIGHT = 50
	# The maximum volume for a single element
	MAX_VOLUME = 50
	# The volume of the knapsack 
	KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

	# create copies
	fill = [COPIES_EACH] * NUM_ITEMS
	copies = array('i', fill)

	# create weights and volumes
	fill = [0] * NUM_ITEMS
	weights = array('d', fill)
	volumes = array('d', fill)
	for i in range(0, NUM_ITEMS):
	    weights[i] = random.nextDouble() * MAX_WEIGHT
	    volumes[i] = random.nextDouble() * MAX_VOLUME


	# create range
	fill = [COPIES_EACH + 1] * NUM_ITEMS
	ranges = array('i', fill)

	ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
	odd = DiscreteUniformDistribution(ranges)
	nf = DiscreteChangeOneNeighbor(ranges)
	mf = DiscreteChangeOneMutation(ranges)
	cf = UniformCrossOver()
	df = DiscreteDependencyTree(.1, ranges)
	hcp = GenericHillClimbingProblem(ef, odd, nf)
	gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
	pop = GenericProbabilisticOptimizationProblem(ef, odd, df)


	
	perf[N] = {}
	perf[N]['fitness'] = []
	perf[N]['calls'] = []
	perf[N]['time'] = []	
	
	exp_fitn = []
	exp_calls = []
	exp_time = []

	i2 = 0;
	while(i2<nexp):
		tstart = time.time()
		
		ga = StandardGeneticAlgorithm(200, 100, 10, gap)	
		fit = FixedIterationTrainer(ga, 1500)
		fit.train()
		exp_fitn.append(ef.value(ga.getOptimal()))
		exp_calls.append(ef.getTotalCalls())

		tot_time = time.time() - tstart;
		exp_time.append(tot_time)

		i2 = i2 + 1
	
	perf[N]['fitness'] = (1.0*sum(exp_fitn)/len(exp_fitn))
	perf[N]['calls'] = (1.0*sum(exp_calls)/len(exp_calls))
	perf[N]['time'] = (1.0*sum(exp_time)/len(exp_time))

	print str(N)+', '+str(perf[N]['fitness']) + ', ' + str(perf[N]['calls']) + ', ' + str(perf[N]['time'])


