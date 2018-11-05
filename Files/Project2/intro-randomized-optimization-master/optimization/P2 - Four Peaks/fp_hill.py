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
import opt.example.CountOnesEvaluationFunction as CountOnesEvaluationFunction
from array import array



"""
Commandline parameter(s):
   none
"""
sys.stdout = open("p2_hill.txt", "w")

print 'N, fitness, func_calls, time'

perf = {}

N_iter = [10, 20, 30, 40, 50, 60]

for N in N_iter:

	
	nexp = 20
	

	T=N/5
	fill = [2] * N
	ranges = array('i', fill)



	ef = FourPeaksEvaluationFunction(T)
	odd = DiscreteUniformDistribution(ranges)
	nf = DiscreteChangeOneNeighbor(ranges)
	mf = DiscreteChangeOneMutation(ranges)
	cf = SingleCrossOver()
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
		
		rhc = RandomizedHillClimbing(hcp)	
		fit = FixedIterationTrainer(rhc, 300000)
		fit.train()
		exp_fitn.append(ef.value(rhc.getOptimal()))
		exp_calls.append(ef.getTotalCalls())

		tot_time = time.time() - tstart;
		exp_time.append(tot_time)

		i2 = i2 + 1
	
	perf[N]['fitness'] = (1.0*sum(exp_fitn)/len(exp_fitn))
	perf[N]['calls'] = (1.0*sum(exp_calls)/len(exp_calls))
	perf[N]['time'] = (1.0*sum(exp_time)/len(exp_time))

	print str(N)+', '+str(perf[N]['fitness']) + ', ' + str(perf[N]['calls']) + ', ' + str(perf[N]['time'])


