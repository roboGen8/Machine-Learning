# randomized-optimization

The purpose of this project is to explore random search. As always, it is important to realize that understanding an algorithm or technique requires more than reading about that algorithm or even implementing it. One should actually have experience seeing how it behaves under a variety of circumstances.

The following four local random search algorithms have been implemented. They are:

- randomized hill climbing
- simulated annealing
- genetic algorithm
- MIMIC

There are two parts to the project. The first part uses these algorithms to learn weights of a neural network as compared to the conventional backprop algorithm. In the second part, all four search techniques have been applied to four optimization problems. The first problem highlight advantages of the genetic algorithm, the second of simulated annealing, and the third of MIMIC.

### Execution Instructions:

Part 1 - Neural Networks - 

- This part has been implemented in MATLAB
- There are 5 .m files, one for the evaluation function and the others - one for each algorithm
- There is also a .mat file which contains the data and the indices for the test/train/val split
- The algorithm files can be run as it is to generate results

Part 2 - Optimization - 

- This part has been implemented using Jython and the ABAGAIL library.
- The .JAR file is included in the submission attachment. Please make sure to use this.
- Every problem-algorithm pair has a code. The ideal way to run this would be to run the run*.py files present in the folders corresponding to each of the problems. This will run all the algorithm files corresponding to that problem.
- Please make sure that you change the paths pointing to the JAR file, JAVA and Jython before executing the file.
