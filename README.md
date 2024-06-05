# Parallel-in-time numerical methods based on Extreme Learning Machines

In this repository we collect the codebase for the paper "Parallel-in-time numerical methods based on Extreme Learning Machines".

Procedure to start using the code:
Install the required libraries
> pip install -r requirements.txt

To run the experiments with ELM as a coarse propagator, either run
> python3 ELM/main_one_run.py

to run a single experiment, or
> python3 ELM/main_repeated_experiments.py

to run several tests and get the average results.

To train the flow map network run
> python3 flowMapNetwork/mainTraining.py

and run
> python3 flowMapNetwork/mainParareal.py

to use the trained flow map-based coarse propagator in the Parareal algorithm.

Here is a brief description of the components of this repository.

- [ELM/scripts](https://github.com/davidemurari/learnTimeSteppingSchemes/tree/main/ELM/scripts) folder: 
  - [dynamics.py](https://github.com/davidemurari/learnTimeSteppingSchemes/blob/main/ELM/scripts/dynamics.py) script: This script contains the definition of the class *vecField* where all the evaluation of the six vector fields is described. The *eval* method of the class allows to evaluate the vector field both on a batch of points and on a single point.
  - [ode_solvers.py](https://github.com/davidemurari/learnTimeSteppingSchemes/blob/main/ELM/scripts/ode_solvers.py) script: This script allows to integrate in time one of the six ODEs. The method *solver* is written so it can be run in parallel, and accepts two arguments: *args* and *final*. The argument *args* is a list of two elements:
    1. the first is again a list containing the initial condition, the final time, and possibly the array of time instants over which the ODE needs to be solved
    2. the second is an object of the class vecField determining the ODE we want to solve.
  The optional argument *final* specifies if one is interested to return also the evaluation time array, in case *final=False*, or not. In this script there is also the definition of the method *RK4_step* which performs one step with the RK4 method, provided the initial condition *x0*, the time step *dt*, and the vecField object *vecRef*.
  - [plotting.py](https://github.com/davidemurari/learnTimeSteppingSchemes/blob/main/ELM/scripts/plotting.py) script: This script contains the plotting routine for the six systems included in the paper. Each of the specific routines is customized to improve the readability of the corresponding plot, and get those included in the paper.
  - [repeated_experiments.py](https://github.com/davidemurari/learnTimeSteppingSchemes/blob/main/ELM/scripts/repeated_experiments.py) script: This script contains the method *run_experiment* that allows to complete the iterates of the hybrid Parareal method once. This method is of use when repeating the experiment several times in Parallel in order to compute the average results. This method accepts three arguments:
    1. *args*: a list of two terms, which are *system*, the name of the ODE to consider, and *nodes*, the type of collocation points to adopt
    2. *return_nets*: an optional boolean argument which is set to True when it is of intrest to return the trained networks, the parameters describing them and the coarse approximation at the discretization nodes. Default value is False.
    3. *verbose*: an optional boolean argument which is set to True if it is of interest to print out some intermediate details about how the algorithm is progressing. Default value is False.
  If one is interested in changing the number of collocation points, the number of processors, or other properties of the hybrid Parareal method, this is where to change them.
  - [utils.py](https://github.com/davidemurari/learnTimeSteppingSchemes/blob/main/ELM/scripts/utils.py) script: This script cocntains several methods of help for the operations we need in the hybrid Parareal method. Here is a list of the included methods with a brief description:
    - *act*: This method defines the set of basis functions we use for the coarse propagator. 
    - *lobattoPoints*, *uniformPoints*: These methods define the collocation points based on their number on the reference interval [0,1].
    - class *flowMap*: This class allows to define the methods and attributes of the ELM-based coarse propagator. The main methods in such a class are:
      - *residual*: which computes the residual between the left and the right hand sides of the ODE where the solution is approximated with an ELM
      - *jac_residual*: which computes the analytical expression of the Jacobian of the residual with respect to the trainable parameters, as presented in the paper's appendix. For the Burgers' equation we implement it as a linearOperator, so we determine it by characterizing its action onto vectors.
      - *approximate_flow_map*: this method solves the optimization problem defined by the minimization of the residual term of the ELM-based coarse propagator. We rely on the *least_squares* method of Scipy, which is also provided with the analytical expression of the Jacobian matrix or linear operator.
      - *analyticalApproximateSolution*: provides an evaluation of the ELM-based coarse propagator on a given input time instant $t$.
      - *plotOverTimeRange*: provides the evaluation of the ELM-based coarse propagator on a numpy array of time instants.
  - [parareal.py](https://github.com/davidemurari/learnTimeSteppingSchemes/blob/main/ELM/scripts/parareal.py) script: This script contains the methods we use to define the hybrid Parareal method. Here is a brief description of the included methods:
    - *fine_integrator*: this is the fine integrator applied to the problem of interest, and it can be applied in parallel since it is provided with the initial conditions of each subinterval. This method relies on the procedure defined in the script *ode_solvers*
    - *getCoarse*: This method uses the utils provided by the script *utils* to obtain the coarse approximation of the solution at the coarse time instants. This is of use for the zeroth iterate of the hybrid Parareal method.
    - *getNextCoarse*: This method computes the next coarse approximation provided with an initial condition.
    - *parallel_solver*: This method coordinates the whole solver since it describes the iterates of the Parareal method, optimizing the weights of a coarse solver when needed. This is the same as Algorithm 4.1 in the paper.
- [ELM/main_one_run.py](https://github.com/davidemurari/learnTimeSteppingSchemes/blob/main/ELM/main_one_run.py): This script allows to run one simulation prescribing the system to consider. It also generates the plots for the obtained simulation.
- [ELM/main_repeated_experiments.py](https://github.com/davidemurari/learnTimeSteppingSchemes/blob/main/ELM/main_repeated_experiments.py): This script allows to repeat the experiments for a given number of times, and stores the average metrics in the folder *savedReports*.
- [flowMapNetwork/scripts](https://github.com/davidemurari/learnTimeSteppingSchemes/tree/main/flowMapNetwork/scripts): The scripts in this folder mostly behave in the same way as those for ELM. The only difference are the following two scripts:
  - [network.py](https://github.com/davidemurari/learnTimeSteppingSchemes/blob/main/flowMapNetwork/scripts/network.py) script: This script contains the definition of the neural network we use to define the flow map based coarse propagator.
  - [training.py](https://github.com/davidemurari/learnTimeSteppingSchemes/blob/main/flowMapNetwork/scripts/training.py) script: This script contains the training routine to find a good set of weights for the flow map-based coarse propagator. 
- [flowMapNetwork/mainTraining.py](https://github.com/davidemurari/learnTimeSteppingSchemes/blob/main/flowMapNetwork/mainTraining.py): This script allows to train the flow map-based coarse propagator, by prescribing the time interval over which it has to be trained.
- [flowMapNetwork/mainParareal.py](https://github.com/davidemurari/learnTimeSteppingSchemes/blob/main/flowMapNetwork/mainParareal.py): This script allows to use the trained flow map-based coarse propagator in the context of the Parareal method. It is asked to input the name of the saved model, which will be stored in the folder *flowMapNetwork/trainedModels/*.