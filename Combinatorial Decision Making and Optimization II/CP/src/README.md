**Running the script**

In order to run the script the following libraries have to be installed:
  - `minizinc` = 0.5.0
  - `numpy` == 1.21.4
  - `matplotlib` == 3.5.1

The script can be run from command line from the working directory with `solver.py` followed by the arguments:

  - `-i`, `--instances` : followed by `all` or a list of integers e.g. `1,21,33` to indicate the instances to be solved - default = `all`
  - `-o`, `--output` : followed by the name of the output directory where results will be written - default = `out`
  - `-t`, `--timeout` : followed by the timeout value for solving instances in seconds - default = `300` (5mins)
  - `-r`, `--rotation` : a boolean variable to indicate whether rotation is allowed or not - default = `False`
  - `-p`, `--plot` : a boolean variable to indicate whether to plot grids for the solutions and save them - default = `False`

A normal call to run all the instances without allowing rotation and plotting results is: `solver.py -p`

For calling different parameters e.g. with a timeout of 30 seconds, allowing rotation and running only the first 5 instances is: `solver.py -i 1,2,3,4,5 -t 30 -r -p`
