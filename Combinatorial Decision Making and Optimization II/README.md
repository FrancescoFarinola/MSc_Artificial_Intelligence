# Very Large Scale Integration problem - CDMO project Module I

This is the repository for the project of Combinatorial and Decision Making Optimization Module I of University of Bologna.

The aim of the project is to try different approaches to solve the optimization problem of very large scale integration. The approaches tried 
include Constraint Programming (CP), propositional SATisfiability (SAT) and/or its extension to Satisfiability Modulo Theories (SMT).

The problem can be formulated as: given a fixed-width plate and a list of rectangular circuits, decide how to place them
on the plate so that the length of the final device is minimized. 

An exhaustive description of the work done can be read in the report. For SMT and SAT formulations i used notebooks since the library Z3Py has some problems
at being imported in a PyCharm environment, while for CP encodings i wrote a script in python which uses the library _minizinc_ and can be run from command 
line from the working directory as `solver.py` with the following commands:

  - `-i`, `--instances` : followed by `all` or a list of integers e.g. `1,21,33` to indicate the instances to be solved - default = `all`
  - `-o`, `--output` : followed by the name of the output directory where results will be written - default = `out`
  - `-t`, `--timeout` : followed by the timeout value for solving instances in seconds - default = `300` (5mins)
  - `-r`, `--rotation` : a boolean variable to indicate whether rotation is allowed or not - default = `False`
  - `-p`, `--plot` : a boolean variable to indicate whether to plot grids for the solutions and save them - default = `False`

## References

[1] -  Peter J. Stuckey. Square Packing lesson - Coursera. url: https://
www.coursera.org/lecture/advanced-modeling/2-4-1-square-packing-oXHGs?utm_source=link&utm_medium=page_share&utm_content=vlp&utm_campaign=top_button.

[2] - Global Constraint Catalog. url: https://sofdem.github.io/gccat/
gccat/Cdiffn.html.

[3] - Helmut Simonis and Barry O’Sullivan. “Search Strategies for Rectangle
Packing”. In: Principles and Practice of Constraint Programming. Ed. by
Peter J. Stuckey. Berlin, Heidelberg: Springer Berlin Heidelberg, 2008,
pp. 52–66. isbn: 978-3-540-85958-1.

[4] - Eric Huang and Richard Korf. “New Improvements in Optimal Rectan-
gle Packing”. In: Jan. 2009, pp. 511–516.

[5] - E. Huang and R. E. Korf. “Optimal Rectangle Packing: An Absolute
Placement Approach”. In: Journal of Artificial Intelligence Research 46
(Jan. 2013), pp. 47–87. issn: 1076-9757. doi: 10.1613/jair.3735. url:
http://dx.doi.org/10.1613/jair.3735.

[6] - Search annotations - 2.5.4 Restart. url: https://www.minizinc.org/
doc-2.5.5/en/mzn_search.html.

[7] - Hakank - My Z3/Z3Py page. url: http://hakank.org/z3/.

[8] - Takehide Soh et al. “A SAT-Based Method for Solving the Two-Dimensional
Strip Packing Problem”. In: Fundam. Inf. 102.3–4 (Aug. 2010), pp. 467–
487. issn: 0169-2968.
