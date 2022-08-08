import re
from utils import input, output
import os
import sys
import argparse
import datetime
from minizinc import Instance, Model, Solver, Status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--instances', help="Instances to be solved. Can be a list of instances as '1,3,7' or `all`", default="all")
    parser.add_argument('-o', '--output', help="Output directory of the files containing the solutions", default='out')
    parser.add_argument('-t', '--timeout', help="Timeout for the execution of the solvers in seconds. Default=300", default=300)
    parser.add_argument('-r', '--rotation', help="Allow rotation of circuits", default=False, action='store_true')
    parser.add_argument('-p', '--plot', help="Whether to plot the circuits or not - plots of solved instances will be saved in the directory out-img", default=False, action='store_true')
    args = parser.parse_args()

    path = '../../instances/'
    if 'all' in args.instances:
        input_files = [path + i for i in os.listdir(path)]
        #inputs = input.convert_instances(path, input_files)
    else:
        input_files = [path + 'ins-' + i + '.txt' for i in sorted(args.instances.split(','), key=int)]
        #inputs = input.convert_instances(path, input_files)

    out_dir = "../" + args.output

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    if not args.rotation:
        model = 'model.mzn'
    else:
        model = 'model_rotation.mzn'

    model = Model(model)
    gecode = Solver.lookup("gecode")

    times = {}
    solved = []
    not_solved = []
    non_optimal = []
    out_paths = []
    failures = {}
    objectives = []
    for i in input_files:
        instance_n = re.findall(r'\d+', i)[0]
        plate_w, n, widths, heights = input.read_instance(i)
        instance = Instance(gecode, model)
        instance["plate_w"] = plate_w
        instance["n"] = n
        instance["width"] = widths
        instance["height"] = heights
        result = instance.solve(timeout=datetime.timedelta(seconds=int(args.timeout)))
        if result.status is Status.OPTIMAL_SOLUTION or result.solution is not None:
            print("Problem {0} solved in {1}s".format(instance_n, result.statistics['time'].total_seconds()))
            out_file = output.write_output_file(out_dir, i, result, instance, args.rotation)
            chip, circuits = output.load_solution(out_file)
            if args.plot:
                if args.rotation:
                    title = "Instance {0} with rotated allowed".format(instance_n)
                else:
                    title = "Instance {0}".format(instance_n)
                output.plot_grid(chip, circuits, title, i)
            times[instance_n] = result.statistics['time'].total_seconds()
            solved.append(instance_n)
            objectives.append(result.objective)
            if result.status is not Status.OPTIMAL_SOLUTION:
                non_optimal.append(instance_n)
        else:
            print("Problem {0} not solved within the time limit {1}".format(instance_n, args.timeout))
            not_solved.append(instance_n)
        failures[instance_n] = result.statistics["failures"]
    if len(times) != 0:
        mean_time = round(sum(times.values())/len(times), 4)
        print("Solved instances: {0} - Total ({1}/{2})\nExecution times (in seconds): {3}\n"
              "Mean time: {4}\nNot solved: {5}\nNon optimal: {6}\n Failures: {7}\n Objectives: {8}"
              "".format(solved, len(solved), len(input_files), times, mean_time, not_solved, non_optimal, failures, objectives))
