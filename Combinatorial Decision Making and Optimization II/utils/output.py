import os
from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def write_output_file(path, file, result, instance, rotation=False):
    out_file = path + "/" + os.path.basename(file).replace('ins', 'out')
    with open(out_file, 'w') as f:
        f.write("{0} {1}\n".format(instance["plate_w"], result["plate_h"]))
        f.write("{0}\n".format(instance["n"]))
        for i in range(0,instance["n"]):
            if rotation:
                if result["rotation"][i]:
                    f.write("{0} {1} {2} {3}\n".format(instance["height"][i], instance["width"][i], result["c_x"][i], result["c_y"][i]))
                else:
                    f.write("{0} {1} {2} {3}\n".format(instance["width"][i], instance["height"][i], result["c_x"][i], result["c_y"][i]))
            else:
                f.write("{0} {1} {2} {3}\n".format(instance["width"][i], instance["height"][i], result["c_x"][i], result["c_y"][i]))
    return out_file


def plot_grid(plate, n, plot_title, out_file):
    fig, ax = plt.subplots()
    ax.set_title(plot_title)
    for idx, (width, height, x, y) in enumerate(n):
        label = "Circuit " + str(idx + 1) + " : " + str(width) + "x" + str(height)
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='grey',
                                 facecolor=colors.hsv_to_rgb((idx / len(n) * 0.3, 1, 1)),
                                 label=label)
        ax.add_patch(rect)

        for i in range(1, width):
            ax.plot((x + i, x + i), (y, y + height), color="grey", linewidth=0.2)
        for j in range(1, height):
            ax.plot((x, x + width), (y + j, y + j), color="grey", linewidth=0.2)
    ax.set_xticks(np.arange(plate[0]+1))
    ax.set_yticks(np.arange(plate[1]+1))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.autoscale(enable=True, axis='y', tight=True)

    #Save png plot in the output dir out-img
    path = "../out-img/"
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.savefig(path + os.path.basename(out_file).replace("txt", "png").replace("ins", "out"))

    plt.show()

def load_solution(filename):
    if os.path.isfile(filename):
        with open(filename, 'r') as sol:
            plate_sizes = tuple(int(el) for el in sol.readline().strip().split(" "))
            n = int(sol.readline().strip())
            coords = []
            for i in range(n):
                coords.append(tuple(int(el) for el in sol.readline().strip().split(" ")))
            return plate_sizes, coords
    return None, []
