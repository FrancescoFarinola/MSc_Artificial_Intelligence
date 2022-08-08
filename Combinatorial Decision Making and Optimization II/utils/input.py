import os

def read_instance(filename):
    with open(filename, 'r') as f:
        plate_w = int(f.readline().strip())
        n = int(f.readline().strip())
        widths, heights = [], []
        for _ in range(n):
            w, h = f.readline().strip().split(' ')
            widths.append(int(w))
            heights.append(int(h))
    return plate_w, n, widths, heights

'''
def create_dzn(filename):
    out = filename.replace('../instances', 'data').replace('.txt', '.dzn')
    plate_width, n, widths, heights = read_instance(filename)
    with open(out, 'w') as f:
        f.write(f"plate_w={plate_width};\n")
        f.write(f"n={n};\n")
        f.write(f"width={widths};\n")
        f.write(f"height={heights};\n")
    return out
'''

def get_instances(path, input_files):
    if not os.path.isdir('data'):
        os.mkdir('data')
    files = []
    for instance in input_files:
        infile = path + instance
        outfile = read_instance(infile)
        files.append(outfile)
    return files

def write_output(filename, sol):
    with open(filename, 'w') as out:
        out.write(str(sol['width']) + ' ' + str(sol['height']) + '\n')
        out.write(str(sol['num_circuits']) + '\n')
        for w, h, x, y in zip(sol['circuits_w'],
                            sol['circuits_h'],
                            sol['circuits_x'],
                            sol['circuits_y']):
            out.write(str(w) + ' ' + str(h) + ' ' + str(x) + ' ' + str(y) + '\n')
