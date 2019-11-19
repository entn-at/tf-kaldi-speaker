import sys
import numpy as np


def main():
    output = sys.argv[1]
    num_input = len(sys.argv) - 2
    index_stat = []
    num_acc = 0
    num_frames = 0
    num_classes = None

    for i in range(num_input):
        with open(sys.argv[i + 2], "r") as f:
            a, n = f.readline().strip().split(" ")
            num_acc += int(a)
            num_frames += int(n)
            c = f.readline().strip()
            if num_classes is None:
                num_classes = int(c)
            else:
                assert(num_classes == int(c))
            for index, line in enumerate(f.readlines()):
                if len(index_stat) <= index:
                    # Add a new entry
                    index_stat.append([0, 0])

                tmp = line.strip().split(" ")
                count = int(tmp[0])
                if count == 0:
                    continue
                index_stat[index][0] += count
                post = np.array([float(t) for t in tmp[1:]])
                index_stat[index][1] += post

    fp_output = open(output, "w")
    fp_output.write("%d %d\n" % (num_acc, num_frames))
    print("%d frames, acc: %f" % (num_frames, float(num_acc)/num_frames))
    fp_output.write("%d\n" % num_classes)
    print("%d classes" % num_classes)
    for i in range(num_classes):
        fp_output.write("%d" % index_stat[i][0])
        if index_stat[i][0] == 0:
            print("No occurrence for class %d" % i)
            continue
        dim = index_stat[i][1].shape[0]
        for j in range(dim):
            fp_output.write(" %e" % index_stat[i][1][j])
        fp_output.write("")
        fp_output.write("\n")
    fp_output.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: %s stat_output stat_1 stat_2 ..." % sys.argv[0])
        quit(1)
    main()