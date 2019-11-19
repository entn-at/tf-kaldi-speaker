import sys

if len(sys.argv) != 4:
    print("usage: %s score1 score2 score_average" % sys.argv[0])
    quit()

score1 = sys.argv[1]
score2 = sys.argv[2]
score = sys.argv[3]

with open(score, 'w') as fp_out:
    with open(score1, "r") as fp_in1:
        with open(score2, "r") as fp_in2:
            for line1 in fp_in1.readlines():
                line2 = fp_in2.readline()
                utt1, utt2, s1 = line1.strip().split(" ")
                utt1_tmp, utt2_tmp, s2 = line2.strip().split(" ")
                assert(utt1 == utt1_tmp and utt2 == utt2_tmp)
                s = (float(s1) + float(s2)) / 2
                fp_out.write("%s %s %f\n" % (utt1, utt2, s))
