import sys

def find_overlap(dur_list, start, end):
    overlap = 0
    for dur in dur_list:
        if start < dur[1] and end > dur[0]:
            overlap += min(end, dur[1]) - max(start, dur[0])
    return overlap


def find_max_overlap(spk_overlap):
    max_spk = None
    max_overlap = 0
    for spk in spk_overlap:
        if spk_overlap[spk] > max_overlap:
            max_overlap = spk_overlap[spk]
            max_spk = spk
    return max_spk, max_overlap


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: %s convmap result stm" % sys.argv[1])
        quit()
    convmap = sys.argv[1]
    result = sys.argv[2]
    stm = sys.argv[3]

    wav2feat = {}
    with open(convmap, "r") as f:
        for line in f.readlines():
            wav, feat = line.strip().split("\t")
            wav2feat[wav] = feat

    rec_dur = {}
    with open(stm, "r") as f:
        for line in f.readlines():
            if line.split(" ")[2] != 'Aggregated':
                start = float(line.split(" ")[3])
                end = float(line.split(" ")[4])
                rec = wav2feat[line.split(" ")[0]]
                if rec not in rec_dur:
                    rec_dur[rec] = []
                rec_dur[rec].append([start, end])

    rec_total_dur = {}
    for rec in rec_dur:
        rec_total_dur[rec] = 0
        for info in rec_dur[rec]:
            rec_total_dur[rec] += info[1] - info[0]

    rec_overlap = {}
    with open(result, "r") as f:
        for line in f.readlines():
            [name, feat] = line.strip().split("=")
            wav, spk, _ = name.split("_", 2)
            start = int(feat.rsplit("[",1)[1].split(",",1)[0]) / 100.0
            end = int(feat.rsplit(",",1)[1].split("]")[0]) / 100.0

            if wav not in rec_dur:
                continue

            overlap = find_overlap(rec_dur[wav], start, end)
            if wav not in rec_overlap:
                rec_overlap[wav] = {}
            if spk not in rec_overlap[wav]:
                rec_overlap[wav][spk] = 0
            rec_overlap[wav][spk] += overlap

    total_dur = 0
    overlap_dur = 0
    all_overlap_dur = 0
    for rec in rec_overlap:
        clust, overlap = find_max_overlap(rec_overlap[rec])
        all_overlap = 0
        for spk in rec_overlap[rec]:
            all_overlap += rec_overlap[rec][spk]

        total_dur += rec_total_dur[rec]
        overlap_dur += overlap
        all_overlap_dur += all_overlap
        print("%s: spk overlap %f, all overlap %f" % (rec, overlap/rec_total_dur[rec], all_overlap/rec_total_dur[rec]))

    print("Speaker overlap: %f, all overlap: %f" % (overlap_dur / total_dur, all_overlap_dur / total_dur))
