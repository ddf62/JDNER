import random
import numpy as np


def read_text(input_file):
    lines = []
    with open(input_file, 'r') as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    lines.append(words)
                    words = []
            else:
                words.append(line)
        if words:
            lines.append(words)
    np.random.seed(42)
    train_idx = np.random.choice(np.arange(len(lines)), int(0.9 * len(lines)), replace=False)
    valid_idx = np.setdiff1d(np.arange(len(lines)), train_idx)
    np.random.seed(None)
    with open('train.txt', 'w') as f:
        for i in train_idx:
            for j in lines[i]:
                f.write(j)
            f.write('\n')
    with open('dev.txt', 'w') as f:
        for i in valid_idx:
            for j in lines[i]:
                f.write(j)
            f.write('\n')
    return lines


read_text('../datasets/JDNER/train.txt')
