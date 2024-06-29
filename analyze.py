import argparse
import re

from scipy.stats import stats
from sklearn.metrics import confusion_matrix

from utils import calculateCohensD, calculateSNV


def analyzeMeanEntropy(args):
    file_content = []
    with open(args.result_file, 'r') as f:
        for line in f.readlines():
            line_content = line[:-1].split(',')
            file_content.append(line_content)
        f.close()
    common_index = -1
    buggy_index = -1
    fix_index = -1
    for index, title in enumerate(file_content[0]):
        if 'entropy' in title and 'common' in title:
            common_index = index
        elif 'entropy' in title and 'buggy' in title:
            buggy_index = index
        elif 'entropy' in title and 'fix' in title:
            fix_index = index
    common_entropy = 0.0
    common_number = 0
    buggy_entropy = 0.0
    buggy_number = 0
    fix_entropy = 0.0
    fix_number = 0
    for file in file_content[1:]:
        if common_index > -1:
            common_entropy += float(file[common_index]) * int(file[common_index-1])
            common_number += int(file[common_index-1])
        if buggy_index > -1:
            buggy_entropy += float(file[buggy_index]) * int(file[buggy_index - 1])
            buggy_number += int(file[buggy_index - 1])
        if fix_index > -1:
            fix_entropy += float(file[fix_index]) * int(file[fix_index - 1])
            fix_number += int(file[fix_index - 1])
    print_info = ""
    if common_number > 0:
        print_info += "mean entropy of common lines: {:.2f}\n".format(common_entropy/common_number)
    if buggy_number > 0:
        print_info += "mean entropy of buggy lines: {:.2f}\n".format(buggy_entropy/buggy_number)
    if fix_number > 0:
        print_info += "mean entropy of fixed lines: {:.2f}\n".format(fix_entropy/fix_number)
    print(print_info)


def analyzeSignificance(args):
    with open(args.result_file, 'r') as f:
        file_lines = f.readlines()
        f.close()
    common_list = []
    buggy_list = []
    fix_list = []
    pattern = r"line content: (.*)  cross entropy: (.*)"
    whole_file = ''.join(file_lines)
    match = re.findall(pattern, whole_file)
    for item in match:
        if item[0].startswith('+ '):
            fix_list.append(float(item[1]))
        elif item[0].startswith('- '):
            buggy_list.append(float(item[1]))
        else:
            common_list.append(float(item[1]))
    wilcoxon_common_buggy = None
    wilcoxon_buggy_fix = None
    cohensd_common_buggy = None
    cohensd_buggy_fix = None
    SNV_common_buggy = None
    SNV_buggy_fix = None
    if common_list and buggy_list:
        wilcoxon_common_buggy = stats.ranksums(common_list, buggy_list).pvalue
        cohensd_common_buggy = calculateCohensD(common_list, buggy_list)
        SNV_common_buggy = calculateSNV(common_list, buggy_list)
    if buggy_list and fix_list:
        wilcoxon_buggy_fix = stats.ranksums(buggy_list, fix_list).pvalue
        cohensd_buggy_fix = calculateCohensD(buggy_list, fix_list)
        SNV_buggy_fix = calculateSNV(fix_list, buggy_list)
    print("wilcoxon between common code and buggy code: {}".format(wilcoxon_common_buggy))
    print("wilcoxon between buggy code and fixed code: {}".format(wilcoxon_buggy_fix))
    print("cohen's d value between common code and buggy code: {}".format(cohensd_common_buggy))
    print("cohen's d value between buggy code and fixed code: {}".format(cohensd_buggy_fix))
    print("SNV between common code and buggy code: {}".format(SNV_common_buggy))
    print("SNV between buggy code and fixed code: {}".format(SNV_buggy_fix))
    if args.capacity_type == "CNM":
        print("CNM value: {}".format(SNV_common_buggy + SNV_buggy_fix))


def analyzeCDF(args):
    fix_file_list = []
    with open(args.result_file_partial, 'r') as f:
        file_lines_partial = f.readlines()
        fix_file_list.append(file_lines_partial)
        f.close()
    with open(args.result_file, 'r') as f:
        file_lines_complete = f.readlines()
        fix_file_list.append(file_lines_complete)
        f.close()
    pattern = r"line content: (.*)  cross entropy: (.*)"
    statistic_list = []
    for file_lines in fix_file_list:
        fix_list = []
        buggy_list = []
        whole_file = ''.join(file_lines)
        match = re.findall(pattern, whole_file)
        for item in match:
            if item[0].startswith('+ '):
                fix_list.append(float(item[1]))
            elif item[0].startswith('- '):
                buggy_list.append(float(item[1]))
        statistic_list.append([fix_list, buggy_list])
    SNV_partial = calculateSNV(statistic_list[0][0], statistic_list[0][1])
    SNV_complete = calculateSNV(statistic_list[1][0], statistic_list[1][1])
    print("CNM value: {}".format(SNV_complete - SNV_partial))


def analyzePerformance(args):
    file_lines = []
    with open(args.result_file, 'r') as f:
        for line in f.readlines():
            line_content = line.split(',')
            file_lines.append(line_content)
        f.close()
    file_lines = file_lines[1:]
    y_true = []
    y_pred = []
    for i in range(len(file_lines)):
        if file_lines[i][3] == '0.0' or file_lines[i][5] == '0.0':
            continue
        if file_lines[i][3] == '' or file_lines[i][5] == '':
            y_pred.append(0)
        elif float(file_lines[i][5]) < float(file_lines[i][3]):
            y_pred.append(1)
        else:
            y_pred.append(0)
        y_true.append(int(file_lines[i][6].strip()))
    assert len(y_true) == len(y_pred)
    y_true = [1 if item == 0 else 0 for item in y_true]
    y_pred = [1 if item == 0 else 0 for item in y_pred]
    conf_matrix = confusion_matrix(y_true, y_pred)
    tp, fp, tn, fn = conf_matrix[1, 1], conf_matrix[0, 1], conf_matrix[0, 0], conf_matrix[1, 0]
    acc = (tn + tp) / (tn + tp + fn + fp)
    precision = tp / (tp + fp)
    recall_correct = tp / (tp + fn)
    recall_overfitting = tn / (tn + fp)
    f1 = 2 * precision * recall_correct / (precision + recall_correct)
    print("Accuracy: {:.4f}; Precision: {:.4f}; +Recall: {:.4f}; -Recall: {:.4f}; F1: {:.4f}".format(acc, precision,
                                                                                                     recall_correct,
                                                                                                     recall_overfitting,
                                                                                                     f1))


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--work_mode", default=None, type=str, required=True,
                        help="To analyze what: e.g. statistic, significance or performance")
    # Other parameters
    parser.add_argument("--result_file", default=None, type=str,
                        help="Path to the result file related to chosen work mode. When computing CDF, please ensure "
                             "that you give the complete fix result file after this parameter")
    parser.add_argument("--capacity_type", default=None, type=str, help="Choose capacity_type, e.g. CNM or CDF")
    parser.add_argument("--result_file_partial", default=None, type=str,
                        help="Used when computing CDF. Ensure that you give the partial fix result file after this "
                             "parameter")

    args = parser.parse_args()

    if args.work_mode == "statistic":
        analyzeMeanEntropy(args)
    elif args.work_mode == "significance" and args.capacity_type != "CDF":
        analyzeSignificance(args)
    elif args.work_mode == "significance" and args.capacity_type == "CDF":
        analyzeCDF(args)
    elif args.work_mode == "performance":
        analyzePerformance(args)


if __name__ == '__main__':
    main()
