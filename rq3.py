import argparse
import os

from utils import loadModelInstance, getPatchPathFromDir, computeEntropyWithContextForChangeLines


def customAPCA(args, lm):
    output_file_name = "result_entropy_apca.csv" if args.output_file_name is None else args.output_file_name
    if os.path.isdir(args.output_dir):
        output_path = os.path.join(args.output_dir, output_file_name)
    else:
        output_path = output_file_name
    if not os.path.isfile(output_path):
        with open(output_path, 'w') as f_out:
            f_out.write(','.join(
                ['patch_id', 'file_name', 'buggy_lines_number', 'entropy_buggy', 'fix_lines_number', 'entropy_fix',
                 'correctness']) + '\n')
    with open(output_path, 'a') as f_out:
        patch_info_list = getPatchPathFromDir("balance")
        for balance_slice in patch_info_list:
            for patch_info in balance_slice:
                patch_id = patch_info[0]
                file_name = patch_info[1]
                ori_file_path = patch_info[2]
                patched_file_path = patch_info[3]
                correctness = patch_info[4]
            common_number, common_entropy, buggy_number, buggy_entropy, fix_number, fix_entropy = computeEntropyWithContextForChangeLines(
                args, ori_file_path, patched_file_path, lm)
            f_out.write(','.join(
                [patch_id, file_name.replace('.java', ''), str(buggy_number), str(buggy_entropy), str(fix_number),
                 str(fix_entropy), correctness]) + '\n')
            f_out.flush()
    f_out.close()


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. Salesforce/codet5-base")
    # Other parameters
    parser.add_argument("--mask_style", default="MLM", type=str,
                        help="The mask style: e.g. MLM or CLM")
    parser.add_argument("--device", default="0", type=str,
                        help="Use cpu or gpu to calculate entropy: e.g. 'cpu' or device number of gpu")
    parser.add_argument("--max_batch_size", default=32, type=int,
                        help="The maximum batch size when calculating entropy. For large models, remember to set it to "
                             "a small number to avoid OOM")
    parser.add_argument("--context_token_number", default=50, type=int,
                        help="The maximum context token number. Remember to limit it within model's maximum input "
                             "length")
    parser.add_argument("--output_dir", default="", type=str, help="The output directory.")
    parser.add_argument("--output_file_name", default=None, type=str, help="The output file name.")
    parser.add_argument("--root_path_dataset", default=None, type=str, help="The root path of dataset.")

    args = parser.parse_args()

    lm = loadModelInstance(args)

    customAPCA(args, lm)


if __name__ == '__main__':
    main()
