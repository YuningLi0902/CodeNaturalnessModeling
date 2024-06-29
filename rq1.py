import argparse
import os

from utils import loadModelInstance, extractExistingResults, getPatchPathFromDir, computeEntropyWithPrefix, \
    computeEntropyWithContext


def alignNGram(args, lm):
    output_file_name = "result_align_n_gram.csv" if args.output_file_name is None else args.output_file_name
    if os.path.isdir(args.output_dir):
        output_path = os.path.join(args.output_dir, output_file_name)
    else:
        output_path = output_file_name
    if not os.path.isfile(output_path):
        with open(output_path, 'w') as f_out:
            f_out.write(
                'bug_id,file_name,number_of_common_lines,entropy_of_common_lines,number_of_buggy_lines,'
                'entropy_of_buggy_lines,number_of_fix_lines,entropy_of_fix_lines' + '\n')
    existing_results = extractExistingResults(output_path)
    with open(output_path, 'a') as f_out:
        patch_info_list = getPatchPathFromDir("dev", args.root_path_dataset) + getPatchPathFromDir("dev_add", args.root_path_dataset)
        for patch_info in patch_info_list:
            patch_id = patch_info[0]
            file_name = patch_info[1]
            ori_file_path = patch_info[2]
            patched_file_path = patch_info[3]
            if patch_id in existing_results:
                continue
            common_number, common_entropy, buggy_number, buggy_entropy, fix_number, fix_entropy = computeEntropyWithPrefix(
                args, ori_file_path, patched_file_path, lm)
            f_out.write(
                ','.join([patch_id, file_name.replace('.java', ''), str(common_number), str(common_entropy),
                          str(buggy_number), str(buggy_entropy), str(fix_number), str(fix_entropy)]) + '\n')
            f_out.flush()
        f_out.close()


def customModelSettings(args, lm):
    output_file_name = "result_with_custom_setting.csv" if args.output_file_name is None else args.output_file_name
    if os.path.isdir(args.output_dir):
        output_path = os.path.join(args.output_dir, output_file_name)
    else:
        output_path = output_file_name
    if not os.path.isfile(output_path):
        with open(output_path, 'w') as f_out:
            f_out.write(
                'bug_id,file_name,number_of_common_lines,entropy_of_common_lines,number_of_buggy_lines,'
                'entropy_of_buggy_lines,number_of_fix_lines,entropy_of_fix_lines' + '\n')
    existing_results = extractExistingResults(output_path)
    with open(output_path, 'a') as f_out:
        patch_info_list = getPatchPathFromDir("dev", args.root_path_dataset) + getPatchPathFromDir("dev_add", args.root_path_dataset)
        for patch_info in patch_info_list:
            patch_id = patch_info[0]
            file_name = patch_info[1]
            ori_file_path = patch_info[2]
            patched_file_path = patch_info[3]
            if patch_id in existing_results:
                continue
            common_number, common_entropy, buggy_number, buggy_entropy, fix_number, fix_entropy = computeEntropyWithContext(
                args, ori_file_path, patched_file_path, lm)
            f_out.write(
                ','.join([patch_id, file_name.replace('.java', ''), str(common_number), str(common_entropy),
                          str(buggy_number), str(buggy_entropy), str(fix_number), str(fix_entropy)]) + '\n')
            f_out.flush()
        f_out.close()


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. Salesforce/codet5-base")
    parser.add_argument("--work_mode", default=None, type=str, required=True,
                        help="The experiment to run: e.g. n-gram or custom")
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

    if args.work_mode == "n-gram":
        alignNGram(args, lm)
    elif args.work_mode == "custom":
        customModelSettings(args, lm)


if __name__ == '__main__':
    main()
