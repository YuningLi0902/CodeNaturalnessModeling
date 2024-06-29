import math
import subprocess
import numpy as np
from scipy.stats import stats

from alpha_repair_code.model import *


def loadModelInstance(args):
    """
    :param args: the parameters input at the beginning of program
    :return: an instance of model according to the parameter args.model_name_or_path
    """
    batch_size = 16
    # Setup CUDA, GPU & distributed training
    device = torch.device(
        "cpu" if args.device == "cpu" or not torch.cuda.is_available() else "cuda:{}".format(args.device))

    if "bert" in args.model_name_or_path:
        model = BertLM(pretrained=args.model_name_or_path, batch_size=batch_size, device=device)
    elif "unixcoder" in args.model_name_or_path:
        model = UniXcoder(pretrained=args.model_name_or_path, batch_size=batch_size, device=device)
    else:
        model = SpanLM(pretrained=args.model_name_or_path, batch_size=batch_size, device=device)

    if "codet5" in args.model_name_or_path:
        model.model.reinit(model.tokenizer, False, set(), 'java', '', '')

    return model


def extractExistingResults(output_path):
    """
    :param output_path: target csv file path to extract existing results
    :return: all the records of the csv file
    """
    with open(output_path, 'r') as f:
        lines = f.readlines()
    patch_id_list = []
    for line in lines:
        patch_id_list.append(line.split(',')[0])
    return patch_id_list


def findFilesRecursive(dir, prefix, postfix):
    """
    :param dir: target directory to find out files recursively
    :param prefix: prefix of a file name
    :param postfix: suffix of a file name
    :return: paths of files in the target directory
    """
    file_paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.startswith(prefix) and file.endswith(postfix):
                file_paths.append(os.path.join(root, file))

    return file_paths


def getPatchPathFromDir(dataset_name, root_dir_of_dataset='./datasets'):
    """
    :param dataset_name: name of dataset
    :param root_dir_of_dataset: the root directory of the "datasets" directory
    :return: patch information contained in dataset
    """
    assert dataset_name in ['dev', 'dev_add', 'prapr', 'prapr_add', 'ase', 'partial', 'complete', 'balance']
    if root_dir_of_dataset is None:
        root_dir_of_dataset = './datasets'
    patch_list_all = []  # each item represents a patch, with its patch_id, file_name, ori_path and patch_path
    if 'dev' in dataset_name:
        dir_name = 'developer_patches_1.2' if dataset_name == 'dev' else 'developer_patches_2.0'
        dataset_dir = os.path.join(root_dir_of_dataset, dir_name)
        for proj in os.listdir(dataset_dir):
            proj_dir = os.path.join(dataset_dir, proj)
            for id in os.listdir(proj_dir):
                proj_id_dir = os.path.join(proj_dir, id)
                patch_dir = os.path.join(proj_id_dir, 'mutant-0')
                patch_id = '_'.join(['dev', proj, id, 'mutant-0'])
                src_patch_path = os.path.join(patch_dir, id + '.src.patch')
                try:
                    with open(src_patch_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                except:
                    with open(src_patch_path, 'r', encoding='ISO-8859-1') as f:
                        lines = f.readlines()
                file_name_get = False
                for idx, line in enumerate(lines):
                    if line.startswith('diff') and not lines[idx + 1].startswith('deleted'):
                        file_name = line.strip().split('/')[-1].split('.')[0]
                        file_name_get = True
                        break
                    elif line.startswith('Index: '):
                        file_name = line.strip().split('/')[-1].split('.')[0]
                        file_name_get = True
                assert file_name_get, 'File name not found in ' + src_patch_path
                ori_file_path = os.path.join(patch_dir, 'buggy-' + file_name + '.java')
                patched_file_path = os.path.join(patch_dir, 'patched-' + file_name + '.java')
                patch_list_all.append([patch_id, file_name, ori_file_path, patched_file_path])
    elif 'prapr' in dataset_name:
        dir_name = 'prapr_src_patches_1.2' if dataset_name == 'prapr' else 'prapr_src_patches_2.0'
        prapr_dir = os.path.join(root_dir_of_dataset, dir_name)
        for proj in os.listdir(prapr_dir):
            proj_dir = os.path.join(prapr_dir, proj)
            for id in os.listdir(proj_dir):
                proj_id_dir = os.path.join(proj_dir, id)
                for mutant_id in os.listdir(proj_id_dir):
                    patch_dir = os.path.join(proj_id_dir, mutant_id)
                    if os.path.isfile(os.path.join(patch_dir, 'NO_DIFF')):
                        continue
                    if os.path.isfile(os.path.join(patch_dir, 'CANT_FIX')):
                        continue
                    patch_id = '_'.join(['prapr', proj, id, mutant_id])
                    man_patched = False
                    fixed_patched = False
                    patched_file_path = None
                    ori_file_path = None
                    file_name = None
                    for f in os.listdir(patch_dir):
                        if f.startswith('man-patched-'):
                            man_patched = True
                            patched_file_path = os.path.join(patch_dir, f)
                            break
                    if not man_patched:
                        for f in os.listdir(patch_dir):
                            if f.startswith('fixed-patched-'):
                                fixed_patched = True
                                patched_file_path = os.path.join(patch_dir, f)
                                break
                    for f in os.listdir(patch_dir):
                        if f.startswith('patched-') and not fixed_patched and not man_patched:
                            patched_file_path = os.path.join(patch_dir, f)
                        if f.startswith('ori-'):
                            ori_file_path = os.path.join(patch_dir, f)
                            file_name = f[4:-5]
                    assert os.path.isfile(patched_file_path) and os.path.isfile(ori_file_path)
                    patch_list_all.append([patch_id, file_name, ori_file_path, patched_file_path])
    elif dataset_name == 'partial' or dataset_name == 'complete':
        prapr_dir_list = [os.path.join(root_dir_of_dataset, 'prapr_src_patches_1.2'),
                          os.path.join(root_dir_of_dataset, 'prapr_src_patches_2.0')]
        ase_dir = os.path.join(root_dir_of_dataset, 'ASE_Patches')
        with open(os.path.join(root_dir_of_dataset, 'overlapping_patches_ASE_prapr.txt'), 'r') as f:
            overlap = f.read()
            f.close()
        for prapr_dir in prapr_dir_list:
            for proj in os.listdir(prapr_dir):
                proj_dir = os.path.join(prapr_dir, proj)
                for id in os.listdir(proj_dir):
                    proj_id_dir = os.path.join(proj_dir, id)
                    for mutant_id in os.listdir(proj_id_dir):
                        patch_dir = os.path.join(proj_id_dir, mutant_id)
                        if os.path.isfile(os.path.join(patch_dir, 'NO_DIFF')):
                            continue
                        if os.path.isfile(os.path.join(patch_dir, 'CANT_FIX')):
                            continue
                        patch_id = '_'.join(['prapr', proj, id, mutant_id])
                        man_patched = False
                        fixed_patched = False
                        patched_file_path = None
                        ori_file_path = None
                        file_name = None
                        if 'correct' in os.listdir(patch_dir) and dataset_name == 'partial':
                            continue
                        if 'correct' not in os.listdir(patch_dir) and dataset_name == 'complete':
                            continue
                        for f in os.listdir(patch_dir):
                            if f.startswith('man-patched-'):
                                man_patched = True
                                patched_file_path = os.path.join(patch_dir, f)
                                break
                        if not man_patched:
                            for f in os.listdir(patch_dir):
                                if f.startswith('fixed-patched-'):
                                    fixed_patched = True
                                    patched_file_path = os.path.join(patch_dir, f)
                                    break
                        for f in os.listdir(patch_dir):
                            if f.startswith('patched-') and not fixed_patched and not man_patched:
                                patched_file_path = os.path.join(patch_dir, f)
                            if f.startswith('ori-'):
                                ori_file_path = os.path.join(patch_dir, f)
                                file_name = f[4:-5]
                        assert os.path.isfile(patched_file_path) and os.path.isfile(ori_file_path)
                        patch_list_all.append([patch_id, file_name, ori_file_path, patched_file_path])
        for patch_path in findFilesRecursive(ase_dir, "src", '.patch'):
            patch_dir = os.path.dirname(patch_path)
            if os.path.isfile(os.path.join(patch_dir, 'NOT_PLAUSIBLE')):
                continue
            patch_id = '_'.join(patch_dir.split('/')[-5:])
            ori_file_path = os.path.join(patch_dir, 'buggy1.java')
            patched_file_path = os.path.join(patch_dir, 'tool-patch1.java')
            assert os.path.isfile(ori_file_path) and os.path.isfile(patched_file_path)
            if patched_file_path[9:] in overlap:
                continue
            if 'Dcorrect' in patched_file_path and dataset_name == 'partial':
                continue
            if 'Doverfitting' in patched_file_path and dataset_name == 'complete':
                continue
            patch_list_all.append([patch_id, 'buggy1', ori_file_path, patched_file_path])
    elif dataset_name == 'balance':
        projects_d4j12 = ['Chart', 'Closure', 'Lang', 'Math', 'Mockito', 'Time']
        projects_d4j20 = ['Cli', 'Codec', 'Compress', 'Csv', 'Gson', 'JacksonCore', 'JacksonDatabind', 'JacksonXml',
                          'Jsoup', 'JxPath']
        for i in range(1, 11):
            patch_names = []
            patch_list = []
            with open(os.path.join(root_dir_of_dataset,
                                   'balanced_dataset/balanced_dataset_patches-{}.txt'.format(str(i)))) as f_in:
                for line in f_in.readlines():
                    patch_names.append(line.strip())
                f_in.close()
            for patch_name in patch_names:
                patch_path = ''
                patched_file_path = ''
                ori_file_path = ''
                file_name = ''
                patch_correctness = str(0)
                bug_id = patch_name.split('_')[0]
                bug = bug_id.split('-')[0]
                b_id = bug_id.split('-')[1]
                specific_id = patch_name.split('_')[1]
                if specific_id.endswith('Dcorrect') or specific_id.endswith('Doverfitting'):
                    if patch_name.endswith('Dcorrect'):
                        patch_correctness = str(1)
                    tool_name = specific_id.split('-')[0]
                    folder1 = specific_id.split('-')[2]
                    folder2 = specific_id.split('-')[1]
                    if folder2 == '0':
                        patch_path = os.path.join(root_dir_of_dataset,
                                                  'ASE_Patches/Patches_ICSE/{}/{}/{}/{}'.format(folder1, tool_name, bug,
                                                                                                b_id))
                    else:
                        patch_path = os.path.join(root_dir_of_dataset,
                                                  'ASE_Patches/Patches_others/{}/{}/{}/{}/{}'.format(folder1, tool_name,
                                                                                                     bug, b_id,
                                                                                                     folder2))
                    patch_id = '_'.join(patch_path.split('/')[-5:])
                    file_name = 'buggy1'
                    ori_file_path = os.path.join(patch_path, 'buggy1.java')
                    patched_file_path = os.path.join(patch_path, 'tool-patch1.java')
                else:
                    if bug in projects_d4j12:
                        patch_path = 'datasets/prapr_src_patches_1.2/{}/{}/{}'.format(bug, b_id, specific_id)
                    elif bug in projects_d4j20:
                        patch_path = 'datasets/prapr_src_patches_2.0/{}/{}/{}'.format(bug, b_id, specific_id)
                    else:
                        assert bug in projects_d4j12 or bug in projects_d4j20
                    patch_id = '_'.join(['prapr', bug, b_id, specific_id])
                    man_patched = False
                    fixed_patched = False
                    for f in os.listdir(patch_path):
                        if f.startswith('man-patched-'):
                            man_patched = True
                            patched_file_path = os.path.join(patch_path, f)
                            break
                    if not man_patched:
                        for f in os.listdir(patch_path):
                            if f.startswith('fixed-patched-'):
                                fixed_patched = True
                                patched_file_path = os.path.join(patch_path, f)
                                break
                    for f in os.listdir(patch_path):
                        if f.startswith('patched-') and not fixed_patched and not man_patched:
                            patched_file_path = os.path.join(patch_path, f)
                        if f.startswith('ori-'):
                            ori_file_path = os.path.join(patch_path, f)
                            file_name = f[4:-5]
                    if os.path.isfile(os.path.join(patch_path, 'correct')):
                        patch_correctness = str(1)
                assert os.path.isfile(patched_file_path) and os.path.isfile(ori_file_path)
                patch_list.append([patch_id, file_name, ori_file_path, patched_file_path, patch_correctness])
            patch_list_all.append(patch_list)
    else:
        dataset_dir = os.path.join(root_dir_of_dataset, 'ASE_Patches')
        for patch_path in findFilesRecursive(dataset_dir, "src", '.patch'):
            patch_dir = os.path.dirname(patch_path)
            if os.path.isfile(os.path.join(patch_dir, 'NOT_PLAUSIBLE')):
                continue
            patch_id = '_'.join(patch_dir.split('/')[-5:])
            ori_file_path = os.path.join(patch_dir, 'buggy1.java')
            patched_file_path = os.path.join(patch_dir, 'tool-patch1.java')
            assert os.path.isfile(ori_file_path) and os.path.isfile(patched_file_path)
            patch_list_all.append([patch_id, 'buggy1', ori_file_path, patched_file_path])
    return patch_list_all


def balanceSequenceLength(seq1, seq2, target_length):
    """
    :param seq1: one sequence
    :param seq2: another sequence
    :param target_length: when concatenating two sequences, the target total length
    :return: two sequences after balancing their length
    """
    while len(seq1) + len(seq2) > target_length:
        if len(seq1) > len(seq2):
            seq1 = seq1[1:]
        else:
            seq2 = seq2[:-1]
    return seq1, seq2


def getFormatInput(args, prefix, suffix, context_tokens=512, unixcoder_mode="encoder_decoder"):
    """
    :param args: the parameters input at the beginning of program
    :param prefix: prefix of masked token
    :param suffix: suffix of masked token
    :param context_tokens: maximum length of prefix or suffix
    :param unixcoder_mode: the working mode of unixcoder
    :return: tokenized sequence with special tokens
    """
    prefix = prefix[:context_tokens]
    suffix = suffix[:context_tokens]
    if 'incoder' in args.model_name_or_path:
        prefix, suffix = balanceSequenceLength(prefix, suffix, context_tokens - 4)
        input_tokens = [2]
        input_tokens.extend(prefix)
        input_tokens.extend([50261])
        input_tokens.extend(suffix)
        input_tokens.extend([50262])
        input_tokens.extend([50261])
    elif 'codet5' in args.model_name_or_path:
        prefix, suffix = balanceSequenceLength(prefix, suffix, context_tokens - 3)
        input_tokens = [1]
        input_tokens.extend(prefix)
        input_tokens.extend([32099])
        input_tokens.extend(suffix)
        input_tokens.extend([2])
    elif 'bert' in args.model_name_or_path and 'codebert' not in args.model_name_or_path:
        prefix, suffix = balanceSequenceLength(prefix, suffix, context_tokens - 3)
        input_tokens = [101]
        input_tokens.extend(prefix)
        input_tokens.extend([103])
        input_tokens.extend(suffix)
        input_tokens.extend([102])
    elif 'codebert' in args.model_name_or_path:
        prefix, suffix = balanceSequenceLength(prefix, suffix, context_tokens - 3)
        input_tokens = [0]
        input_tokens.extend(prefix)
        input_tokens.extend([50264])
        input_tokens.extend(suffix)
        input_tokens.extend([2])
    elif 'unixcoder' in args.model_name_or_path:
        if unixcoder_mode == 'encoder_only':
            prefix, suffix = balanceSequenceLength(prefix, suffix, context_tokens - 5)
            input_tokens = [0, 6, 2]
            input_tokens.extend(prefix)
            input_tokens.extend([19])
            input_tokens.extend(suffix)
            input_tokens.extend([2])
        elif unixcoder_mode == 'decoder_only':
            prefix, suffix = balanceSequenceLength(prefix, suffix, context_tokens - 5)
            input_tokens = [0, 7, 2]
            input_tokens.extend(prefix)
            input_tokens.extend([19])
            input_tokens.extend(suffix)
            input_tokens.extend([2])
        else:
            prefix, suffix = balanceSequenceLength(prefix, suffix, context_tokens - 5)
            input_tokens = [0, 5, 2]
            input_tokens.extend(prefix)
            input_tokens.extend([19])
            input_tokens.extend(suffix)
            input_tokens.extend([2])
    else:
        input_tokens = []
    return input_tokens


def getBatchedTokenScores(args, lm, input_token_list, token_id_list, max_batch_size=16,
                          unixcoder_mode='encoder_decoder'):
    """
    :param args: the parameters input at the beginning of program
    :param lm: the model instance
    :param input_token_list: the list of masked tokens
    :param token_id_list: the id list of masked tokens
    :param max_batch_size: maximum batch size
    :param unixcoder_mode: the working mode of unixcoder
    :return: the probability of input masked tokens
    """
    min_score = 1e-10
    if len(set([len(item) for item in input_token_list])) != 1:
        max_batch_size = 1
    ori_input_token_list = input_token_list
    ori_token_id_list = token_id_list
    batch_counter = math.ceil(len(input_token_list) / max_batch_size)
    score_next_target_tokens = []
    for batch_c in range(batch_counter):
        input_token_list = ori_input_token_list[batch_c * max_batch_size:(batch_c + 1) * max_batch_size]
        token_id_list = ori_token_id_list[batch_c * max_batch_size:(batch_c + 1) * max_batch_size]
        if 'incoder' in args.model_name_or_path:
            input_tokens = torch.tensor(input_token_list).to(lm.model.device)
            token_ids = [torch.tensor(int(token_id)).to(lm.model.device) for token_id in token_id_list]
            context_tokens = len(input_token_list[0])
            with torch.no_grad():
                raw_o = lm.model.generate(input_tokens,
                                          max_length=1 + context_tokens,
                                          do_sample=True,
                                          output_scores=True,
                                          return_dict_in_generate=True,
                                          temperature=1,
                                          top_k=200,
                                          top_p=1,
                                          use_cache=True)
                t_outputs = lm.tokenizer.batch_decode(raw_o.sequences, skip_special_tokens=False)
                for t_o in t_outputs:
                    assert lm.infill_ph in t_o, 'infill_ph not in output'
                for i in range(len(token_ids)):
                    next_target_token_score_dist = raw_o.scores[0][i].softmax(dim=0)
                    score_next_target_token = next_target_token_score_dist[token_ids[i]]
                    score_next_target_tokens.append(max(score_next_target_token, min_score))
        elif 'codet5' in args.model_name_or_path:
            input_tokens = torch.tensor(input_token_list).to(lm.model.device)
            token_ids = [torch.tensor(int(token_id)).to(lm.model.device) for token_id in token_id_list]
            max_length = 3
            if args.model_name_or_path.split('-')[-2] == 'base':
                max_length = 4
            with torch.no_grad():
                lm.model.reinit(lm.tokenizer, False, set(), 'java', '', '')
                raw_o = lm.model.generate(input_tokens,
                                          max_length=max_length,
                                          do_sample=True,
                                          output_scores=True,
                                          return_dict_in_generate=True,
                                          temperature=1,
                                          top_k=200,
                                          top_p=1,
                                          use_cache=True)
                t_outputs = lm.model.tokenizer.batch_decode(raw_o.sequences, skip_special_tokens=False)
                for t_o in t_outputs:
                    assert lm.infill_ph in t_o, 'infill_ph not in output'
                for i in range(len(token_ids)):
                    output_sequences = raw_o.sequences
                    min_index = output_sequences[i, 1:].tolist().index(
                        lm.tokenizer.encode(lm.infill_ph, add_special_tokens=False)[0])
                    next_target_token_score_dist = raw_o.scores[min_index + 1][i].softmax(dim=0)
                    score_next_target_token = next_target_token_score_dist[token_ids[i]]
                    score_next_target_tokens.append(max(score_next_target_token, min_score))
        elif 'bert' in args.model_name_or_path:
            input_tokens = torch.tensor(input_token_list).to(lm.model.device)
            token_ids = [torch.tensor(int(token_id)).to(lm.model.device) for token_id in token_id_list]
            context_tokens = len(input_token_list[0])
            attention_masks = []
            for i in range(len(input_token_list)):
                attention_masks.append([1] * context_tokens)
            attention_masks = torch.tensor(attention_masks).to(lm.model.device)
            with torch.no_grad():
                raw_o = lm.model(input_ids=input_tokens, attention_mask=attention_masks)
                for i in range(len(token_ids)):
                    mask_index = input_token_list[i].index(
                        lm.tokenizer.encode(lm.infill_ph, add_special_tokens=False)[0])
                    next_target_token_score_dist = raw_o['logits'][i, mask_index, :].softmax(dim=0)
                    score_next_target_token = next_target_token_score_dist[token_ids[i]]
                    score_next_target_tokens.append(max(score_next_target_token, min_score))
        elif 'unixcoder' in args.model_name_or_path:
            input_tokens = torch.tensor(input_token_list).to(lm.model.device)
            token_ids = [torch.tensor(int(token_id)).to(lm.model.device) for token_id in token_id_list]
            with torch.no_grad():
                if unixcoder_mode == 'encoder_only':
                    attention_masks = input_tokens.ne(lm.config.pad_token_id)
                    raw_o = lm.model(input_ids=input_tokens, attention_mask=attention_masks)
                    for i in range(len(token_ids)):
                        mask_index = input_token_list[i].index(
                            lm.tokenizer.encode(lm.infill_ph, add_special_tokens=False)[0])
                        next_target_token_score_dist = raw_o['logits'][i, mask_index, :].softmax(dim=0)
                        score_next_target_token = next_target_token_score_dist[token_ids[i]]
                        score_next_target_tokens.append(max(score_next_target_token, min_score))
                elif unixcoder_mode == 'decoder_only':
                    for i in range(len(token_ids)):
                        _, scores = lm.generate(input_tokens, decoder_only=True, beam_size=1, max_length=2)
                        next_target_token_score_dist = scores[i][1][0].softmax(dim=0)
                        score_next_target_token = next_target_token_score_dist[token_ids[i]]
                        score_next_target_tokens.append(max(score_next_target_token, min_score))
                if unixcoder_mode == 'encoder_decoder':
                    for i in range(len(token_ids)):
                        _, scores = lm.generate(input_tokens, decoder_only=False, beam_size=1, max_length=2)
                        next_target_token_score_dist = scores[i][1][0].softmax(dim=0)
                        score_next_target_token = next_target_token_score_dist[token_ids[i]]
                        score_next_target_tokens.append(max(score_next_target_token, min_score))
    return score_next_target_tokens


def computeEntropyWithPrefix(args, buggy_file, fix_file, model, tokens_n=10, remove_redundant=True):
    """
    :param args: the parameters input at the beginning of program
    :param buggy_file: file path of buggy file
    :param fix_file: file path of fixed file
    :param model: instance of model
    :param tokens_n: the "n" of n-gram model
    :param remove_redundant: whether remove redundant information of buggy file and fixed file
    :return: cross entropy of code lines given prefix
    """
    tmp_patch_file = 'tmp_no_comment_patch_{}.java'.format(os.getpid())
    tmp_buggy_file = 'tmp_no_comment_buggy_{}.java'.format(os.getpid())
    line_entropies = [["file name: {}".format(buggy_file.split('/')[-1].split('-')[1])]]
    if remove_redundant:
        with open(tmp_buggy_file, 'w') as f:
            stdout = subprocess.run(
                ['java', '-jar', 'JavaFileExtractor.jar', '-path', buggy_file, '-customEdit', '-removeComments',
                 '-removePackage', '-removeImports'],
                stdout=subprocess.PIPE, env={'PATH': '/usr/bin/'}).stdout
            try:
                stdout = stdout.decode('utf-8')
            except UnicodeDecodeError:
                stdout = stdout.decode('iso-8859-1')
            buggy_file = tmp_buggy_file
            f.write(stdout)
            f.close()
        with open(tmp_patch_file, 'w') as f:
            stdout = subprocess.run(
                ['java', '-jar', 'JavaFileExtractor.jar', '-path', fix_file, '-customEdit', '-removeComments',
                 '-removePackage', '-removeImports'],
                stdout=subprocess.PIPE, env={'PATH': '/usr/bin/'}).stdout
            try:
                stdout = stdout.decode('utf-8')
            except UnicodeDecodeError:
                stdout = stdout.decode('iso-8859-1')
            fix_file = tmp_patch_file
            f.write(stdout)
            f.close()
    stdout = diff_output = subprocess.run(['diff', '--unified=1000000', '-b', buggy_file, fix_file],
                                          stdout=subprocess.PIPE).stdout
    try:
        diff_output = stdout.decode('utf-8')
    except UnicodeDecodeError:
        diff_output = stdout.decode('iso-8859-1')
    if remove_redundant:
        os.remove(tmp_patch_file)
        os.remove(tmp_buggy_file)
    file_lines = diff_output.split('\n')[3:]
    common_lines_counter = 0
    buggy_lines_counter = 0
    fix_lines_counter = 0
    common_lines_entropy = 0.0
    buggy_lines_entropy = 0.0
    fix_lines_entropy = 0.0
    common_token_prefix = []
    buggy_token_prefix = []
    fix_token_prefix = []
    for line in file_lines:
        if line.startswith('-'):
            if remove_redundant:
                line = line[1:].strip()
            else:
                line = line[1:]
            if line == '':
                continue
            line_record = ["- {}".format(line)]
            buggy_lines_counter += 1
            line_tokenize = model.tokenizer.encode(line, add_special_tokens=False)
            line_format_list = []
            line_id_list = []
            for id in line_tokenize:
                prefix = buggy_token_prefix[max(len(buggy_token_prefix) - tokens_n + 1, 0):]
                buggy_token_prefix.append(id)
                format_token_ids = getFormatInput(args, prefix, [])
                line_format_list.append(format_token_ids)
                line_id_list.append(id)
            line_scores = getBatchedTokenScores(args, model, line_format_list, line_id_list, args.max_batch_size)
            neg_logs = [-math.log(score) for score in line_scores]
            cross_entropy = sum(neg_logs) / len(neg_logs)
            buggy_lines_entropy += cross_entropy
            line_record.append(cross_entropy)
            line_entropies.append(line_record)
        elif line.startswith('+'):
            if remove_redundant:
                line = line[1:].strip()
            else:
                line = line[1:]
            if line == '':
                continue
            line_record = ["+ {}".format(line)]
            fix_lines_counter += 1
            line_tokenize = model.tokenizer.encode(line.strip(), add_special_tokens=False)
            line_format_list = []
            line_id_list = []
            for id in line_tokenize:
                prefix = fix_token_prefix[max(len(fix_token_prefix) - tokens_n + 1, 0):]
                fix_token_prefix.append(id)
                format_token_ids = getFormatInput(args, prefix, [])
                line_format_list.append(format_token_ids)
                line_id_list.append(id)
            line_scores = getBatchedTokenScores(args, model, line_format_list, line_id_list, args.max_batch_size)
            neg_logs = [-math.log(score) for score in line_scores]
            cross_entropy = sum(neg_logs) / len(neg_logs)
            fix_lines_entropy += cross_entropy
            line_record.append(cross_entropy)
            line_entropies.append(line_record)
        else:
            if remove_redundant:
                line = line.strip()
            if line == '':
                continue
            line_record = [line]
            common_lines_counter += 1
            line_tokenize = model.tokenizer.encode(line.strip(), add_special_tokens=False)
            line_format_list = []
            line_id_list = []
            for id in line_tokenize:
                prefix = common_token_prefix[max(len(common_token_prefix) - tokens_n + 1, 0):]
                common_token_prefix.append(id)
                buggy_token_prefix.append(id)
                fix_token_prefix.append(id)
                format_token_ids = getFormatInput(args, prefix, [])
                line_format_list.append(format_token_ids)
                line_id_list.append(id)
            line_scores = getBatchedTokenScores(args, model, line_format_list, line_id_list, args.max_batch_size)
            neg_logs = [-math.log(score) for score in line_scores]
            cross_entropy = sum(neg_logs) / len(neg_logs)
            common_lines_entropy += cross_entropy
            line_record.append(cross_entropy)
            line_entropies.append(line_record)
    if common_lines_counter > 0:
        common_lines_entropy = common_lines_entropy / common_lines_counter
    else:
        common_lines_entropy = 0.0
    if buggy_lines_counter > 0:
        buggy_lines_entropy = buggy_lines_entropy / buggy_lines_counter
    else:
        buggy_lines_entropy = 0.0
    if fix_lines_counter > 0:
        fix_lines_entropy = fix_lines_entropy / fix_lines_counter
    else:
        fix_lines_entropy = 0.0
    print(line_entropies[0][0])
    for line_entropie in line_entropies[1:]:
        print("line content: {}  cross entropy: {}".format(line_entropie[0], str(line_entropie[1])))
    print()
    return common_lines_counter, common_lines_entropy, \
           buggy_lines_counter, buggy_lines_entropy, \
           fix_lines_counter, fix_lines_entropy


def computeEntropyWithContext(args, buggy_file, fix_file, model, blank_token='\n', remove_redundant=True):
    """
    :param args: the parameters input at the beginning of program
    :param buggy_file: file path of buggy file
    :param fix_file: file path of fixed file
    :param model: instance of model
    :param blank_token: the token to replace empty code snippet
    :param remove_redundant: whether remove redundant information of buggy file and fixed file
    :return: cross entropy of code lines given context
    """
    tmp_patch_file = 'tmp_no_comment_patch_{}.java'.format(os.getpid())
    tmp_buggy_file = 'tmp_no_comment_buggy_{}.java'.format(os.getpid())
    line_entropies = [["file name: {}".format(buggy_file.split('/')[-1].split('-')[1])]]
    if remove_redundant:
        with open(tmp_buggy_file, 'w') as f:
            stdout = subprocess.run(
                ['java', '-jar', 'JavaFileExtractor.jar', '-path', buggy_file, '-customEdit', '-removeComments',
                 '-removePackage', '-removeImports'],
                stdout=subprocess.PIPE, env={'PATH': '/usr/bin/java/bin'}).stdout
            try:
                stdout = stdout.decode('utf-8')
            except UnicodeDecodeError:
                stdout = stdout.decode('iso-8859-1')
            buggy_file = tmp_buggy_file
            f.write(stdout)
            f.close()
        with open(tmp_patch_file, 'w') as f:
            stdout = subprocess.run(
                ['java', '-jar', 'JavaFileExtractor.jar', '-path', fix_file, '-customEdit', '-removeComments',
                 '-removePackage', '-removeImports'],
                stdout=subprocess.PIPE, env={'PATH': '/usr/bin/java/bin'}).stdout
            try:
                stdout = stdout.decode('utf-8')
            except UnicodeDecodeError:
                stdout = stdout.decode('iso-8859-1')
            fix_file = tmp_patch_file
            f.write(stdout)
            f.close()
    stdout = diff_output = subprocess.run(['diff', '--unified=1000000', '-b', buggy_file, fix_file],
                                          stdout=subprocess.PIPE).stdout
    try:
        diff_output = stdout.decode('utf-8')
    except UnicodeDecodeError:
        diff_output = stdout.decode('iso-8859-1')
    if remove_redundant:
        os.remove(tmp_patch_file)
        os.remove(tmp_buggy_file)
    file_lines = diff_output.split('\n')[3:]
    common_lines_counter = 0
    buggy_lines_counter = 0
    fix_lines_counter = 0
    common_lines_entropy = 0.0
    buggy_lines_entropy = 0.0
    fix_lines_entropy = 0.0
    common_token_prefix = []
    buggy_token_prefix = []
    fix_token_prefix = []
    deal_with_one_hunk = False
    have_add = False
    have_del = False
    hunk_prefix = []
    suffix = []
    suffix_id_length = 0
    bottom_line = 0  # point to the line not been contained in the suffix
    for line_id in range(len(file_lines)):
        line = file_lines[line_id]
        if line.startswith('-'):
            if remove_redundant:
                line = line[1:].strip()
            else:
                line = line[1:]
            if line == '':
                continue
            line_record = ["- {}".format(line)]
            buggy_lines_counter += 1
            line_tokenize = model.tokenizer.encode(line, add_special_tokens=False)
            line_format_list = []
            line_id_list = []
            local_cache = []
            for id_id in range(len(line_tokenize)):
                id = line_tokenize[id_id]
                prefix = buggy_token_prefix[max(len(buggy_token_prefix) - args.context_token_number, 0):]
                buggy_token_prefix.append(id)
                ids_left = line_tokenize[id_id + 1:]
                if id_id == 0:
                    if not deal_with_one_hunk:
                        hunk_prefix = prefix
                    next_line_id = line_id + 1
                    while next_line_id < len(file_lines):
                        next_line = file_lines[next_line_id]
                        if next_line.startswith('-'):
                            next_line = next_line[1:]
                            if remove_redundant:
                                next_line = next_line.strip()
                            next_line_tokenize = model.tokenizer.encode(next_line, add_special_tokens=False)
                            local_cache.extend(next_line_tokenize)
                            next_line_id += 1
                        else:
                            break
                    if bottom_line <= line_id:
                        bottom_line = line_id + 1
                    while bottom_line < len(file_lines):
                        next_line = file_lines[bottom_line]
                        if next_line.startswith('+') or next_line.startswith('-'):
                            bottom_line += 1
                            continue
                        if remove_redundant:
                            next_line = next_line.strip()
                        if next_line == '':
                            bottom_line += 1
                            continue
                        next_line_tokenize = model.tokenizer.encode(next_line, add_special_tokens=False)
                        if suffix_id_length < args.context_token_number:
                            suffix.append(next_line_tokenize)
                            suffix_id_length += len(next_line_tokenize)
                        else:
                            break
                        bottom_line += 1
                if args.mask_style == 'MLM':
                    current_suffix = ids_left
                    current_suffix.extend(local_cache)
                    current_suffix.extend([item for sublist in suffix for item in sublist])
                else:
                    current_suffix = [item for sublist in suffix for item in sublist]
                format_token_ids = getFormatInput(args, prefix, current_suffix, args.context_token_number)
                line_format_list.append(format_token_ids)
                line_id_list.append(id)
            line_scores = getBatchedTokenScores(args, model, line_format_list, line_id_list, args.max_batch_size)
            neg_logs = [-math.log(score) for score in line_scores]
            cross_entropy = sum(neg_logs) / len(neg_logs)
            buggy_lines_entropy += cross_entropy
            line_record.append(cross_entropy)
            line_entropies.append(line_record)
            if not deal_with_one_hunk:
                deal_with_one_hunk = True
            if deal_with_one_hunk:
                have_del = True
        elif line.startswith('+'):
            if remove_redundant:
                line = line[1:].strip()
            else:
                line = line[1:]
            if line == '':
                continue
            line_record = ["+ {}".format(line)]
            fix_lines_counter += 1
            line_tokenize = model.tokenizer.encode(line.strip(), add_special_tokens=False)
            line_format_list = []
            line_id_list = []
            local_cache = []
            for id_id in range(len(line_tokenize)):
                id = line_tokenize[id_id]
                prefix = fix_token_prefix[max(len(fix_token_prefix) - args.context_token_number, 0):]
                fix_token_prefix.append(id)
                ids_left = line_tokenize[id_id + 1:]
                if id_id == 0:
                    if not deal_with_one_hunk:
                        hunk_prefix = prefix
                    next_line_id = line_id + 1
                    while next_line_id < len(file_lines):
                        next_line = file_lines[next_line_id]
                        if next_line.startswith('+'):
                            next_line = next_line[1:]
                            if remove_redundant:
                                next_line = next_line.strip()
                            next_line_tokenize = model.tokenizer.encode(next_line, add_special_tokens=False)
                            local_cache.extend(next_line_tokenize)
                            next_line_id += 1
                        else:
                            break
                    if bottom_line <= line_id:
                        bottom_line = line_id + 1
                    while bottom_line < len(file_lines):
                        next_line = file_lines[bottom_line]
                        if next_line.startswith('-') or next_line.startswith('+'):
                            bottom_line += 1
                            continue
                        if remove_redundant:
                            next_line = next_line.strip()
                        if next_line == '':
                            bottom_line += 1
                            continue
                        next_line_tokenize = model.tokenizer.encode(next_line, add_special_tokens=False)
                        if suffix_id_length < args.context_token_number:
                            suffix.append(next_line_tokenize)
                            suffix_id_length += len(next_line_tokenize)
                        else:
                            break
                        bottom_line += 1
                if args.mask_style == 'MLM':
                    current_suffix = ids_left
                    current_suffix.extend(local_cache)
                    current_suffix.extend([item for sublist in suffix for item in sublist])
                else:
                    current_suffix = [item for sublist in suffix for item in sublist]
                format_token_ids = getFormatInput(args, prefix, current_suffix, args.context_token_number)
                line_format_list.append(format_token_ids)
                line_id_list.append(id)
            line_scores = getBatchedTokenScores(args, model, line_format_list, line_id_list, args.max_batch_size)
            neg_logs = [-math.log(score) for score in line_scores]
            cross_entropy = sum(neg_logs) / len(neg_logs)
            fix_lines_entropy += cross_entropy
            line_record.append(cross_entropy)
            line_entropies.append(line_record)
            if not deal_with_one_hunk:
                deal_with_one_hunk = True
            if deal_with_one_hunk:
                have_add = True
        else:
            if deal_with_one_hunk:
                deal_with_one_hunk = False
                if blank_token != '' and (not have_add or not have_del):
                    token_id_list = getFormatInput(args, hunk_prefix, [item for sublist in suffix for item in sublist],
                                                   args.context_token_number)
                    score = getBatchedTokenScores(args, model, [token_id_list],
                                                  [model.tokenizer.encode(blank_token, add_special_tokens=False)[0]],
                                                  args.max_batch_size)[0]
                    neg_logs = -math.log(score)
                    if not have_add:
                        fix_lines_counter += 1
                        fix_lines_entropy += neg_logs
                    if not have_del:
                        buggy_lines_counter += 1
                        buggy_lines_entropy += neg_logs
                have_del = False
                have_add = False
            if remove_redundant:
                line = line.strip()
            if line == '':
                continue
            line_record = [line]
            common_lines_counter += 1
            line_tokenize = model.tokenizer.encode(line.strip(), add_special_tokens=False)
            line_format_list = []
            line_id_list = []
            for id_id in range(len(line_tokenize)):
                id = line_tokenize[id_id]
                prefix = common_token_prefix[max(len(common_token_prefix) - args.context_token_number, 0):]
                common_token_prefix.append(id)
                buggy_token_prefix.append(id)
                fix_token_prefix.append(id)
                ids_left = line_tokenize[id_id + 1:]
                if id_id == 0:
                    if len(suffix) > 0:
                        suffix_id_length -= len(suffix[0])
                        suffix = suffix[1:]
                    if bottom_line <= line_id:
                        bottom_line = line_id + 1
                    while bottom_line < len(file_lines):
                        next_line = file_lines[bottom_line]
                        if next_line.startswith('+') or next_line.startswith('-'):
                            bottom_line += 1
                            continue
                        if remove_redundant:
                            next_line = next_line.strip()
                        if next_line == '':
                            bottom_line += 1
                            continue
                        next_line_tokenize = model.tokenizer.encode(next_line, add_special_tokens=False)
                        if suffix_id_length < args.context_token_number:
                            suffix.append(next_line_tokenize)
                            suffix_id_length += len(next_line_tokenize)
                        else:
                            break
                        bottom_line += 1
                if args.mask_style == 'MLM':
                    current_suffix = ids_left
                    current_suffix.extend([item for sublist in suffix for item in sublist])
                else:
                    current_suffix = [item for sublist in suffix for item in sublist]
                format_token_ids = getFormatInput(args, prefix, current_suffix, args.context_token_number)
                line_format_list.append(format_token_ids)
                line_id_list.append(id)
            line_scores = getBatchedTokenScores(args, model, line_format_list, line_id_list, args.max_batch_size)
            neg_logs = [-math.log(score) for score in line_scores]
            cross_entropy = sum(neg_logs) / len(neg_logs)
            common_lines_entropy += cross_entropy
            line_record.append(cross_entropy)
            line_entropies.append(line_record)
    if common_lines_counter > 0:
        common_lines_entropy = common_lines_entropy / common_lines_counter
    else:
        common_lines_entropy = 0.0
    if buggy_lines_counter > 0:
        buggy_lines_entropy = buggy_lines_entropy / buggy_lines_counter
    else:
        buggy_lines_entropy = 0.0
    if fix_lines_counter > 0:
        fix_lines_entropy = fix_lines_entropy / fix_lines_counter
    else:
        fix_lines_entropy = 0.0
    print(line_entropies[0][0])
    for line_entropie in line_entropies[1:]:
        print("line content: {}  cross entropy: {}".format(line_entropie[0], str(line_entropie[1])))
    print()
    return common_lines_counter, common_lines_entropy, \
           buggy_lines_counter, buggy_lines_entropy, \
           fix_lines_counter, fix_lines_entropy


def computeEntropyWithContextForChangeLines(args, buggy_file, fix_file, model, blank_token='\n', remove_redundant=True):
    """
    :param args: the parameters input at the beginning of program
    :param buggy_file: file path of buggy file
    :param fix_file: file path of fixed file
    :param model: instance of model
    :param blank_token: the token to replace empty code snippet
    :param remove_redundant: whether remove redundant information of buggy file and fixed file
    :return: cross entropy of modified code lines between buggy file and fixed file given context
    """
    tmp_patch_file = 'tmp_no_comment_patch_{}.java'.format(os.getpid())
    tmp_buggy_file = 'tmp_no_comment_buggy_{}.java'.format(os.getpid())
    try:
        line_entropies = [["file name: {}".format(buggy_file.split('/')[-1].split('-')[1])]]
    except:
        line_entropies = [["file name: {}".format(buggy_file.split('/')[-1])]]
    if remove_redundant:
        with open(tmp_buggy_file, 'w') as f:
            stdout = subprocess.run(
                ['java', '-jar', 'JavaFileExtractor.jar', '-path', buggy_file, '-customEdit', '-removeComments',
                 '-removePackage', '-removeImports'],
                stdout=subprocess.PIPE, env={'PATH': '/usr/bin/java/bin'}).stdout
            try:
                stdout = stdout.decode('utf-8')
            except UnicodeDecodeError:
                stdout = stdout.decode('iso-8859-1')
            buggy_file = tmp_buggy_file
            f.write(stdout)
            f.close()
        with open(tmp_patch_file, 'w') as f:
            stdout = subprocess.run(
                ['java', '-jar', 'JavaFileExtractor.jar', '-path', fix_file, '-customEdit', '-removeComments',
                 '-removePackage', '-removeImports'],
                stdout=subprocess.PIPE, env={'PATH': '/usr/bin/java/bin'}).stdout
            try:
                stdout = stdout.decode('utf-8')
            except UnicodeDecodeError:
                stdout = stdout.decode('iso-8859-1')
            fix_file = tmp_patch_file
            f.write(stdout)
            f.close()
    stdout = subprocess.run(['diff', '--unified=1000000', '-b', buggy_file, fix_file], stdout=subprocess.PIPE).stdout
    try:
        diff_output = stdout.decode('utf-8')
    except UnicodeDecodeError:
        diff_output = stdout.decode('iso-8859-1')
    if remove_redundant:
        os.remove(tmp_patch_file)
        os.remove(tmp_buggy_file)
    file_lines = diff_output.split('\n')[3:]
    buggy_lines_counter = 0
    fix_lines_counter = 0
    buggy_lines_entropy = 0.0
    fix_lines_entropy = 0.0
    buggy_token_prefix = []
    fix_token_prefix = []
    deal_with_one_hunk = False
    have_add = False
    have_del = False
    hunk_prefix = []
    suffix = []
    suffix_id_length = 0
    bottom_line = 0  # point to the line not been contained in the suffix
    for line_id in range(len(file_lines)):
        line = file_lines[line_id]
        if line.startswith('-'):
            if remove_redundant:
                line = line[1:].strip()
            else:
                line = line[1:]
            if line == '':
                continue
            line_record = ["- {}".format(line)]
            buggy_lines_counter += 1
            line_tokenize = model.tokenizer.encode(line, add_special_tokens=False)
            line_format_list = []
            line_id_list = []
            local_cache = []
            for id_id in range(len(line_tokenize)):
                id = line_tokenize[id_id]
                prefix = buggy_token_prefix[max(len(buggy_token_prefix) - args.context_token_number, 0):]
                buggy_token_prefix.append(id)
                ids_left = line_tokenize[id_id + 1:]
                if id_id == 0:
                    if not deal_with_one_hunk:
                        hunk_prefix = prefix
                    next_line_id = line_id + 1
                    while next_line_id < len(file_lines):
                        next_line = file_lines[next_line_id]
                        if next_line.startswith('-'):
                            next_line = next_line[1:]
                            if remove_redundant:
                                next_line = next_line.strip()
                            next_line_tokenize = model.tokenizer.encode(next_line, add_special_tokens=False)
                            local_cache.extend(next_line_tokenize)
                            next_line_id += 1
                        else:
                            break
                    if bottom_line <= line_id:
                        bottom_line = line_id + 1
                    while bottom_line < len(file_lines):
                        next_line = file_lines[bottom_line]
                        if next_line.startswith('+') or next_line.startswith('-'):
                            bottom_line += 1
                            continue
                        if remove_redundant:
                            next_line = next_line.strip()
                        if next_line == '':
                            bottom_line += 1
                            continue
                        next_line_tokenize = model.tokenizer.encode(next_line, add_special_tokens=False)
                        if suffix_id_length < args.context_token_number:
                            suffix.append(next_line_tokenize)
                            suffix_id_length += len(next_line_tokenize)
                        else:
                            break
                        bottom_line += 1
                if args.mask_style == 'MLM':
                    current_suffix = ids_left
                    current_suffix.extend(local_cache)
                    current_suffix.extend([item for sublist in suffix for item in sublist])
                else:
                    current_suffix = [item for sublist in suffix for item in sublist]
                format_token_ids = getFormatInput(args, prefix, current_suffix, args.context_token_number)
                line_format_list.append(format_token_ids)
                line_id_list.append(id)
            line_scores = getBatchedTokenScores(args, model, line_format_list, line_id_list, args.max_batch_size)
            neg_logs = [-math.log(score) for score in line_scores]
            cross_entropy = sum(neg_logs) / len(neg_logs)
            buggy_lines_entropy += cross_entropy
            line_record.append(cross_entropy)
            line_entropies.append(line_record)
            if not deal_with_one_hunk:
                deal_with_one_hunk = True
            if deal_with_one_hunk:
                have_del = True
        elif line.startswith('+'):
            if remove_redundant:
                line = line[1:].strip()
            else:
                line = line[1:]
            if line == '':
                continue
            line_record = ["+ {}".format(line)]
            fix_lines_counter += 1
            line_tokenize = model.tokenizer.encode(line.strip(), add_special_tokens=False)
            line_format_list = []
            line_id_list = []
            local_cache = []
            for id_id in range(len(line_tokenize)):
                id = line_tokenize[id_id]
                prefix = fix_token_prefix[max(len(fix_token_prefix) - args.context_token_number, 0):]
                fix_token_prefix.append(id)
                ids_left = line_tokenize[id_id + 1:]
                if id_id == 0:
                    if not deal_with_one_hunk:
                        hunk_prefix = prefix
                    next_line_id = line_id + 1
                    while next_line_id < len(file_lines):
                        next_line = file_lines[next_line_id]
                        if next_line.startswith('+'):
                            next_line = next_line[1:]
                            if remove_redundant:
                                next_line = next_line.strip()
                            next_line_tokenize = model.tokenizer.encode(next_line, add_special_tokens=False)
                            local_cache.extend(next_line_tokenize)
                            next_line_id += 1
                        else:
                            break
                    if bottom_line <= line_id:
                        bottom_line = line_id + 1
                    while bottom_line < len(file_lines):
                        next_line = file_lines[bottom_line]
                        if next_line.startswith('-') or next_line.startswith('+'):
                            bottom_line += 1
                            continue
                        if remove_redundant:
                            next_line = next_line.strip()
                        if next_line == '':
                            bottom_line += 1
                            continue
                        next_line_tokenize = model.tokenizer.encode(next_line, add_special_tokens=False)
                        if suffix_id_length < args.context_token_number:
                            suffix.append(next_line_tokenize)
                            suffix_id_length += len(next_line_tokenize)
                        else:
                            break
                        bottom_line += 1
                if args.mask_style == 'MLM':
                    current_suffix = ids_left
                    current_suffix.extend(local_cache)
                    current_suffix.extend([item for sublist in suffix for item in sublist])
                else:
                    current_suffix = [item for sublist in suffix for item in sublist]
                format_token_ids = getFormatInput(args, prefix, current_suffix, args.context_token_number)
                line_format_list.append(format_token_ids)
                line_id_list.append(id)
            line_scores = getBatchedTokenScores(args, model, line_format_list, line_id_list, args.max_batch_size)
            neg_logs = [-math.log(score) for score in line_scores]
            cross_entropy = sum(neg_logs) / len(neg_logs)
            fix_lines_entropy += cross_entropy
            line_record.append(cross_entropy)
            line_entropies.append(line_record)
            if not deal_with_one_hunk:
                deal_with_one_hunk = True
            if deal_with_one_hunk:
                have_add = True
        else:
            if deal_with_one_hunk:
                deal_with_one_hunk = False
                if blank_token != '' and (not have_add or not have_del):
                    token_id_list = getFormatInput(args, hunk_prefix, [item for sublist in suffix for item in sublist],
                                                   args.context_token_number)
                    score = getBatchedTokenScores(args, model, [token_id_list],
                                                  [model.tokenizer.encode(blank_token, add_special_tokens=False)[0]],
                                                  args.max_batch_size)[0]
                    neg_logs = -math.log(score)
                    if not have_add:
                        fix_lines_counter += 1
                        fix_lines_entropy += neg_logs
                    if not have_del:
                        buggy_lines_counter += 1
                        buggy_lines_entropy += neg_logs
                have_del = False
                have_add = False
            if remove_redundant:
                line = line.strip()
            if line == '':
                continue
            line_tokenize = model.tokenizer.encode(line.strip(), add_special_tokens=False)
            for id_id in range(len(line_tokenize)):
                id = line_tokenize[id_id]
                buggy_token_prefix.append(id)
                fix_token_prefix.append(id)
                if id_id == 0:
                    if len(suffix) > 0:
                        suffix_id_length -= len(suffix[0])
                        suffix = suffix[1:]
                    if bottom_line <= line_id:
                        bottom_line = line_id + 1
                    while bottom_line < len(file_lines):
                        next_line = file_lines[bottom_line]
                        if next_line.startswith('+') or next_line.startswith('-'):
                            bottom_line += 1
                            continue
                        if remove_redundant:
                            next_line = next_line.strip()
                        if next_line == '':
                            bottom_line += 1
                            continue
                        next_line_tokenize = model.tokenizer.encode(next_line, add_special_tokens=False)
                        if suffix_id_length < args.context_token_number:
                            suffix.append(next_line_tokenize)
                            suffix_id_length += len(next_line_tokenize)
                        else:
                            break
                        bottom_line += 1
    if buggy_lines_counter > 0:
        buggy_lines_entropy = buggy_lines_entropy / buggy_lines_counter
    else:
        buggy_lines_entropy = 0.0
    if fix_lines_counter > 0:
        fix_lines_entropy = fix_lines_entropy / fix_lines_counter
    else:
        fix_lines_entropy = 0.0
    print(line_entropies[0][0])
    for line_entropie in line_entropies[1:]:
        print("line content: {}  cross entropy: {}".format(line_entropie[0], str(line_entropie[1])))
    print()
    return buggy_lines_counter, buggy_lines_entropy, fix_lines_counter, fix_lines_entropy


def calculateCohensD(x, y):
    """
    :param x: sequence 1
    :param y: sequence 2
    :return: the cohen's d value of input sequences
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return abs((np.mean(x) - np.mean(y)) / np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof))


def calculateSNV(x, y):
    """
    :param x: sequence 1
    :param y: sequence 2
    :return: the SNV value of input sequences
    """
    wilcoxon = stats.ranksums(x, y).pvalue
    cohensd = calculateCohensD(x, y)
    if wilcoxon > 0.05:
        snv = 0
    elif sum(x)/len(x) <= sum(y)/len(y):
        snv = cohensd
    else:
        snv = -cohensd
    return snv
