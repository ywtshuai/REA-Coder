import argparse
import concurrent
import json
import os
import random
import threading
from datetime import datetime

import numpy as np
from tqdm import tqdm

from apps_eval.data import get_data
from data import get_specification, to_code_prompt
from eval_code import eval_code
from model import load_model, generate_code, generate_code_api
from sanitize import sanitize_code, remove_code_blocks

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def alignment_rule(args, ori_specification, ori_code, public_test_cases, model, tokenizer,
                   optimization_list, cache_list):
    # 初始十条可选的规格优化维度，按需被模型挑选
    initial_mutation_instruction_list = [
        ["Specification Background",
         ["Let's provide a concise background for the provided programming specification, explaining the motivation, application context, or any domain-specific knowledge needed. Ensure the explanation is clear and accessible to developers without deep prior knowledge of the domain. This will help large language models understand it better. (Response constraints: Max 200 words, NO code)",
          ]],
        ["Specification Purpose",
         ["Let's concisely explain the purpose of the provided programming specification to improve clarity for large language models. This will help large language models understand it better. (Response constraints: Max 200 words, NO code)",
          ]],
        ["Key Concepts",
         ["Let's identify and explain the key concepts in the provided programming specification for better understanding by large language models. This will help large language models understand it better. (Response constraints: Max 200 words, NO code)",
          ]],
        ["Input Requirement",
         ["Let's analyze the input of the provided programming specification (e.g., data types and format) and list all constraints or boundaries for the inputs (e.g., value ranges, size limits, or specific conditions to be met). This will help large language models understand it better. (Response constraints: Max 200 words, NO code)",
          ]],
        ["Output Requirement",
         ["Let's clearly specify the format of the output, including data types, any required precision (e.g., number of decimal places), separators, or ordering rules. This will help large language models understand it better. (Response constraints: Max 200 words, NO code)",
          ]],
        ["Examples with Explanations",
         ["Provide three test case examples only if none are included in the programming specification. Then, offer a step-by-step explanation of the logic that produces the output. This will help large language models understand it better. (Response constraints: Max 200 words, NO code)",
          ]],
        ["Edge/Corner Cases",
         ["Let's retain the original test cases in the provided programming specification and generate three additional test cases to cover more edge/corner cases. This will help large language models understand it better. (Response constraints: Max 200 words, NO code)",
          ]],
        ["APIs",
         ["Let's specify external APIs or library functions that may be relevant for solving this task. Include only API/library names and their purpose. This will help large language models understand it better. (Response constraints: Max 200 words, NO code)",
          ]],
        ["Error Handling Requirements",
         ["Let's describe the expected behavior when the function encounters invalid inputs. Should it return a default value, throw an error, or handle the case differently? This will help large language models understand it better. (Response constraints: Max 200 words, NO code)",
          ]],
        ["Hints or Tips",
         ["Offer optional hints or implementation suggestions (e.g., specific algorithms or data structures to use). Keep the hints concise and high-level. This will help large language models understand it better. (Response constraints: Max 200 words, NO code)",
          ]]
    ]

    ori_code = ori_code.strip()
    # 复用规格->优化点的缓存，避免重复调用大模型
    if ori_specification in cache_list.keys():
        optimization_points = cache_list[ori_specification]
    else:
        # 提取代码语义形成“提升后的规格”，用于对比找出缺失点
        code_understanding_instruction = '''Analyze the given code exclusively and translate it as the lifted specification based on the format of defined DSL requirement. Ensure the response does not contain code. Limit the response to 500 words.\nFormat:\n'''
        code_understanding_instruction += '''(1) Problem Background: Describes the background and context of the code, providing the necessary background knowledge for understanding the code's functionality.\n'''
        code_understanding_instruction += '''(2) Functional Requirements: Summarizes the core functionality of the code, specifying its specific objectives or tasks, such as the problem being solved or the task being performed.\n'''
        code_understanding_instruction += '''(3) Input Requirements: Details the inputs required by the code, including input types, formats, and any constraints.\n'''
        code_understanding_instruction += '''(4) Output Requirements: Specifies the outputs of the code, including output types, formats, and any constraints.\n'''
        code_understanding_instruction += '''(5) Test Case Examples (Optional): Extracts the test cases included in the code, which are often used to verify code correctness and serve as usage examples.\n'''
        code_understanding_instruction += '''(6) External APIs (Optional): Lists any external APIs or library functions used by the code and describes their purpose and interactions.\n'''
        code_understanding_instruction += '''(7) Additional Explanation: Provides supplementary information, such as design intentions, potential limitations, or special considerations.\n'''
        code_understanding_prompt = f"#CODE:\n```python\n{ori_code}\n```\n\n#INSTRUCTION:\n{code_understanding_instruction}"
        if args.model_name in ['Qwen2.5-Coder-7B-Instruct', 'deepseek-coder-7b-instruct-v1.5']:
            code_understanding = generate_code(args, code_understanding_prompt, model, tokenizer, 512)
        elif args.model_name in ['gpt-4o-mini-2024-07-18', 'gemini-1.5-flash-002', 'deepseek-chat',
                                 'qwen3-coder-30b-a3b-instruct', 'gpt-5-mini-2025-08-07',
                                 'gemini-3-flash-preview-minimal', 'gemini-3-flash-preview']:
            code_understanding = generate_code_api(args, code_understanding_prompt, 512)
        else:
            raise ValueError("Unsupported model name")
        code_understanding = code_understanding.replace("\n\n", "\n")

        # 让模型指出当前规格缺失的维度，返回按重要度排序的列表字符串
        optimization_points_instruction = "Analyze the input programming specification and the lifted specification to identify misalignments or omissions. Select one or more ingredients to improve the input programming specification from the following ten options: ['Specification Background', 'Specification Purpose', 'Key Concepts', 'Input Requirement', 'Output Requirement', 'Examples with Explanations', 'Edge/Corner Cases', 'APIs', 'Error Handling Requirements']. Respond with a SORTED list by importance, e.g., ['Output Requirement', 'Specification Purpose', 'Examples with Explanations']. (Response constraints: Max 50 words, NO code)"
        optimization_points_prompt = f"{ori_specification}\n\n#INCORRECT GENERATED CODE:\n```python\n{ori_code}\n```\n\n#LIFTED SPECIFICATION OF INCORRECT GENERATED CODE:\n```plaintext\n{code_understanding}\n```\n\n#INSTRUCTION:\n{optimization_points_instruction}"
        if args.model_name in ['Qwen2.5-Coder-7B-Instruct', 'deepseek-coder-7b-instruct-v1.5']:
            optimization_points = generate_code(args, optimization_points_prompt, model, tokenizer, 64)
        elif args.model_name in ['gpt-4o-mini-2024-07-18', 'gemini-1.5-flash-002', 'deepseek-chat',
                                 'qwen3-coder-30b-a3b-instruct', 'gpt-5-mini-2025-08-07',
                                 'gemini-3-flash-preview-minimal', 'gemini-3-flash-preview']:
            optimization_points = generate_code_api(args, optimization_points_prompt, 64)
        else:
            raise ValueError("Unsupported model name")
        cache_list[ori_specification] = optimization_points

    optimization_points_list = []
    now_opl = 0
    # 根据返回的列表为每个维度打分排序，未命中的放到队尾
    for i_op in range(len(initial_mutation_instruction_list)):
        if initial_mutation_instruction_list[i_op][0] in optimization_list:
            continue
        if f"'{initial_mutation_instruction_list[i_op][0]}'" in optimization_points:
            now_opl += 1
            i = optimization_points.find(f"'{initial_mutation_instruction_list[i_op][0]}'")
            optimization_points_list.append([i, initial_mutation_instruction_list[i_op]])
        elif f'"{initial_mutation_instruction_list[i_op][0]}"' in optimization_points:
            now_opl += 1
            i = optimization_points.find(f'"{initial_mutation_instruction_list[i_op][0]}"')
            optimization_points_list.append([i, initial_mutation_instruction_list[i_op]])
        else:
            optimization_points_list.append(
                [1000000 + random.randint(0, 1000000), initial_mutation_instruction_list[i_op]])

    if len(optimization_points_list) == 0:
        optimization_points_list = [
            initial_mutation_instruction_list[random.randint(0, len(initial_mutation_instruction_list) - 1)]]
    else:
        optimization_points_list = [elem[1] for elem in
                                    sorted(optimization_points_list, key=lambda x: x[0], reverse=False)]

    # 将选中的维度生成新的子规格并拼到原始规格底部
    new_specification = ori_specification.replace('\n\n\n\n', '\n').replace('\n\n\n', '\n').strip()
    optimization_rule_prompt = f"{new_specification}\n\n#INSTRUCTION:\n{optimization_points_list[0][1][0]}"
    if args.model_name in ['Qwen2.5-Coder-7B-Instruct', 'deepseek-coder-7b-instruct-v1.5']:
        optimization_rule = generate_code(args, optimization_rule_prompt, model, tokenizer, 256)
    elif args.model_name in ['gpt-4o-mini-2024-07-18', 'gemini-1.5-flash-002', 'deepseek-chat',
                             'qwen3-coder-30b-a3b-instruct', 'gpt-5-mini-2025-08-07', 'gemini-3-flash-preview-minimal',
                             'gemini-3-flash-preview']:
        optimization_rule = generate_code_api(args, optimization_rule_prompt, 256)
    else:
        raise ValueError("Unsupported model name")
    optimization_rule = remove_code_blocks(optimization_rule).replace('\n\n', '\n')
    new_specification += f"\n\n#{optimization_points_list[0][0].upper()}:\n```plaintext\n{optimization_rule}\n```"

    # 重新组装代码生成提示，强制标准输入格式
    new_prompt = new_specification + f"\n\n#INSTRUCTION:\n"
    new_prompt = new_prompt.replace("Use Standard Input format.", "")
    new_prompt = new_prompt.replace("\n\n\n", "\n")
    new_prompt += "Use Standard Input format. "
    new_prompt += \
        "Please provide a self-contained Python script that solves the above programming specification in a markdown code block (without text and test cases):"
    new_prompt += "\n\n#CODE:\n```python\n\n```\n"

    # 调用本地/云端模型生成代码，并用公开测试用例即时评分
    if args.model_name in ['Qwen2.5-Coder-7B-Instruct', 'deepseek-coder-7b-instruct-v1.5']:
        new_code = generate_code(args, new_prompt, model, tokenizer, 1024)
    elif args.model_name in ['gpt-4o-mini-2024-07-18', 'gemini-1.5-flash-002', 'deepseek-chat',
                             'qwen3-coder-30b-a3b-instruct', 'gpt-5-mini-2025-08-07', 'gemini-3-flash-preview-minimal', 'gemini-3-flash-preview']:
        new_code = generate_code_api(args, new_prompt)
    else:
        raise ValueError("Unsupported model name")
    return new_specification, new_code, optimization_points_list[0][0], cache_list


def assemble_prompt(data_name, data_instance):
    # 针对不同数据集组装初始提示
    if data_name in ['apps', 'livecodebench']:
        prompt = data_instance.problem_statement
        starter_code = data_instance.starter_code
        if not starter_code:
            starter_code = None
        input_from, output_to, input_spec, output_spec, notes = None, None, None, None, None
    elif data_name == 'code_contests':
        prompt = data_instance.problem_statement
        starter_code = None
        input_from, output_to, input_spec, output_spec, notes = None, None, None, None, None
    elif data_name == 'xCodeEval':
        prompt = data_instance.problem_statement
        starter_code = None
        input_from, output_to, input_spec, output_spec, notes = None, None, None, None, None
    else:
        raise ValueError("Unsupported data name")
    return prompt, starter_code, input_from, output_to, input_spec, output_spec, notes


def save_when_pass(args, index, new_test_results, ori_code, ori_prompt, ori_test_result_all, problem_id, save_base_path,
                   test_data):
    print('*' * 40)
    for iter_n in range(args.max_iter):
        os.makedirs(f'{save_base_path}', exist_ok=True)
        open(f'{save_base_path}/{problem_id}_prompt_{iter_n}', 'w', encoding='utf-8').write(ori_prompt)
        open(f'{save_base_path}/{problem_id}_code_{iter_n}', 'w', encoding='utf-8').write(ori_code)
        open(f'{save_base_path}/{problem_id}_test_result_{iter_n}', 'w', encoding='utf-8').write(
            str(ori_test_result_all))

        new_test_results[iter_n][problem_id] = ori_test_result_all
        new_pass1 = round(list(new_test_results[iter_n].values()).count(1.0) / len(new_test_results[0]) * 100, 2)
        new_apr = round(np.average(list(new_test_results[iter_n].values())) * 100, 2)
        print(
            f">> ({args.model_name}, {args.data_name}-{index + 1}/{len(test_data)}, iter={iter_n})        Pass@1: {new_pass1}%, AvgPassRatio: {new_apr}%")
    print('*' * 40)


def generate_test_cases(args, model, ori_specification, public_test_cases, tokenizer):
    generated_test_cases_prompt = ori_specification
    if len(json.dumps(public_test_cases)) < 1024:
        generated_test_cases_prompt += f'\n\n#TEST CASES:\n```json\n{json.dumps(public_test_cases)}\n```'
    generated_test_cases_prompt += f"\n\n#INSTRUCTION:\nImplement a representative set of test cases for the above programming specification, ensure that the generated test cases are correct:"
    generated_test_cases_prompt += f"\n(1) To verify the fundamental functionality of the programming specification under normal conditions."
    generated_test_cases_prompt += f"\n(2) To evaluate the function's behavior under extreme or unusual conditions."
    generated_test_cases_prompt += f"\n(3) To assess the function's performance and scalability with large data samples."
    generated_test_cases_prompt += "\nPlease only provide additional test cases in a json format (Response constraints: Max 1000 words, NO code and text):"
    generated_test_cases_prompt += '\ne.g.,\n```json\n{"inputs": ["x1\\n", "x2\\n", "x3\\n", "x4\\n", "x5\\n", "x6\\n"], "outputs": ["y1\\n", "y2\\n", "y3\\n", y4\\n", "y5\\n", "y6\\n"]}\n```'
    if args.model_name in ['Qwen2.5-Coder-7B-Instruct', 'deepseek-coder-7b-instruct-v1.5']:
        generated_test_cases_ori = generate_code(args, generated_test_cases_prompt, model, tokenizer, 1024)
    elif args.model_name in ['gpt-4o-mini-2024-07-18', 'gemini-1.5-flash-002', 'deepseek-chat',
                             'qwen3-coder-30b-a3b-instruct', 'gpt-5-mini-2025-08-07', 'gemini-3-flash-preview-minimal', 'gemini-3-flash-preview']:
        generated_test_cases_ori = generate_code_api(args, generated_test_cases_prompt, 1024)
    else:
        raise ValueError("Unsupported model name")

    try:
        # 清洗模型输出并校验结构，失败则回退为空用例
        generated_test_cases_ori = sanitize_code(generated_test_cases_ori, ["```json", "```"])
        generated_test_cases_ori = json.loads(generated_test_cases_ori)

        generated_test_cases = {"inputs": [], "outputs": []}
        if "inputs" in generated_test_cases_ori.keys() and "outputs" in generated_test_cases_ori.keys():
            if type(generated_test_cases_ori["inputs"]) is list and type(generated_test_cases_ori["outputs"]) is list:
                if len(generated_test_cases_ori["inputs"]) == len(generated_test_cases_ori["outputs"]):
                    generated_test_cases['inputs'].extend(generated_test_cases_ori['inputs'])
                    generated_test_cases['outputs'].extend(generated_test_cases_ori['outputs'])
    except:
        generated_test_cases = {"inputs": [], "outputs": []}

    return generated_test_cases


def alignment():
    # 主迭代入口：按数据集逐题对齐规格与代码
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default='', type=str, required=True, help='apps, code_contests, xCodeEval')
    parser.add_argument("--model_name", default='', type=str, required=True,
                        help='Qwen2.5-Coder-7B-Instruct, deepseek-coder-7b-instruct-v1.5, gpt-4o-mini-2024-07-18, gemini-1.5-flash-002, deepseek-chat, qwen3-coder-30b-a3b-instruct, gpt-5-mini-2025-08-07, gemini-3-flash-preview-minimal, gemini-3-flash-preview')
    parser.add_argument("--save_dir", default='save', type=str, required=False)
    parser.add_argument("--max_iter", default=10, type=int)
    parser.add_argument("--debug", default=False, type=bool)
    args = parser.parse_args()

    test_base_path = f"./Results/{args.model_name}/{args.data_name}/test"
    save_base_path = f"./Results/{args.model_name}/{args.data_name}/{args.save_dir}"

    test_data = get_data(args.data_name)

    if args.model_name in ['Qwen2.5-Coder-7B-Instruct', 'deepseek-coder-7b-instruct-v1.5']:
        model, tokenizer = load_model(args.model_name)
    elif args.model_name in ['gpt-4o-mini-2024-07-18', 'gemini-1.5-flash-002', 'deepseek-chat',
                             'qwen3-coder-30b-a3b-instruct', 'gpt-5-mini-2025-08-07', 'gemini-3-flash-preview-minimal', 'gemini-3-flash-preview']:
        model, tokenizer = None, None
    else:
        raise ValueError("Unsupported model name")

    ori_test_results = {}
    new_test_results = [{} for _ in range(args.max_iter)]

    # 使用锁以保证并发写入共享结构的线程安全
    lock = threading.Lock()

    # ------------------ 并发化：首次生成/加载原始答案与测试结果 ------------------
    def _init_problem(index, data_instance):
        try:
            problem_id = data_instance.instance_id
            all_test_cases = data_instance.all_test_cases
            public_test_cases = data_instance.public_test_cases

            if not os.path.exists(f'{test_base_path}/{problem_id}_test_result'):
                # 针对不同数据集组装初始提示
                prompt, starter_code, input_from, output_to, input_spec, output_spec, notes = assemble_prompt(
                    args.data_name, data_instance)
                ori_specification = get_specification(args, all_test_cases, prompt, starter_code, input_from, output_to,
                                                      input_spec, output_spec, notes)
                ori_prompt = to_code_prompt(ori_specification, all_test_cases)

                # 首轮生成代码并用官方全量测试集评分
                if args.model_name in ['Qwen2.5-Coder-7B-Instruct', 'deepseek-coder-7b-instruct-v1.5']:
                    ori_code = generate_code(args, ori_prompt, model, tokenizer, 1024)
                elif args.model_name in ['gpt-4o-mini-2024-07-18', 'gemini-1.5-flash-002', 'deepseek-chat',
                                         'qwen3-coder-30b-a3b-instruct', 'gpt-5-mini-2025-08-07',
                                         'gemini-3-flash-preview-minimal', 'gemini-3-flash-preview']:
                    ori_code = generate_code_api(args, ori_prompt)
                else:
                    raise ValueError("Unsupported model name")
                if ori_code is None:
                    ori_code = ''
                ori_code = sanitize_code(ori_code, ["```python", "```"])
                _, ori_test_result = eval_code(args, all_test_cases, ori_code)

                # 缓存首轮提示/代码/结果，避免重复生成
                os.makedirs(f'{test_base_path}', exist_ok=True)
                open(f'{test_base_path}/{problem_id}_prompt', 'w', encoding='utf-8').write(ori_prompt)
                open(f'{test_base_path}/{problem_id}_code', 'w', encoding='utf-8').write(ori_code)
                open(f'{test_base_path}/{problem_id}_test_result', 'w', encoding='utf-8').write(str(ori_test_result))

            # 记录初始与各轮对齐结果，用于后续 Pass@1 统计
            with lock:
                ori_test_results[problem_id] = float(
                    open(f'{test_base_path}/{problem_id}_test_result', 'r', encoding='utf-8').read())
                for iter_n in range(args.max_iter):
                    new_test_results[iter_n][problem_id] = float(
                        open(f'{test_base_path}/{problem_id}_test_result', 'r', encoding='utf-8').read())
        except Exception as e:
            print(f"Error processing problem {index + 1}: {e}")

    # 并发执行初始化阶段
    max_workers = 32
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_init_problem, index, data_instance) for index, data_instance in
                   enumerate(test_data)]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='init'):
            pass

    ori_pass1 = round(list(ori_test_results.values()).count(1.0) / len(ori_test_results) * 100, 2)
    ori_apr = round(np.average(list(ori_test_results.values())) * 100, 2)
    print(f">> ({args.model_name}, {args.data_name}) Initial Pass@1: {ori_pass1}%, Initial AvgPassRatio: {ori_apr}%")

    # ------------------ 并发化：逐题对齐迭代 ------------------
    def _process_problem(index, data_instance):
        problem_id = data_instance.instance_id
        all_test_cases = data_instance.all_test_cases
        public_test_cases = data_instance.public_test_cases

        # 缓存命中则直接读取各轮结果并打印汇总
        all_files_exist = []
        for iter_n in range(args.max_iter):
            all_files_exist.append(os.path.exists(f'{save_base_path}/{problem_id}_prompt_{iter_n}'))
            all_files_exist.append(os.path.exists(f'{save_base_path}/{problem_id}_code_{iter_n}'))
            all_files_exist.append(os.path.exists(f'{save_base_path}/{problem_id}_test_result_{iter_n}'))
        if all(all_files_exist):
            print('*' * 40)
            for iter_n in range(args.max_iter):
                new_test_result_all = float(
                    open(f'{save_base_path}/{problem_id}_test_result_{iter_n}', 'r', encoding='utf-8').read())
                with lock:
                    new_test_results[iter_n][problem_id] = new_test_result_all
                    new_pass1 = round(
                        list(new_test_results[iter_n].values()).count(1.0) / len(new_test_results[0]) * 100, 2)
                    new_apr = round(np.average(list(new_test_results[iter_n].values())) * 100, 2)
                print(
                    f">> ({args.model_name}, {args.data_name}-{index + 1}/{len(test_data)}, iter={iter_n})        Pass@1: {new_pass1}%, AvgPassRatio: {new_apr}%")
            print('*' * 40)
            return

        # 针对不同数据集组装初始提示
        prompt, starter_code, input_from, output_to, input_spec, output_spec, notes = assemble_prompt(args.data_name,
                                                                                                      data_instance)
        ori_specification = get_specification(args, public_test_cases, prompt, starter_code, input_from, output_to,
                                              input_spec, output_spec, notes)
        ori_specification = f'#PROGRAMMING SPECIFICATION:\n```plaintext\n{ori_specification}\n```'

        if os.path.exists(f'{save_base_path}/{problem_id}_test_case'):
            generated_test_cases = json.loads(
                open(f'{save_base_path}/{problem_id}_test_case', 'r', encoding='utf-8').read())
        else:
            generated_test_cases = generate_test_cases(args, model, ori_specification, public_test_cases, tokenizer)

        os.makedirs(f'{save_base_path}', exist_ok=True)
        open(f'{save_base_path}/{problem_id}_test_case', 'w', encoding='utf-8').write(
            json.dumps(generated_test_cases))

        # 如果首轮已满分，直接复制到后续迭代并输出统计
        if os.path.exists(f'{test_base_path}/{problem_id}_prompt') and \
                os.path.exists(f'{test_base_path}/{problem_id}_code') and \
                os.path.exists(f'{test_base_path}/{problem_id}_test_result'):
            ori_prompt = open(f'{test_base_path}/{problem_id}_prompt', 'r', encoding='utf-8').read()
            ori_code = open(f'{test_base_path}/{problem_id}_code', 'r', encoding='utf-8').read()
            ori_test_result_all = float(
                open(f'{test_base_path}/{problem_id}_test_result', 'r', encoding='utf-8').read())
            ori_code = sanitize_code(ori_code, ["```python", "```"])

            _, ori_test_result = eval_code(args, public_test_cases, ori_code)
            if len(generated_test_cases["inputs"]):
                _, ori_test_result2 = eval_code(args, generated_test_cases, ori_code)
            else:
                ori_test_result2 = 0.0

            if ori_test_result == 1.0 and ori_test_result2 == 1.0:
                save_when_pass(args, index, new_test_results, ori_code, ori_prompt, ori_test_result_all, problem_id,
                               save_base_path, test_data)
                return
        else:
            return

        _, ori_test_result = eval_code(args, public_test_cases, ori_code)
        if len(generated_test_cases["inputs"]):
            _, ori_test_result2 = eval_code(args, generated_test_cases, ori_code)
        else:
            ori_test_result2 = 0.0

        # best_* 维护当前最优规格/代码/分数，按公共->补充->全量的层级比较
        best_specification, best_code, best_test_result, best_test_result2, best_test_result_all = ori_specification, ori_code, ori_test_result, ori_test_result2, ori_test_result_all
        best_optimization = 'None'
        optimization_list = []
        cache_list = {}
        for iter_n in range(args.max_iter):
            if os.path.exists(f'{save_base_path}/{problem_id}_prompt_{iter_n}') and \
                    os.path.exists(f'{save_base_path}/{problem_id}_code_{iter_n}') and \
                    os.path.exists(f'{save_base_path}/{problem_id}_test_result_{iter_n}') and \
                    os.path.exists(f'{save_base_path}/{problem_id}_optimization_{iter_n}'):
                new_optimization = open(f'{save_base_path}/{problem_id}_optimization_{iter_n}', 'r',
                                        encoding='utf-8').read()
                new_specification = open(f'{save_base_path}/{problem_id}_prompt_{iter_n}', 'r', encoding='utf-8').read()
                new_code = open(f'{save_base_path}/{problem_id}_code_{iter_n}', 'r', encoding='utf-8').read()
                new_test_result_all = float(
                    open(f'{save_base_path}/{problem_id}_test_result_{iter_n}', 'r', encoding='utf-8').read())

                optimization_list.append(new_optimization)
                new_code = sanitize_code(new_code, ["```python", "```"])
                _, new_test_result = eval_code(args, public_test_cases, new_code)
                if len(generated_test_cases["inputs"]):
                    _, new_test_result2 = eval_code(args, generated_test_cases, new_code)
                else:
                    new_test_result2 = 0.0

                print(
                    f"        >> Iter={iter_n} [id={problem_id}](hierarchical criteria): {round(best_test_result * 100, 2)}%({round(best_test_result2 * 100, 2)}%) ==> {round(new_test_result * 100, 2)}%({round(new_test_result2 * 100, 2)}%)")
                if (best_test_result < new_test_result) or (
                        best_test_result == new_test_result and best_test_result2 < new_test_result2):
                    best_specification = new_specification
                    best_code = new_code
                    best_test_result = new_test_result
                    best_test_result2 = new_test_result2
                    best_test_result_all = new_test_result_all
                with lock:
                    new_test_results[iter_n][problem_id] = best_test_result_all
            else:
                new_specification, new_code, new_optimization, cache_list = \
                    alignment_rule(args, best_specification, best_code, public_test_cases, model, tokenizer,
                                   optimization_list, cache_list)

                optimization_list.append(new_optimization)
                new_code = sanitize_code(new_code, ["```python", "```"])
                _, new_test_result = eval_code(args, public_test_cases, new_code)
                _, new_test_result_all = eval_code(args, all_test_cases, new_code)
                if len(generated_test_cases["inputs"]):
                    _, new_test_result2 = eval_code(args, generated_test_cases, new_code)
                else:
                    new_test_result2 = 0.0

                print(
                    f"        >> Iter={iter_n} [id={problem_id}](hierarchical criteria): {round(best_test_result * 100, 2)}%({round(best_test_result2 * 100, 2)}%) ==> {round(new_test_result * 100, 2)}%({round(new_test_result2 * 100, 2)}%)")
                if (best_test_result < new_test_result) or (
                        best_test_result == new_test_result and best_test_result2 < new_test_result2):
                    best_specification = new_specification
                    best_code = new_code
                    best_test_result = new_test_result
                    best_test_result2 = new_test_result2
                    best_test_result_all = new_test_result_all
                with lock:
                    new_test_results[iter_n][problem_id] = best_test_result_all

                os.makedirs(f'{save_base_path}/', exist_ok=True)
                if best_test_result == 1.0 and best_test_result2 == 1.0:
                    for temp_iter_n in range(iter_n, args.max_iter):
                        open(f'{save_base_path}/{problem_id}_prompt_{temp_iter_n}', 'w', encoding='utf-8').write(
                            best_specification)
                        open(f'{save_base_path}/{problem_id}_code_{temp_iter_n}', 'w', encoding='utf-8').write(
                            best_code)
                        open(f'{save_base_path}/{problem_id}_test_result_{temp_iter_n}', 'w', encoding='utf-8').write(
                            str(best_test_result_all))
                        open(f'{save_base_path}/{problem_id}_optimization_{temp_iter_n}', 'w', encoding='utf-8').write(
                            optimization_list[-1])
                        with lock:
                            new_test_results[temp_iter_n][problem_id] = best_test_result_all
                        print(
                            f"        >> Iter={temp_iter_n} [id={problem_id}](hierarchical criteria): {round(best_test_result * 100, 2)}%({round(best_test_result2 * 100, 2)}%) ==> {round(new_test_result * 100, 2)}%({round(new_test_result2 * 100, 2)}%)")
                    return
                else:
                    open(f'{save_base_path}/{problem_id}_prompt_{iter_n}', 'w', encoding='utf-8').write(
                        best_specification)
                    open(f'{save_base_path}/{problem_id}_code_{iter_n}', 'w', encoding='utf-8').write(best_code)
                    open(f'{save_base_path}/{problem_id}_test_result_{iter_n}', 'w', encoding='utf-8').write(
                        str(best_test_result_all))
                    open(f'{save_base_path}/{problem_id}_optimization_{iter_n}', 'w', encoding='utf-8').write(
                        optimization_list[-1])

        print('*' * 40)
        for iter_n in range(args.max_iter):
            with lock:
                new_pass1 = round(list(new_test_results[iter_n].values()).count(1.0) / len(new_test_results[0]) * 100,
                                  2)
                new_apr = round(np.average(list(new_test_results[iter_n].values())) * 100, 2)
            print(
                f">> ({args.model_name}, {args.data_name}-{index + 1}/{len(test_data)}, iter={iter_n})        Pass@1: {new_pass1}%, AvgPassRatio: {new_apr}%")
        print('*' * 40)

    # 并发执行逐题对齐阶段
    max_workers = 32
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_problem, index, data_instance) for index, data_instance in
                   enumerate(test_data)]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='align'):
            pass

    print('*' * 40)
    for iter_n in range(args.max_iter):
        with lock:
            new_pass1 = round(list(new_test_results[iter_n].values()).count(1.0) / len(new_test_results[0]) * 100, 2)
            new_apr = round(np.average(list(new_test_results[iter_n].values())) * 100, 2)
        print(
            f">> ({args.model_name}, iter={iter_n})        Pass@1: {new_pass1}%, AvgPassRatio: {new_apr}%")
    print('*' * 40)


if __name__ == '__main__':
    alignment()
    print(datetime.now())
