import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_data(data_name):
    test_data = []
    for temp in open(f"./Datasets/{data_name}.jsonl", 'r', encoding='utf-8').readlines():
        test_data.append(json.loads(temp))

    return test_data


def get_specification(args, test_case, prompt, starter_code=None, input_from=None, output_to=None, input_spec=None,
                      output_spec=None, notes=None):
    if args.data_name in ['apps', 'code_contests', 'livecodebench']:
        _input = ""
        data = prompt
        _input += data
        if starter_code != None:
            data = starter_code
            data = "\n" + data
            _input += data
        else:
            pass

        data = test_case
        if data is None:
            _input += "\n\n"
        elif not data.get("fn_name"):
            _input += "\n\nUse Standard Input format. "
        else:
            _input += "\n\nUse Call-Based format. "
    elif args.data_name in ['xCodeEval']:
        _input = ""
        data = prompt
        _input += data

        if _input is not None and output_spec is not None:
            _input += f"\nInput Specification: {input_spec}"
            _input += f"\nOutput Specification: {output_spec}"
        if notes is not None:
            _input += f"\nNotes: {notes}"
        if input_from is not None and output_to is not None:
            _input += f"\nTake input from {input_from} and output to {output_to}."
    return _input


def to_code_prompt(specification, test_case_list):
    new_prompt = f"#QUESTION:\n{specification.strip()}\n\n#INSTRUCTION:\n"
    if test_case_list is None:
        new_prompt += ""
    elif not test_case_list.get("fn_name"):
        new_prompt = new_prompt.replace("Use Standard Input format.", "")
        new_prompt += "Use Standard Input format. "
    else:
        new_prompt = new_prompt.replace("Use Call-Based format.", "")
        new_prompt += "Use Call-Based format. "
    instruction_suffix = \
        "Please provide a self-contained Python script that solves the above programming specification in a markdown code block (without text and test cases):"
    new_prompt += instruction_suffix
    new_prompt += "\n\n#ANSWER:\n```python\n\n```\n"
    return new_prompt
