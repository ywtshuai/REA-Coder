import re


def sanitize_code(input_string, split_word):
    if input_string.find(split_word[0]) != -1:
        begin = input_string.find(split_word[0])
        if input_string.find(split_word[1], begin+1) != -1:
            pattern = re.compile(fr'{re.escape(split_word[0])}(.*?){re.escape(split_word[1])}', re.DOTALL)
            matches = re.findall(pattern, input_string)
            input_string = ''.join(matches)
    code_1 = []
    for i in input_string.split('\n'):
        if i[:7] != 'assert ' and i[:1] != '#':
            code_1.append(i)
    output_string = '\n'.join(code_1)

    return output_string


def remove_code_blocks(input_text):
    code_block_pattern = r"```python[\s\S]*?```"
    cleaned_text = re.sub(code_block_pattern, "", input_text)

    return cleaned_text


def get_content(input_string, split_word):
    if input_string.find(split_word[0]) != -1:
        begin = input_string.find(split_word[0])
        if input_string.rfind(split_word[1]) != -1:
            end = input_string.rfind(split_word[1])
            if begin == end:
                output_string = input_string[begin+len(split_word[0]):]
            else:
                output_string = input_string[begin+len(split_word[0]):end]
            return output_string

    while input_string.find('```python') != -1:
        start = input_string.find('```python')
        end = input_string.find('```', start+1)
        if end != -1:
            input_string = input_string[:start] + input_string[end+1:]
        else:
            input_string = input_string[:start]
    input_string = input_string.replace('```python', '')
    input_string = input_string.replace('```plaintext', '')
    input_string = input_string.replace('```text', '')
    input_string = input_string.replace('```', '')
    return input_string


def remove_unindented_lines(code, ok_starts):
    new_code = ""
    for line in code.splitlines():
        if any([line.startswith(t) for t in ok_starts]) or line.strip() == "":
            new_code += line + "\n"
            continue

        lspace = len(line) - len(line.lstrip())
        if lspace == 0:
            continue

        new_code += line + "\n"

    return new_code


def to_four_space_indents(old_code):
    new_code = ""
    for line in old_code.splitlines():
        lspace = len(line) - len(line.lstrip())
        if lspace == 3:
            new_code += " "
        new_code += line + "\n"
    return new_code
