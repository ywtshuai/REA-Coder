import json
import re
import ast
import time
import difflib
import copy


def code_truncate_regex(code):
    code_regex = r"```(.*?|)\n(?P<code>.*?)```"
    match = re.search(code_regex, code, re.DOTALL)
    code = match.group("code") if match else ""
    return code
    
def code_truncate(response):
    """
    从 LLM 响应中提取完整的 Python 代码
    
    提取优先级：
    1. 如果有 markdown 代码块（```python ... ```），提取其中的内容
    2. 否则，智能提取所有 Python 代码相关内容：
       - import/from 语句
       - 全局变量和常量定义
       - 类定义
       - 函数定义
       - if __name__ == "__main__" 入口
    """
    # 尝试提取 markdown 代码块
    code = code_truncate_regex(response)
    if code != "":
        return code
    
    # 没有 markdown 代码块，智能提取代码
    lines = response.split('\n')
    
    # 找到第一个函数/类定义的位置
    first_code_index = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('def ') or stripped.startswith('class '):
            first_code_index = i
            break
    
    if first_code_index == -1:
        # 没有找到函数或类定义，返回清理后的原响应
        return response.strip('```').strip()
    
    # 收集各类代码块
    code_lines = []
    
    # 1. 提取函数/类定义之前的代码（import、全局变量等）
    for i in range(first_code_index):
        line = lines[i]
        stripped = line.strip()
        
        # 跳过空行和注释
        if not stripped or stripped.startswith('#'):
            continue
        
        # 保留 import 语句
        if stripped.startswith('import ') or stripped.startswith('from '):
            code_lines.append(stripped)
            continue
        
        # 保留全局变量定义（简单的赋值语句）
        # 例如：MOD = 10**9 + 7, INF = float('inf')
        if '=' in stripped and not any(kw in stripped for kw in ['def ', 'class ', 'if ', 'for ', 'while ', 'with ']):
            code_lines.append(stripped)
            continue
    
    # 2. 提取函数和类定义（从第一个定义开始）
    i = first_code_index
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # 函数或类定义
        if stripped.startswith('def ') or stripped.startswith('class '):
            code_lines.append(line)
            i += 1
            
            # 提取函数/类体（所有缩进的行）
            while i < len(lines):
                next_line = lines[i]
                next_stripped = next_line.strip()
                
                # 空行也保留（函数内部的空行）
                if not next_stripped:
                    code_lines.append('')
                    i += 1
                    continue
                
                # 如果是缩进的行，属于当前函数/类
                if next_line.startswith((' ', '\t')):
                    code_lines.append(next_line)
                    i += 1
                # 如果遇到新的函数/类定义或 if __name__，跳出内层循环
                elif next_stripped.startswith(('def ', 'class ', 'if __name__')):
                    break
                else:
                    # 其他情况，跳出
                    i += 1
                    break
        
        # if __name__ == "__main__" 入口
        elif stripped.startswith('if __name__'):
            code_lines.append(line)
            i += 1
            
            # 提取入口代码块的内容
            while i < len(lines):
                next_line = lines[i]
                next_stripped = next_line.strip()
                
                # 空行也保留
                if not next_stripped:
                    code_lines.append('')
                    i += 1
                    continue
                
                # 缩进的行属于 if __name__ 块
                if next_line.startswith((' ', '\t')):
                    code_lines.append(next_line)
                    i += 1
                else:
                    # 非缩进行，跳出
                    break
        else:
            i += 1
    
    # 组装代码
    if not code_lines:
        return response.strip('```').strip()
    
    # 清理代码：移除开头和结尾的空行
    while code_lines and not code_lines[0].strip():
        code_lines.pop(0)
    while code_lines and not code_lines[-1].strip():
        code_lines.pop()
    
    return '\n'.join(code_lines)

def prompt_split_humaneval(prompt, mehotd_name):
    prompt = prompt.strip()
    prompt = prompt.replace("\r\n", "\n")
    before_func = prompt[:prompt.rfind("def ")]
    code = prompt[prompt.rfind("def "):]

    comment_start_1 = re.search("\"\"\"", code)
    comment_start_2 = re.search("\'\'\'", code)
    if comment_start_1:
        comment_start = comment_start_1.end()
    elif comment_start_2:
        comment_start = comment_start_2.end()


    example_start_1 = re.search("[eE]xample(:)?", code)
    example_start_2 = re.search("[fF]or [eE]xamble(:)?", code)
    example_start_3 = re.search(">>>", code)
    example_start_4 = re.search(mehotd_name+"\(.+\)", code[comment_start:])


    if example_start_1:
        comment = code[comment_start:example_start_1.start()]
        example = code[example_start_1.start():-4]
    elif example_start_2:
        comment = code[comment_start:example_start_2.start()]
        example = code[example_start_2.start():-4]
    elif example_start_3:
        comment = code[comment_start:example_start_3.start()]
        example = "Example:\n"+code[example_start_3.start():-4]
    elif example_start_4:
        comment = code[comment_start:example_start_4.start()+comment_start]
        example = "Example:\n"+code[example_start_4.start()+comment_start:-4]
    else:
        comment = code[comment_start:-4]
        example = ""
    comment = comment.strip().replace("\n", " ")
    comment = re.sub("\s+", " ", comment)

    example = re.sub("\n(\s)*","\n\t",example)
    test_case = "\t"+example.strip()
    signature = code[:code.index("\n")+1]

    return before_func, signature, comment, test_case

def build_test_method(test_list, test_imports, method_name):
    if test_imports:
        test_imports = "\n".join(test_imports)
        test_method = test_imports + "\n"
    else:
        test_method = ""
    test_method = "def check(" + method_name + "):\n"
    if len(test_list) == 0:
        return test_method + "\treturn True" + "\n"
    for test in test_list:
        test_method += '\t' + test + "\n"
    return test_method.strip("\n")

def find_method_name(code, lang="python"):
    try:
        parsed = ast.parse(code)
        function_defs = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]
        if function_defs:
            if len(function_defs) == 1:
                method_name = function_defs[0].name
            else:
                method_name = function_defs[-1].name if function_defs[-1].name != "main" else function_defs[-2].name
        else:
            method_name = None
    except:
        method_name = None

    return method_name


def code_split(func):
    '''
    Split code into signature, comment and function body
    '''
    func = func.replace("\r\n", "\n")
    before_func = func[:func.rfind("def ")]
    code = func[func.rfind("def "):]

    is_comment = False
    comments = []
    
    statements = code.split("\n")
    for s_idx, s in enumerate(statements):
        s = s.strip()
        if s.startswith("def"):
            signature = statements[:s_idx+1]
            method_name = s.split("def ")[1].split("(")[0]
            func_body_idx = s_idx+1
            tmp_statement = statements[func_body_idx].strip()
            if not tmp_statement.startswith("'''"):
                break
        elif s.startswith("'''") and not is_comment:
            is_comment = True

        elif is_comment:
            if s.startswith("'''"):
                is_comment = False
                func_body_idx = s_idx+1
                break
            comments.append(s)
    func_body = statements[func_body_idx:]
    return method_name, "\n".join(signature), "\n".join(comments), "\n".join(func_body), before_func

def construct_system_message(requirement, role, team=''):
    if team == '':
        system_message = "The requirement from users is: \n{'requirement':\n"  +  "'"+ requirement.replace('\n\n','\n').strip(".") + "'\n}\n\n" + role
    else:
        system_message = team + '\n '+ \
                    "The requirement from users is: \n{'requirement':\n"  +  "'"+ requirement.replace('\n\n','\n').strip(".") + "'\n}\n\n" + \
                    role
                
    return system_message
    