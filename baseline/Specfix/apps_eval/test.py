from apps_eval.parallel_runner import parallel_evaluate

# stdio，语法错误时返回信息
print("===== stdio mode =====")
tasks = [
    {
        "code": "print(int(input()) + 1)",
        "input_data": "1\n",
        "expected": "2\n",
        "mode": "stdio"
    }
]

results = parallel_evaluate(tasks)
print(results[0])

# call，函数调用
# 注意：call-based必须有一个Solution类，内含名为fn_name的函数(通常这部分限制条件在starter_code中)
print("===== call mode =====")
tasks = [
    {
        "code": '''
class Solution:
    def add(self, x, y):
        return x + y
''',
        "input_data": (1, 2),
        "expected": 3,
        "mode": "call",
        "entry_func": "add"
    }
]
results = parallel_evaluate(tasks)
print(results[0])
