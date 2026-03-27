from multiprocessing import Pool
from typing import List, Optional

from apps_eval.data import InstanceData
from apps_eval.executor import evaluate_case, EvalResult

GLOBAL_POOL: Optional[Pool] = None


def init_pool(workers):
    global GLOBAL_POOL
    if GLOBAL_POOL is None:
        GLOBAL_POOL = Pool(workers)
    return GLOBAL_POOL


def _worker(args):
    return evaluate_case(**args)


def parallel_evaluate(tasks, workers=64):
    init_pool(workers)
    return GLOBAL_POOL.map(_worker, tasks)


def eval_code(dataset: List[InstanceData], solutions: List[str],
              timeout: float = 10.0, workers: int = 64) -> List[tuple[float, List[EvalResult]]]:
    results = []
    for instance, solution in zip(dataset, solutions):
        if 'fn_name' in instance.all_test_cases:
            tasks = [
                {
                    "code": solution,
                    "input_data": test_input,
                    "expected": test_output,
                    "mode": "call",
                    "entry_func": instance.all_test_cases["fn_name"],
                    "timeout": timeout
                }
                for test_input, test_output in zip(
                    instance.all_test_cases["inputs"],
                    instance.all_test_cases["outputs"],
                )
            ]
        else:
            tasks = [
                {
                    "code": solution,
                    "input_data": test_input,
                    "expected": test_output,
                    "mode": "stdio",
                    "timeout": timeout
                }
                for test_input, test_output in zip(
                    instance.all_test_cases["inputs"],
                    instance.all_test_cases["outputs"],
                )
            ]

        result = parallel_evaluate(tasks, workers=workers)

        acc_num = 0
        for r in result:
            if r.status == 'AC':
                acc_num += 1
        acc_rate = acc_num / len(result)
        results.append((acc_rate, result))
    return results


if __name__ == "__main__":
    dataset = [
        InstanceData(
            instance_id="1",
            problem_statement="Compute the sum of two integers.",
            starter_code="",
            all_test_cases={
                "fn_name": "add",
                "inputs": [(1, 2), (3, 4), (5, 6)],
                "outputs": [3, 7, 11]
            },
            solutions=[]
        )
    ]
    solutions = [
        '''
class Solution:
    def add(self, a, b):
        return a + b
'''
    ]
    results = eval_code(dataset, solutions)
    for acc_rate, eval_results in results:
        print(f"Accuracy Rate: {acc_rate}")
        print(eval_results)
