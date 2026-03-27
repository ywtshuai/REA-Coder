from multiprocessing import Pool
from typing import List

from .data import InstanceData
from .executor import evaluate_case, EvalResult

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def _worker(args):
    return evaluate_case(**args)


def parallel_evaluate(tasks, workers=16, show_progress=True):
    with Pool(workers) as p:
        if show_progress and TQDM_AVAILABLE:
            results = list(tqdm(
                p.imap(_worker, tasks),
                total=len(tasks),
                desc="测试用例评估",
                unit="case"
            ))
        elif show_progress:
            results = []
            total = len(tasks)
            print(f"开始评估 {total} 个测试用例...")
            for i, result in enumerate(p.imap(_worker, tasks), 1):
                results.append(result)
                if i % 100 == 0 or i == total:
                    print(f"进度: {i}/{total} ({i/total*100:.1f}%)")
            return results
        else:
            results = p.map(_worker, tasks)
        return results


def eval_code(dataset: List[InstanceData], solutions: List[str],
              timeout: float = 10.0, workers: int = 64, show_progress: bool = True) -> List[tuple]:
    results = []
    total_problems = len(dataset)
    
    if show_progress and TQDM_AVAILABLE:
        iterator = tqdm(
            zip(dataset, solutions),
            total=total_problems,
            desc="题目评估",
            unit="题"
        )
    elif show_progress:
        iterator = enumerate(zip(dataset, solutions), 1)
        print(f"\n开始评估 {total_problems} 道题目...")
    else:
        iterator = zip(dataset, solutions)
    
    for idx_or_pair in iterator:
        if show_progress and not TQDM_AVAILABLE:
            idx, (instance, solution) = idx_or_pair
        else:
            instance, solution = idx_or_pair if TQDM_AVAILABLE else idx_or_pair
            idx = None
        
        if 'fn_name' in instance.test_cases:
            tasks = [
                {
                    "code": solution,
                    "input_data": test_input,
                    "expected": test_output,
                    "mode": "call",
                    "entry_func": instance.test_cases["fn_name"],
                    "timeout": timeout
                }
                for test_input, test_output in zip(
                    instance.test_cases["inputs"],
                    instance.test_cases["outputs"],
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
                    instance.test_cases["inputs"],
                    instance.test_cases["outputs"],
                )
            ]

        result = parallel_evaluate(tasks, workers=workers, show_progress=False)

        acc_num = 0
        for r in result:
            if r.status == 'AC':
                acc_num += 1
        acc_rate = acc_num / len(result) if len(result) > 0 else 0.0
        results.append((acc_rate, result))
        
        if show_progress and not TQDM_AVAILABLE and idx:
            if idx % 10 == 0 or idx == total_problems:
                passed = sum(1 for acc, _ in results if acc == 1.0)
                print(f"  [{idx}/{total_problems}] 当前 Pass@1: {passed}/{idx} ({passed/idx*100:.1f}%)")
    
    return results
