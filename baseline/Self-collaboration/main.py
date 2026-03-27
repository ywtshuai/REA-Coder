# ============================================================
# 任务 3: 修改 main.py（核心逻辑注入）
# ============================================================

import os
import sys
import json
import time
import re
from typing import List, Dict, Any, Tuple
from multiprocessing import Pool, cpu_count
from datetime import datetime
from pathlib import Path


# ============================================================
# 数据集与模型选项
# ============================================================
DATASET_OPTIONS = ["code_contests_raw", "code_contests", "apps", "apps_eval", "xCodeEval", "livecodebench"]
# 常用模型：deepseek-chat, qwen3-coder-30b-a3b-instruct, gpt-5-mini-2025-08-07（可传任意模型名，输出目录按后缀区分）
MODEL_SUFFIX_MAP = {
    "gpt-5-mini-2025-08-07": "gpt5mini",
    "gemini-3-flash-preview": "gemini3flash",
}

_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _sanitize_fs_component(name: str, replacement: str = "_", max_len: int = 150) -> str:
    """
    将任意字符串清洗为跨平台可用的单个路径组件（文件名/目录名）。
    重点修复 Windows 下 <>:"/\\|?* 以及控制字符导致的 WinError 123。
    """
    if name is None:
        name = ""
    name = str(name)

    # Windows 非法字符 + ASCII 控制字符
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', replacement, name)

    # Windows 不允许末尾为空格或点
    name = name.rstrip(" .")

    # 收敛重复 replacement
    if replacement:
        rep_esc = re.escape(replacement)
        name = re.sub(rf"{rep_esc}{{2,}}", replacement, name)

    name = name.strip()
    if not name:
        name = "unnamed"

    # Windows 保留名（含扩展名形式，如 CON.txt 也不允许）
    base = name.split(".", 1)[0].upper()
    if base in _WINDOWS_RESERVED_NAMES:
        name = f"_{name}"

    if max_len and len(name) > max_len:
        name = name[:max_len].rstrip(" .")
        if not name:
            name = "unnamed"

    return name


def _model_to_suffix(model_name: str) -> str:
    if model_name in MODEL_SUFFIX_MAP:
        return MODEL_SUFFIX_MAP[model_name]
    if "qwen" in (model_name or "").lower():
        return "qwen"
    return "deepseek"


def _setup_model_env(model_name: str, api_key: str = None) -> None:
    """根据模型名设置 API base_url、api_key_env 与 MODEL_C；可选传入 --api-key。"""
    if model_name == "gpt-5-mini-2025-08-07":
        os.environ["MODEL_API_BASE_URL"] = "http://api.yesapikey.com/v1"
        os.environ["MODEL_API_KEY_ENV"] = "GPT5_MINI_API_KEY"
        os.environ["MODEL_C"] = model_name
        if api_key:
            os.environ["GPT5_MINI_API_KEY"] = api_key
        else:
            os.environ.setdefault("GPT5_MINI_API_KEY", "***")
    elif model_name == "gemini-3-flash-preview":
        os.environ["MODEL_API_BASE_URL"] = "http://api.yesapikey.com/v1"
        os.environ["MODEL_API_KEY_ENV"] = "GEMINI_FLASH_API_KEY"
        os.environ["MODEL_C"] = model_name
        if api_key:
            os.environ["GEMINI_FLASH_API_KEY"] = api_key
    elif "qwen" in (model_name or "").lower():
        os.environ["MODEL_API_BASE_URL"] = os.environ.get("MODEL_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        os.environ["MODEL_API_KEY_ENV"] = "DASHSCOPE_API_KEY"
        os.environ["MODEL_C"] = model_name
        if api_key:
            os.environ["DASHSCOPE_API_KEY"] = api_key
    else:
        os.environ["MODEL_API_BASE_URL"] = os.environ.get("MODEL_API_BASE_URL", "https://api.deepseek.com/v1")
        os.environ["MODEL_API_KEY_ENV"] = os.environ.get("MODEL_API_KEY_ENV", "DEEPSEEK_API_KEY")
        os.environ["MODEL_C"] = model_name
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key


# 全局配置：数据集默认值（未传 --dataset 时使用）
DATASET_NAME = 'xCodeEval'

# 导入依赖
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session import Session
from apps_eval.data import get_data, InstanceData
from apps_eval.parallel_runner import eval_code

# 导入角色定义
from roles.rule_descriptions_actc import TEAM, ANALYST, PYTHON_DEVELOPER, TESTER

# 导入全局 LLM（从 backend 中获取）
from core.backend import _GLOBAL_LLM


# ============================================================
# 详细日志系统
# ============================================================

class DetailedLogger:
    """为每个题目创建详细的日志记录"""
    
    def __init__(self, output_dir: str = "baseline_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
    def create_problem_dir(self, instance_id: str) -> Path:
        """为单个问题创建目录"""
        safe_instance_id = _sanitize_fs_component(instance_id)
        problem_dir = self.run_dir / safe_instance_id
        problem_dir.mkdir(exist_ok=True)
        return problem_dir
    
    def save_problem_info(self, problem_dir: Path, instance: InstanceData):
        """保存问题描述"""
        with open(problem_dir / "problem_statement.txt", "w", encoding="utf-8") as f:
            f.write(instance.problem_statement)
    
    def save_round_info(self, problem_dir: Path, round_num: int, 
                       code: str, report: str, role: str):
        """保存每一轮的信息"""
        round_dir = problem_dir / f"round_{round_num}"
        round_dir.mkdir(exist_ok=True)
        
        # 保存代码
        with open(round_dir / f"code_{role}.py", "w", encoding="utf-8") as f:
            f.write(code)
        
        # 保存报告
        with open(round_dir / f"report_{role}.txt", "w", encoding="utf-8") as f:
            f.write(report)
    
    def save_session_history(self, problem_dir: Path, session_history: Dict):
        """保存完整的 session 历史"""
        with open(problem_dir / "session_history.json", "w", encoding="utf-8") as f:
            json.dump(session_history, f, indent=2, ensure_ascii=False)
    
    def save_final_code(self, problem_dir: Path, code: str):
        """保存最终生成的代码"""
        with open(problem_dir / "final_solution.py", "w", encoding="utf-8") as f:
            f.write(code)
    
    def save_summary(self, summary: Dict):
        """保存总体摘要"""
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


# ============================================================
# 并行生成器
# ============================================================

def process_single_problem(args: Tuple[InstanceData, int, int, str, Path]) -> Dict:
    """
    处理单个问题的工作函数（用于并行）
    
    Args:
        args: (instance, idx, total, output_dir, run_dir) 元组
    
    Returns:
        包含结果的字典
    """
    instance, idx, total, output_dir, run_dir = args
    
    try:
        # 直接使用传入的 run_dir，不创建新的 logger
        # 为单个问题创建目录
        safe_instance_id = _sanitize_fs_component(instance.instance_id)
        problem_dir = run_dir / safe_instance_id
        problem_dir.mkdir(exist_ok=True)
        
        # 保存问题描述
        with open(problem_dir / "problem_statement.txt", "w", encoding="utf-8") as f:
            f.write(instance.problem_statement)
        
        print(f"[{idx}/{total}] 开始处理: {instance.instance_id}")
        
        # 初始化 Session（从环境变量读取模型名称）
        model_name = os.environ.get('MODEL_C', 'deepseek-chat')
        
        # 在运行 Session 前记录起始 token 数
        tokens_before = 0
        try:
            from core.backend import _get_llm
            llm = _get_llm()
            if hasattr(llm, 'total_tokens'):
                tokens_before = llm.total_tokens
        except Exception as e:
            pass  # 忽略错误，继续执行
        
        session = Session(
            TEAM=TEAM,
            ANALYST=ANALYST,
            PYTHON_DEVELOPER=PYTHON_DEVELOPER,
            TESTER=TESTER,
            requirement=instance.problem_statement,
            model=model_name,
            majority=1,
            max_tokens=8192,  
            temperature=0, 
            top_p=0.95,
            max_round=4,  
            before_func=''
        )
        
        # 运行 Session
        code, session_history = session.run_session()
        
        # 获取当前子进程的 token 使用量（计算差值）
        tokens_used = 0
        try:
            # 导入 backend 模块以访问该子进程的 _GLOBAL_LLM
            from core.backend import _get_llm
            llm = _get_llm()
            if hasattr(llm, 'total_tokens'):
                tokens_after = llm.total_tokens
                tokens_used = tokens_after - tokens_before  # 计算本次问题使用的 token 数
                print(f"[{idx}/{total}] 📊 {instance.instance_id} 使用了 {tokens_used} tokens")
        except Exception as e:
            print(f"⚠️  警告: [{idx}/{total}] {instance.instance_id} 获取 token 使用量失败: {e}")
        
        # 保存详细历史
        with open(problem_dir / "session_history.json", "w", encoding="utf-8") as f:
            json.dump(session_history, f, indent=2, ensure_ascii=False)
        
        # 保存每一轮的详细信息
        for round_key, round_data in session_history.items():
            if round_key.startswith('Round_'):
                round_num = int(round_key.split('_')[1])
                if 'code' in round_data:
                    # 创建轮次目录
                    round_dir = problem_dir / f"round_{round_num}"
                    round_dir.mkdir(exist_ok=True)
                    
                    # 保存代码
                    with open(round_dir / f"code_iteration.py", "w", encoding="utf-8") as f:
                        f.write(round_data['code'])
                    
                    # 保存静态分析报告（Tester 的反馈）
                    if 'tester_analysis' in round_data:
                        with open(round_dir / f"tester_analysis.txt", "w", encoding="utf-8") as f:
                            f.write(round_data['tester_analysis'])
                    
                    # 保存状态
                    if 'status' in round_data:
                        with open(round_dir / f"status.txt", "w", encoding="utf-8") as f:
                            f.write(round_data['status'])
        
        # 保存最终代码（添加必要的导入和入口点）
        final_code = code  # 初始化 final_code
        if code and code != "error":
            # 添加必要的导入
            imports_needed = []
            
            if 'import sys' not in final_code and ('sys.' in final_code or 'stdin' in final_code):
                imports_needed.append('import sys')
            
            if 'cmp_to_key' in final_code and 'from functools import' not in final_code:
                imports_needed.append('from functools import cmp_to_key')
            
            if 'math.' in final_code and 'import math' not in final_code:
                imports_needed.append('import math')
            
            if imports_needed:
                final_code = '\n'.join(imports_needed) + '\n\n' + final_code
            
            # 添加入口点
            if 'if __name__' not in final_code:
                if 'def solve()' in final_code:
                    final_code += '\n\nif __name__ == "__main__":\n    solve()'
                elif 'def main()' in final_code:
                    final_code += '\n\nif __name__ == "__main__":\n    main()'
            
            # 保存最终代码
            with open(problem_dir / "final_solution.py", "w", encoding="utf-8") as f:
                f.write(final_code)
            print(f"[{idx}/{total}] ✅ {instance.instance_id} 生成成功")
        else:
            final_code = "# Generation failed"
            print(f"[{idx}/{total}] ❌ {instance.instance_id} 生成失败")
        
        return {
            'instance_id': instance.instance_id,
            'code': final_code,  # 返回补全后的代码或失败标记
            'test_cases': instance.test_cases,
            'session_history': session_history,
            'problem_dir': str(problem_dir),
            'tokens_used': tokens_used
        }
        
    except Exception as e:
        print(f"[{idx}/{total}] ❌ {instance.instance_id} 异常: {e}")
        return {
            'instance_id': instance.instance_id,
            'code': f"# Exception: {e}",
            'test_cases': instance.test_cases,
            'session_history': {},
            'error': str(e),
            'tokens_used': 0
        }


# ============================================================
# 从 summary.json 重新评估（使用 code_contests 新 all_test_cases）
# ============================================================

def _run_reeval_from_summary(
    summary_path: str,
    workers: int = None,
    output_dir: str = None,
    limit: int = None
):
    """
    从老版本生成的 summary.json 读取代码，使用更新后的 code_contests 数据集
    （含 generated_cases 的 all_test_cases）重新评估。
    
    Args:
        summary_path: summary.json 文件路径，如 baseline_outputs/run_20260212_061720/summary.json
        workers: 并行进程数
        output_dir: 输出目录（None 则为 reeval_code_contests）
        limit: 限制评估的问题数量
    """
    import platform
    
    summary_file = Path(summary_path)
    if not summary_file.exists():
        print(f"❌ 错误: 找不到 summary 文件: {summary_path}")
        sys.exit(1)
    
    print(f"\n📂 加载 summary: {summary_file}")
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
    
    old_results = summary_data.get('results', [])
    if not old_results:
        print("❌ 错误: summary 中没有 results 数据")
        sys.exit(1)
    
    # 建立 instance_id -> code 映射
    code_by_id = {r['instance_id']: r['code'] for r in old_results}
    print(f"✅ 从 summary 读取 {len(code_by_id)} 个问题的代码")
    
    # 加载更新后的 code_contests 数据集（含完整 all_test_cases）
    print(f"\n📂 加载 code_contests 数据集（含 generated_cases 的 all_test_cases）...")
    dataset = get_data('code_contests')
    dataset_by_id = {inst.instance_id: inst for inst in dataset}
    
    # 匹配：仅评估 summary 中有的题目，且需在新数据集中存在
    eval_dataset = []
    eval_solutions = []
    all_results = []
    skipped = 0
    for instance_id, code in code_by_id.items():
        if instance_id not in dataset_by_id:
            skipped += 1
            continue
        inst = dataset_by_id[instance_id]
        eval_dataset.append(inst)
        eval_solutions.append(code)
        all_results.append({
            'instance_id': instance_id,
            'code': code,
            'problem_dir': '',
            'tokens_used': 0,
            'session_history': {}
        })
    
    if skipped > 0:
        print(f"⚠️  跳过 {skipped} 个在新数据集中不存在的题目")
    
    if limit is not None and limit > 0:
        eval_dataset = eval_dataset[:limit]
        eval_solutions = eval_solutions[:limit]
        all_results = all_results[:limit]
    
    print(f"✅ 将评估 {len(eval_dataset)} 个问题")
    
    # 输出目录
    if output_dir is None:
        output_dir = 'reeval_code_contests'
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_path / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    # 确定评估进程数
    if workers is None:
        workers = max(1, cpu_count() // 2)
    if platform.system() == 'Windows':
        eval_workers = workers
    else:
        eval_workers = min(workers , 60)
    
    print(f"\n[步骤 1/2] 使用 {eval_workers} 个进程并行评估...")
    eval_start_time = time.time()
    eval_results = eval_code(eval_dataset, eval_solutions, timeout=10.0, workers=eval_workers)
    eval_time = time.time() - eval_start_time
    
    print(f"✅ 评估完成！耗时: {eval_time:.2f} 秒")
    
    # 统计
    total_problems = len(eval_results)
    passed = sum(1 for acc_rate, _ in eval_results if acc_rate == 1.0)
    pass_at_1 = (passed / total_problems * 100) if total_problems > 0 else 0.0
    
    print(f"\n📊 重新评估结果")
    print("=" * 80)
    print(f"✅ Pass@1: {pass_at_1:.2f}% ({passed}/{total_problems})")
    print(f"⏱️  评估耗时: {eval_time:.2f} 秒")
    print("=" * 80)
    
    # 整理详细结果
    detailed_results = []
    for i, (result, (acc_rate, eval_result_list)) in enumerate(zip(all_results, eval_results)):
        detailed_results.append({
            'instance_id': result['instance_id'],
            'problem_dir': result.get('problem_dir', ''),
            'code': result['code'],
            'accuracy': acc_rate,
            'passed': acc_rate == 1.0,
            'tokens_used': result.get('tokens_used', 0),
            'test_results': [
                {
                    'status': r.status,
                    'time_cost': r.time_cost,
                    'stdin': str(r.stdin)[:100] if r.stdin else '',
                    'stdout': str(r.stdout)[:100] if r.stdout else '',
                    'expected': str(r.expected)[:100] if r.expected else ''
                }
                for r in eval_result_list
            ],
            'session_history': result.get('session_history', {})
        })
    
    summary = {
        'summary': {
            'method': 'Self-collaboration-Code-Generation',
            'dataset': 'code_contests',
            'mode': 'reeval_summary',
            'source_summary': str(summary_file),
            'pass_at_1': pass_at_1,
            'passed': passed,
            'total': total_problems,
            'time_cost': {'evaluation': eval_time},
            'config': {'eval_workers': eval_workers},
            'timestamp': datetime.now().isoformat()
        },
        'results': detailed_results
    }
    
    with open(run_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    output_file = "reeval_code_contests_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存到: {run_dir}")
    print(f"✅ 摘要已保存到: {output_file}")
    print("\n🎉 重新评估完成！")


# ============================================================
# 主流程
# ============================================================

def main(
    parallel: bool = True,
    workers: int = None,
    output_dir: str = None,
    limit: int = None,
    instance_ids: str = None,
    mode: str = 'all',
    resume_from: str = None,
    reeval_summary: str = None,
    model_name: str = None,
    dataset_name: str = None,
    api_key: str = None,
    eval_dataset: str = None,
    eval_output: str = None,
):
    """
    主函数
    
    Args:
        parallel: 是否使用并行生成（默认 True）
        workers: 并行进程数（默认为 CPU 核心数的一半）
        output_dir: 输出目录（None 则根据数据集+模型自动选择）
        limit: 限制处理的问题数量
        mode: 运行模式 ('all': 生成+评估, 'generate': 仅生成, 'evaluate': 仅评估)
        resume_from: 评估模式时，指定包含 generation_checkpoint.json 的 run 目录路径
        reeval_summary: 从 summary.json 读取代码，使用 code_contests 新数据集的 all_test_cases 重新评估
        model_name: 模型名称（默认从环境变量 MODEL_C 读取）
        dataset_name: 数据集名称（默认 DATASET_NAME）
        api_key: API Key（也可用环境变量）
        eval_dataset: 评估模式：覆盖评估数据集（如从 apps 生成 → 用 apps_eval 评估）
        eval_output: 评估模式：评估结果输出目录，会在此下创建 run_{timestamp}
    """
    print("=" * 80)
    print("Self-collaboration-Code-Generation Baseline")
    if reeval_summary:
        print("模式: 从 summary.json 重新评估（使用 code_contests 新 all_test_cases）")
    elif mode == 'generate':
        print("模式: 仅生成代码")
    elif mode == 'evaluate':
        print("模式: 仅评估代码")
    else:
        print("模式: 完整流程（生成+评估）")
    print("=" * 80)

    # =====================================================
    # 重新评估分支：从 summary.json 读取代码，用 code_contests 新数据集评估
    # =====================================================
    if reeval_summary:
        _run_reeval_from_summary(reeval_summary, workers, output_dir, limit)
        return

    if dataset_name is None:
        dataset_name = DATASET_NAME
    selected_instance_ids = None
    if instance_ids:
        selected_instance_ids = {
            item.strip() for item in instance_ids.split(',') if item.strip()
        }
    # evaluate 模式：若未显式传 --model，则以 checkpoint 中的 model_name 为准
    if mode != 'evaluate' and model_name is None:
        model_name = os.environ.get('MODEL_C', 'deepseek-chat')

    # 评估模式必须指定 resume_from
    if mode == 'evaluate' and not resume_from:
        print("❌ 评估模式需要指定 --resume-from（包含 generation_checkpoint.json 的 run 目录）")
        sys.exit(1)

    # 非 evaluate：设置模型环境并确定输出目录
    generation_dataset_name = dataset_name
    if mode != 'evaluate':
        _setup_model_env(model_name, api_key)
        # 根据数据集 + 模型自动选择输出目录（与 scot_baseline 一致：self_collaboration_outputs/{dataset}_{suffix}）
        if output_dir is None:
            suffix = _model_to_suffix(model_name)
            run_name = f"{dataset_name}_{suffix}"
            output_dir = os.path.join("self_collaboration_outputs", run_name)

    print(f"\n⚙️  配置信息:")
    print(f"  - 运行模式: {mode.upper()}")
    if mode != 'evaluate':
        print(f"  - 数据集: {dataset_name.upper()}")
        print(f"  - 模型: {model_name}")
        print(f"  - 并行模式: {'开启' if parallel else '关闭'}")
        print(f"  - 输出目录: {output_dir}")
    else:
        print(f"  - resume_from: {resume_from}")
        print(f"  - 并行模式: {'开启' if parallel else '关闭'}")

    # 加载数据集（generate/all 模式；evaluate 模式稍后加载）
    if mode != 'evaluate':
        print(f"\n[步骤 1/5] 加载 {dataset_name.upper()} 数据集...")
        if dataset_name == 'code_contests_raw':
            try:
                dataset = get_data('code_contests_raw')
            except FileNotFoundError:
                print("❌ code_contests 数据集文件未找到！")
                sys.exit(1)
        elif dataset_name == 'code_contests':
            dataset = get_data('code_contests')
        elif dataset_name == 'apps':
            try:
                dataset = get_data('apps')
            except FileNotFoundError:
                print("❌ APPS 数据集文件未找到！")
                print("💡 请先运行以下命令准备数据集:")
                print("   python prepare_apps_data.py")
                sys.exit(1)
        elif dataset_name == 'apps_eval':
            try:
                dataset = get_data('apps_eval')
            except FileNotFoundError:
                print("❌ APPS-Eval 数据集文件未找到！")
                print("💡 请先运行以下命令下载数据集:")
                print("   python prepare_apps_eval_data.py")
                sys.exit(1)
        elif dataset_name == 'xCodeEval':
            try:
                dataset = get_data('xCodeEval')
            except FileNotFoundError:
                print("❌ xCodeEval 数据集文件未找到！")
                print("💡 请确保 Datasets/xCodeEval.jsonl 存在")
                sys.exit(1)
        elif dataset_name == 'livecodebench':
            try:
                dataset = get_data('livecodebench')
            except FileNotFoundError:
                print("❌ livecodebench 数据集文件未找到！")
                print("💡 请确保 Datasets/livecodebench.jsonl 存在")
                sys.exit(1)
        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")

        if selected_instance_ids:
            dataset = [inst for inst in dataset if inst.instance_id in selected_instance_ids]
            print(f"按题号过滤后，共 {len(dataset)} 个问题")

        # 如果指定了限制，只取前N个问题
        if limit is not None and limit > 0:
            dataset = dataset[:limit]
            print(f"✅ 加载完成，共 {len(dataset)} 个问题（限制为前 {limit} 个）")
        else:
            print(f"✅ 加载完成，共 {len(dataset)} 个问题")

    # 创建/恢复日志记录器
    if mode == 'evaluate' and resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            print(f"❌ 错误: 指定的目录不存在: {resume_from}")
            sys.exit(1)
        if resume_path.name.startswith('run_'):
            logger = DetailedLogger.__new__(DetailedLogger)
            logger.output_dir = resume_path.parent
            logger.run_dir = resume_path
            print(f"✅ 使用已有运行目录: {logger.run_dir}")
        else:
            run_dirs = sorted(resume_path.glob('run_*'), key=lambda x: x.name, reverse=True)
            if not run_dirs:
                print(f"❌ 错误: 在 {resume_from} 中未找到 run_* 目录")
                sys.exit(1)
            logger = DetailedLogger.__new__(DetailedLogger)
            logger.output_dir = resume_path
            logger.run_dir = run_dirs[0]
            print(f"✅ 使用最新运行目录: {logger.run_dir}")
    else:
        logger = DetailedLogger(output_dir)
        print(f"✅ 输出目录: {logger.run_dir}")

    # 确定并行进程数
    if workers is None:
        workers = max(1, cpu_count() // 2)

    print(f"  - 工作进程数: {workers}")
    print(f"  - CPU 核心数: {cpu_count()}")

    # =====================================================
    # 评估模式：从已有结果文件加载
    # =====================================================
    if mode == 'evaluate':
        print(f"\n🔍 评估模式：从运行目录加载生成结果...")

        checkpoint_file = logger.run_dir / "generation_checkpoint.json"
        if not checkpoint_file.exists():
            print(f"❌ 错误: 找不到生成结果文件: {checkpoint_file}")
            print(f"💡 请先运行生成模式: python main.py --mode generate --dataset <name> --model <name>")
            sys.exit(1)

        print(f"📂 加载生成结果: {checkpoint_file}")
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)

        all_results = checkpoint_data['results']
        if selected_instance_ids:
            all_results = [r for r in all_results if r['instance_id'] in selected_instance_ids]
        generation_time = checkpoint_data.get('generation_time', 0.0)
        generation_dataset_name = checkpoint_data.get('dataset_name', generation_dataset_name)
        # evaluate 模式的模型：若未显式传 --model，以 checkpoint 的 model_name 为准
        if model_name is None:
            model_name = checkpoint_data.get('model_name', None) or os.environ.get('MODEL_C', 'deepseek-chat')
        _setup_model_env(model_name, api_key)
        # 评估时使用的数据集：--eval-dataset 覆盖 checkpoint 中的 dataset_name
        eval_dataset_name = eval_dataset or generation_dataset_name
        dataset_name = eval_dataset_name

        print(f"✅ 加载完成，共 {len(all_results)} 个问题的生成结果")
        print(f"   - generation_dataset: {generation_dataset_name}")
        print(f"   - eval_dataset: {eval_dataset_name}")
        print(f"   - model: {model_name}")
        print(f"   生成耗时: {generation_time:.2f} 秒（之前运行）")
        print(f"\n[步骤 1/5] 加载 {eval_dataset_name.upper()} 评估数据集...")
        dataset = get_data(eval_dataset_name)
        if selected_instance_ids:
            dataset = [inst for inst in dataset if inst.instance_id in selected_instance_ids]
        print(f"✅ 评估数据集加载完成，共 {len(dataset)} 个问题")

        start_time = time.time()

    # =====================================================
    # 生成模式或完整模式
    # =====================================================
    else:
        # 收集所有生成结果
        all_results = []
        start_time = time.time()

        # 主循环：生成代码
        print(f"\n[步骤 2/5] 开始生成代码...")
        print("=" * 80)

        if parallel and len(dataset) > 1:
            # 并行生成
            print(f"🚀 使用 {workers} 个进程并行生成...")

            args_list = [
                (instance, idx + 1, len(dataset), output_dir, logger.run_dir)
                for idx, instance in enumerate(dataset)
            ]

            with Pool(workers) as pool:
                all_results = pool.map(process_single_problem, args_list)

        else:
            # 顺序生成
            print("⏩ 顺序生成模式...")
            for idx, instance in enumerate(dataset):
                result = process_single_problem(
                    (instance, idx + 1, len(dataset), output_dir, logger.run_dir)
                )
                all_results.append(result)

        generation_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"✅ 代码生成完成！")
        print(f"⏱️  生成耗时: {generation_time:.2f} 秒")
        print(f"📈 平均每题: {generation_time / len(dataset):.2f} 秒")
        print("=" * 80)

        # 保存生成结果 checkpoint（供 evaluate 模式使用）
        print(f"\n💾 保存生成结果...")
        generation_checkpoint = {
            'results': all_results,
            'generation_time': generation_time,
            'dataset_name': dataset_name,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat()
        }
        checkpoint_file = logger.run_dir / "generation_checkpoint.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(generation_checkpoint, f, indent=2, ensure_ascii=False)
        print(f"✅ 生成结果已保存到: {checkpoint_file}")

        # 如果是仅生成模式，到此结束
        if mode == 'generate':
            print("\n" + "=" * 80)
            print("🎉 生成阶段完成！")
            print("=" * 80)
            print(f"✅ 详细结果已保存到: {logger.run_dir}")
            print(f"\n💡 下一步: 运行评估")
            print(f"   python main.py --mode evaluate --resume-from \"{logger.run_dir}\"")
            return

    # 最终评估
    print(f"\n[步骤 3/5] 最终评估所有生成结果...")
    print("=" * 80)
    
    eval_start_time = time.time()
    
    # 准备评估数据
    eval_dataset = []
    eval_solutions = []
    
    for result in all_results:
        # 从原始数据集中找到对应的实例
        instance = next((inst for inst in dataset if inst.instance_id == result['instance_id']), None)
        if instance:
            eval_dataset.append(instance)
            eval_solutions.append(result['code'])
    
    # 确定评估的并行进程数（与 scot_baseline 一致：直接使用用户指定的 workers）
    import platform
    eval_workers = workers
    print(f"🔍 使用 {eval_workers} 个进程并行评估... (平台: {platform.system()})")
    
    # 调用 eval_code 进行评估
    try:
        eval_results = eval_code(eval_dataset, eval_solutions, timeout=10.0, workers=eval_workers)
        eval_time = time.time() - eval_start_time
        
        print(f"✅ 评估完成！")
        print(f"⏱️  评估耗时: {eval_time:.2f} 秒")
        print(f"📈 平均每题: {eval_time / len(dataset):.2f} 秒")
        
    except Exception as e:
        print(f"\n❌ 评估过程出错: {e}")
        print(f"⚠️  已生成的代码已保存在: {logger.run_dir}")
        print(f"💡 提示: 可以使用恢复脚本完成评估:")
        print(f"   python recover_and_eval.py --run-dir {logger.run_dir}")
        raise  # 重新抛出异常
    
    # 💾 立即保存评估结果（防止后续统计阶段出错导致丢失）
    print(f"\n💾 保存评估中间结果...")
    try:
        eval_checkpoint = {
            'eval_results': [
                {
                    'instance_id': result['instance_id'],
                    'accuracy': acc_rate,
                    'passed': acc_rate == 1.0,
                    'test_count': len(eval_result_list),
                    'passed_tests': sum(1 for r in eval_result_list if r.status == 'AC'),
                    'test_statuses': [r.status for r in eval_result_list]
                }
                for result, (acc_rate, eval_result_list) in zip(all_results, eval_results)
            ],
            'eval_time': eval_time,
            'eval_workers': eval_workers,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = logger.run_dir / "eval_checkpoint.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(eval_checkpoint, f, indent=2, ensure_ascii=False)
        print(f"✅ 评估结果已保存到: {checkpoint_file}")
        print(f"   即使后续步骤出错，评估数据也不会丢失")
    except Exception as e:
        print(f"⚠️  警告: 保存评估结果失败: {e}")
        print(f"   将继续执行，但建议检查")
    
    # 统计结果
    print(f"\n[步骤 4/5] 统计结果...")
    print("=" * 80)
    
    total_problems = len(eval_results)
    passed = sum(1 for acc_rate, _ in eval_results if acc_rate == 1.0)
    pass_at_1 = (passed / total_problems * 100) if total_problems > 0 else 0.0
    
    total_time = time.time() - start_time
    
    # 从所有结果中汇总 token 使用量
    total_tokens = 0
    for result in all_results:
        total_tokens += result.get('tokens_used', 0)
    
    if total_tokens == 0:
        print("⚠️  注意: 未能统计到 Token 使用量")
    else:
        print(f"✅ 成功汇总所有子进程的 Token 使用量")
    
    # 打印最终结果
    print(f"\n📊 最终结果")
    print("=" * 80)
    print(f"✅ Pass@1: {pass_at_1:.2f}% ({passed}/{total_problems})")
    print(f"⏱️  总耗时: {total_time:.2f} 秒")
    if mode == 'evaluate':
        print(f"   - 代码评估: {eval_time:.2f} 秒")
        print(f"   - 代码生成: {generation_time:.2f} 秒（之前运行）")
    else:
        print(f"   - 代码生成: {generation_time:.2f} 秒 ({generation_time/total_time*100:.1f}%)")
        print(f"   - 代码评估: {eval_time:.2f} 秒 ({eval_time/total_time*100:.1f}%)")
    
    if total_tokens > 0:
        print(f"🔢 总 Token 使用量: {total_tokens:,}")
        print(f"📈 平均每题 Token: {total_tokens/total_problems:.0f}")
        print(f"💰 估算成本 (按 $0.27/1M tokens): ${total_tokens * 0.27 / 1_000_000:.4f}")
    else:
        print(f"🔢 总 Token 使用量: N/A {'(仅评估模式)' if mode == 'evaluate' else '(未能统计到 token 使用量)'}")
    
    print("=" * 80)
    
    # 保存结果到文件
    print(f"\n[步骤 5/5] 保存结果...")
    print("=" * 80)
    
    # 整理详细结果
    detailed_results = []
    for i, (result, (acc_rate, eval_result_list)) in enumerate(zip(all_results, eval_results)):
        detailed_results.append({
            'instance_id': result['instance_id'],
            'problem_dir': result.get('problem_dir', ''),
            'code': result['code'],
            'accuracy': acc_rate,
            'passed': acc_rate == 1.0,
            'tokens_used': result.get('tokens_used', 0),
            'test_results': [
                {
                    'status': r.status,
                    'time_cost': r.time_cost,
                    'stdin': str(r.stdin)[:100] if r.stdin else '',
                    'stdout': str(r.stdout)[:100] if r.stdout else '',
                    'expected': str(r.expected)[:100] if r.expected else ''
                }
                for r in eval_result_list
            ],
            'session_history': result.get('session_history', {})
        })
    
    # 保存到 run 目录
    summary = {
        'summary': {
            'method': 'Self-collaboration-Code-Generation',
            'dataset': dataset_name,
            'generation_dataset': generation_dataset_name,
            'model': model_name,
            'mode': mode,
            'pass_at_1': pass_at_1,
            'passed': passed,
            'total': total_problems,
            'time_cost': {
                'total': total_time,
                'generation': generation_time,
                'evaluation': eval_time
            },
            'token_usage': {
                'total': total_tokens if total_tokens > 0 else 'N/A',
                'average_per_problem': total_tokens / total_problems if (total_problems > 0 and total_tokens > 0) else 'N/A',
                'per_problem_details': [r.get('tokens_used', 0) for r in all_results]
            },
            'config': {
                'parallel': parallel,
                'workers': workers,
                'eval_workers': eval_workers
            },
            'timestamp': datetime.now().isoformat()
        },
        'results': detailed_results
    }
    
    logger.save_summary(summary)
    
    # 也保存到根目录（根据数据集+模型命名，与 scot_baseline 一致）
    suffix = _model_to_suffix(model_name)
    output_file = f"self_collaboration_results_{dataset_name}_{suffix}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 详细结果已保存到: {logger.run_dir}")
    print(f"✅ 摘要已保存到: {output_file}")
    print(f"✅ 每个问题的详细日志: {logger.run_dir}/<problem_id>/")
    
    # 生成可读的摘要报告
    report_file = logger.run_dir / "REPORT.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Self-collaboration-Code-Generation Baseline 运行报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据集(评估): {dataset_name.upper()}\n")
        if generation_dataset_name and generation_dataset_name != dataset_name:
            f.write(f"数据集(生成): {generation_dataset_name.upper()}\n")
        f.write(f"模型: {model_name}\n")
        f.write(f"运行模式: {mode.upper()}\n")
        f.write(f"题目数量: {total_problems}\n\n")
        f.write(f"配置信息:\n")
        f.write(f"  - 并行模式: {'开启' if parallel else '关闭'}\n")
        f.write(f"  - 生成进程数: {workers}\n")
        f.write(f"  - 评估进程数: {eval_workers}\n\n")
        f.write(f"结果统计:\n")
        f.write(f"  - Pass@1: {pass_at_1:.2f}% ({passed}/{total_problems})\n")
        f.write(f"  - 总耗时: {total_time:.2f} 秒\n")
        if mode == 'evaluate':
            f.write(f"  - 代码评估: {eval_time:.2f} 秒\n")
            f.write(f"  - 代码生成: {generation_time:.2f} 秒（之前运行）\n")
        else:
            f.write(f"  - 生成耗时: {generation_time:.2f} 秒\n")
            f.write(f"  - 评估耗时: {eval_time:.2f} 秒\n")
        if total_tokens > 0:
            f.write(f"  - 总 Token: {total_tokens:,}\n")
            f.write(f"  - 平均每题 Token: {total_tokens/total_problems:.0f}\n\n")
        else:
            f.write(f"  - 总 Token: N/A (未能统计到 token 使用量)\n\n")
        f.write(f"详细结果:\n")
        for result in detailed_results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            f.write(f"  {status} {result['instance_id']} (准确率: {result['accuracy']*100:.0f}%)\n")
    
    print(f"✅ 可读报告已保存到: {report_file}")

    # 评估模式且指定了 --eval-output 时，额外写入到该目录下的 run_{timestamp}
    if mode == 'evaluate' and eval_output:
        out_base = Path(eval_output)
        out_base.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_run_dir = out_base / f"run_{ts}"
        out_run_dir.mkdir(parents=True, exist_ok=True)
        (out_run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        report_out = out_run_dir / "REPORT.txt"
        with open(report_out, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Self-collaboration-Code-Generation Baseline 运行报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据集(评估): {dataset_name.upper()}\n")
            if generation_dataset_name and generation_dataset_name != dataset_name:
                f.write(f"数据集(生成): {generation_dataset_name.upper()}\n")
            f.write(f"模型: {model_name}\n")
            f.write(f"题目数量: {total_problems}\n\n")
            f.write(f"结果统计: Pass@1 {pass_at_1:.2f}% ({passed}/{total_problems})\n")
            f.write(f"详细结果:\n")
            for result in detailed_results:
                status = "✅ PASS" if result['passed'] else "❌ FAIL"
                f.write(f"  {status} {result['instance_id']} (准确率: {result['accuracy']*100:.0f}%)\n")
        print(f"✅ 评估结果已写入: {out_run_dir}")
    
    print("\n" + "=" * 80)
    print("🎉 所有任务完成！")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Self-collaboration-Code-Generation Baseline Runner',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model', type=str, default=None,
                       help='模型名称（默认: deepseek-chat，可选 qwen3-coder-30b-a3b-instruct、gpt-5-mini-2025-08-07 等）')
    parser.add_argument('--dataset', type=str, default=None,
                       choices=DATASET_OPTIONS,
                       help=f'数据集（默认: xCodeEval）。可选: {", ".join(DATASET_OPTIONS)}')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API Key（也可用环境变量 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / GPT5_MINI_API_KEY）')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='使用并行生成（默认开启）')
    parser.add_argument('--sequential', action='store_true',
                       help='使用顺序生成（覆盖 --parallel）')
    parser.add_argument('--workers', type=int, default=None,
                       help='并行进程数（默认为 CPU 核心数的一半）')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录（默认: 根据数据集+模型自动选择）')
    parser.add_argument('--limit', type=int, default=None,
                       help='限制处理的问题数量（用于测试，如: --limit 5）')
    parser.add_argument('--instance-ids', type=str, default=None,
                       help='仅处理指定题号，逗号分隔，如: --instance-ids 3708,3677,3832')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'generate', 'evaluate'],
                       help='运行模式:\n'
                            '  all      = 生成+评估（默认）\n'
                            '  generate = 仅生成代码，保存 checkpoint\n'
                            '  evaluate = 仅评估代码，需配合 --resume-from')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='评估模式：指定包含 generation_checkpoint.json 的 run 目录路径\n'
                            '示例: --resume-from self_collaboration_outputs/apps_deepseek/run_20260218_120000')
    parser.add_argument('--eval-dataset', type=str, default=None,
                       help='评估模式：覆盖评估数据集（如从 apps 生成 → 用 apps_eval 评估）')
    parser.add_argument('--eval-output', type=str, default=None,
                       help='评估模式：评估结果输出目录，会在此下创建 run_{timestamp}')
    parser.add_argument('--reeval-summary', type=str, default=None,
                       help='从 summary.json 读取代码，使用 code_contests 新数据集的 all_test_cases 重新评估\n'
                            '示例: --reeval-summary baseline_outputs/run_20260212_061720/summary.json')
    
    args = parser.parse_args()
    
    # 打印使用提示
    if args.mode != 'all':
        print("\n" + "=" * 80)
        if args.mode == 'generate':
            print("💡 提示: 生成完成后，使用以下命令评估:")
            print(f"   python main.py --mode evaluate --resume-from <run_dir>")
        elif args.mode == 'evaluate':
            if not args.resume_from:
                print("💡 提示: 建议使用 --resume-from 指定生成结果所在的 run 目录")
                print("   示例: --resume-from self_collaboration_outputs/apps_deepseek/run_20260218_120000")
        print("=" * 80 + "\n")
    
    # 如果指定了 sequential，则关闭并行
    parallel = not args.sequential if args.sequential else args.parallel
    
    main(
        parallel=parallel,
        workers=args.workers,
        output_dir=args.output_dir,
        limit=args.limit,
        instance_ids=args.instance_ids,
        mode=args.mode,
        resume_from=args.resume_from,
        reeval_summary=args.reeval_summary,
        model_name=args.model,
        dataset_name=args.dataset,
        api_key=args.api_key,
        eval_dataset=args.eval_dataset,
        eval_output=args.eval_output,
    )
