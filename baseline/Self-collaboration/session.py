from roles import Analyst, Coder, Tester
from utils import find_method_name
import time
import re


class Session(object):
    def __init__(self, TEAM, ANALYST, PYTHON_DEVELOPER, TESTER, requirement, model='gpt-3.5-turbo', majority=1, max_tokens=512,
                                temperature=0.0, top_p=0.95, max_round=4, before_func=''):

        self.session_history = {}
        self.max_round = max_round
        self.before_func = before_func
        self.requirement = requirement
        self.analyst = Analyst(TEAM, ANALYST, requirement, model, majority, max_tokens, temperature, top_p)
        self.coder = Coder(TEAM, PYTHON_DEVELOPER, requirement, model, majority, max_tokens, temperature, top_p)
        self.tester = Tester(TEAM, TESTER, requirement, model, majority, max_tokens, temperature, top_p)
    
    def run_session(self):
        plan = self.analyst.analyze()
        report = plan
        is_init=True
        self.session_history["plan"] = plan
        code = ""
        final_code = ""

        for i in range(self.max_round):

            naivecode = self.coder.implement(report, is_init)
            method_name = find_method_name(naivecode)
            if method_name:
                code = naivecode
                
            if code.strip() == "":
                if i == 0:
                    code = "error"
                else:
                    code = self.session_history['Round_{}'.format(i-1)]["code"]
                break
            
            if i == self.max_round-1:
                self.session_history['Round_{}'.format(i)] = {"code": code}
                final_code = code
                break
            
            # 静态分析: Tester 直接分析代码，不执行
            tester_analysis = self.tester.test(code)
            
            # 解析 Tester 的分析报告，判断是否通过
            status_passed = self._parse_tester_status(tester_analysis)
            
            # 保存历史记录
            is_init = False
            self.session_history['Round_{}'.format(i)] = {
                "code": code, 
                "tester_analysis": tester_analysis,
                "status": "PASS" if status_passed else "FAIL"
            }

            if (plan == "error") or (code == "error"):
                code = "error"
                break
            
            # 提前终止: 如果 Tester 判定 PASS，立即跳出
            if status_passed:
                final_code = code
                break
            else:
                # FAIL: 将完整的分析报告作为反馈传给 Coder
                report = f"The static analysis report:\n{tester_analysis}"

        self.analyst.itf.clear_history()
        self.coder.itf.clear_history()
        self.tester.itf.clear_history()

        return final_code if final_code else code, self.session_history
    
    def _parse_tester_status(self, tester_output: str) -> bool:
        """
        解析 Tester 的输出，判断是否通过 (PASS)
        
        Args:
            tester_output: Tester 的完整输出文本
            
        Returns:
            True 如果包含 [Status]: PASS，否则 False
        """
        # 使用正则表达式提取 [Status]: 后的内容（忽略大小写）
        match = re.search(r'\[Status\]\s*:\s*(\w+)', tester_output, re.IGNORECASE)
        if match:
            status = match.group(1).strip().upper()
            return status == "PASS"
        return False

    def run_analyst_coder(self):
        plan = self.analyst.analyze()
        is_init=True
        self.session_history["plan"] = plan
        code = self.coder.implement(plan, is_init)

        if (plan == "error") or (code == "error"):
            code = "error"

        self.analyst.itf.clear_history()
        self.coder.itf.clear_history()
        self.tester.itf.clear_history()

        return code, self.session_history


    def run_coder_tester(self):
        report = ""
        is_init=True
        code = ""
        final_code = ""
        
        for i in range(self.max_round):

            naivecode = self.coder.implement(report, is_init)
            if find_method_name(naivecode):
                code = naivecode

            if code.strip() == "":
                if i == 0:
                    code = self.coder.implement(report, is_init=True)
                else:
                    code = self.session_history['Round_{}'.format(i-1)]["code"]
                break
            
            if i == self.max_round-1:
                self.session_history['Round_{}'.format(i)] = {"code": code}
                final_code = code
                break
                
            # 静态分析: Tester 直接分析代码，不执行
            tester_analysis = self.tester.test(code)
            
            # 解析 Tester 的分析报告，判断是否通过
            status_passed = self._parse_tester_status(tester_analysis)

            is_init = False
            self.session_history['Round_{}'.format(i)] = {
                "code": code, 
                "tester_analysis": tester_analysis,
                "status": "PASS" if status_passed else "FAIL"
            }

            if code == "error":
                break
            
            # 提前终止: 如果 Tester 判定 PASS，立即跳出
            if status_passed:
                final_code = code
                break
            else:
                # FAIL: 将完整的分析报告作为反馈传给 Coder
                report = f"The static analysis report:\n{tester_analysis}"

        self.analyst.itf.clear_history()
        self.coder.itf.clear_history()
        self.tester.itf.clear_history()

        return final_code if final_code else code, self.session_history

    def run_coder_only(self):
        plan = ""
        code = self.coder.implement(plan, is_init=True)
        self.coder.itf.clear_history()
        return code, self.session_history