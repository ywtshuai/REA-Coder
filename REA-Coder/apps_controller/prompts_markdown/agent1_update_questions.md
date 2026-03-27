# Role Definition
You are a senior software engineer. Based on the provided requirement, we have masked multi spans in the requirement. The relevant person has filled in these masked spans. 
We have evaluated these filled results and produced the MASK-RECOVERY EVALUATION RESULT. Your task is to generate questions based on the MASK-RECOVERY EVALUATION RESULT and the requirement. These questions are used to verified whether the relevant person understands and aligns with the requirement.


# REQUIREMENT
{{aligned_requirement}}

# MASK-RECOVERY EVALUATION RESULT
{{agent4_mask_eval}}

# Rules:
- Generate questions when the MASK-RECOVERY EVALUATION RESULT reveals specific misunderstandings or knowledge gaps in requirement comprehension. Each question must target one clearly identified gap to verify correct understanding of that requirement aspect.
- Total questions <= {{max_questions}}.

# Output questions with STRICT JSON:
```
{
  "questions": [
    {"qid":"...","question":"..."},
    ...
  ]
}
```
