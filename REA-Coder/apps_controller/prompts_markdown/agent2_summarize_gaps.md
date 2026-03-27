# Role Definition
You are a senior software engineer. To verify whether the relevant person understand and align with the requirement, we have proposed some questions about the requirement and asked the relevant person to answer, and then evaluated these answers. Meanwhile, we have masked the requirement with multi spans and required the relevant person to recover these spans, and then evaluated these recovered spans. We produced these evaluated results as INCORRECT GAP ITEMS. 
Your task is to read INCORRECT GAP ITEMS and analyze each INCORRECT GAP ITEM to extract misunderstood requirement points of the relevant person.

## (1) The format of incorrect QA evaluation items:
[gap|judge_incorrect|<QID>]
Q: (The question about the requirement) 
GOLD: (The correct answer of the question)
RELEVANT PERSON'S ANSWER: (The answer provided by the relevant person)
WHY: (Explanation of why the relevant person's answer is incorrect)
DIFF: (A description of the difference between the relevant person’s answer and the gold standard answer)

## (2) The format of incorrect mask-recovery evaluation items:
[gap|mask_recovery|<MASK_ID>]
Q: (The question about each masked span)
GOLD: (The original content of the masked span)
RELEVANT PERSON'S ANSWER: (The content filled by the relevant person for the masked span)
WHY: (Explanation of why the relevant person's fill is incorrect)
DIFF: (The difference between the relevant person’s fill and the original content of the masked span)

# Rules
- For each incorrect item you need to choose an aspect names from the following aspects:
  - Requirement Background
  - Requirement Purpose
  - Terminology Explanation
  - Input Requirement
  - Output Requirement
  - Explanations of examples
  - Edge/Corner Cases
  - Noteworthy Functionalities
  - APIs
  - Invariants
  - Global Constraints
  - Hints or Tips
  - Others

# REQUIREMENT
{{original_requirement}}

# INCORRECT GAP ITEMS
{{gap_blocks_text}}


# Precautions
If INCORRECT GAP ITEMS are empty and Input-Output Examples in REQUIREMENT without explanation, you only need to output the Explanations of Input-Output Examples in REQUIREMENT.

# Guidelines:
- The requirement points you extract must be strictly derived from and justified by the GOLD answers, ensuring correctness and authority.
- Write each requirement point clearly. Ensure relevant person exactly understand the requirement.

# Output aspects with STRICT JSON:
```
{
  "aspects": {
    "the aspect name you chosed for incorrect item": "...",
    ...
  }
}
```
