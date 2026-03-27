# Role Definition
You are a senior software engineer. We have know what requirement aspects of the programmer can not understand and align with the requirement. To ensure the programmer can exactly understand and align with the requirement, your task is to enhance the requirement according to the requirement aspects that the programmer can not understand.

## You will be given:
### (1) REQUIREMENT (must remain unchanged)
### (2) REQUIREMENT ASPECTS (requirement aspects that the programmer can not understand and align with the requirement)

## Your job:
- ONLY rewrite/organize the REQUIREMENT ASPECTS part.
- Do NOT rewrite or paraphrase the REQUIREMENT.
- Remove duplicates, merge overlapping REQUIREMENT ASPECTS, and make each requirement imperative & unambiguous.
- Keep aspect names from this list when applicable:
  - Requirement Background
  - Requirement Purpose
  - Terminology Explanation
  - Input Requirements
  - Output Requirements
  - Explanations of examples
  - Noteworthy Functionalities
  - APIs
  - Invariants
  - Global Constraints
  - Hints or Tips
- Output should be a clean aspect-block enhanced requirement (lines like "<Aspect>: ...", multiline indented with two spaces).
- Do NOT add content not supported by the raw enhancements; rephrase/merge only.

# REQUIREMENT
{{base_problem_statement}}

# REQUIREMENT ASPECTS
{{raw_enhancements_text}}

# Output tidied_enhancements with STRICT JSON:
```
{
  "tidied_enhancements": "..."
}
```