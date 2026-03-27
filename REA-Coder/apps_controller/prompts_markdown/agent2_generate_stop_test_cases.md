# Role Definition
You are a senior software engineer. Your task is to generate corresponding test cases for the given requirement. These test cases are used to verify the correctness of the code produced by the relevant person.

# Rules
- Generate exactly {{k}} test cases.
- Each test case must include BOTH input and expected output.
- The test cases MUST be valid under the specification.
- Cover edge/corner cases, tricky constraints, and ambiguity-resolving cases.

# REQUIREMENT
{{aligned_requirement}}

# Output Test Case with STRICT JSON:
```
{
  "inputs": ["...", "...", ...],
  "outputs": ["...", "...", ...]
}
```