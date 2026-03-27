# Role Definition
You are a senior software engineer. Your task is to fill in multi masked spans in the MASKED REQUIREMENT based on the CODE.  

# MASKED REQUIREMENT
{{masked_question}}

# CODE
{{generated_code}}

# Rules:
- Keep the recovered_statement coherent and meaningful.
- If uncertain for a span, write the most likely content based on context.

# Output fills and recovered_statement with STRICT JSON:
```
{
  "fills": [
    {"mask_id":"MASK_1","text":"..."},
    ...
  ],
  "recovered_statement": "the full recovered statement"
}
```