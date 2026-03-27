Given a structured explanatory text (such as a problem statement, specification, or technical description), produce a masked version by following the principles below.

[Objective]
The goal of masking is to hide high-information, semantics-critical content while preserving the overall structure, grammar, and readability of the text, so that a model must rely on contextual understanding and logical inference to recover the masked parts.

[What to Mask]
- Core objectives and task definitions:
Mask key verbs or phrases that describe what needs to be computed, optimized, determined, or produced. These typically define the main goal of the task.

- Essential definitions and rules:
Mask the critical conditions or constraints that determine when a definition applies or how a rule works, especially clauses that affect correctness or scope.

- Structural or property-related statements:
Mask phrases that describe fundamental properties of the system or structure involved, such as uniqueness, connectivity, directionality, hierarchy, or constraints.

- Semantics (not formatting) in input and output specifications:
Mask descriptions of what the output values mean or represent, even if the printing format itself is preserved.

- High-information elements in examples:
In example sections, mask key values, full lines, or results that directly reveal the solution, while keeping the example structure recognizable.

- Composite structural definitions:
Mask sentences or clauses that jointly define a structure and derive key quantities from it (e.g. spatial arrangement assumptions and resulting size formulas), even if they appear descriptive rather than imperative.
This includes phrases describing layout assumptions (e.g. alignment, ordering) and derived aggregate measures (e.g. total size, bounding dimensions).

- Symbol-to-meaning bindings:
Mask explanations that bind symbols or variables to their semantic meaning (e.g. “W is …”, “h_i represents …”), especially when they are required to compute the result.

- Indexing and exclusion logic:
Mask index ranges, quantifiers, and exclusion conditions that define which elements are included or omitted in each case (e.g. “for all j”, “except the i-th element”).

[What to Preserve]
- Sentence structure, paragraph structure, and ordering
- Function words and grammatical connectors (articles, prepositions, conjunctions)
- Background or narrative text that does not affect task logic

[Mask Format Rules]
Use placeholders of the form [MASK_1], [MASK_2], …[MASK_n]
Number masks in order of first appearance
Each mask corresponds to one coherent semantic unit
Do not merge unrelated information into one mask

[Hard constraints]
- You MUST create between 12 and 20 masked spans.
- You MUST mask at least:
  * 2 spans from story/background/context
  * 2 spans from core functionality (what is asked)
  * 2 spans from Input/Output sections (types/meaning, not only numeric ranges)
  * 2 spans from terminology/explanations sections (e.g., "Specification Background", "Key Concepts")
  * 2 spans from algorithm/process sections if present (e.g., "Steps Process")
- Disallowed spans:
  * spans that are only numbers/symbols (e.g., "10^18", "3 2") without words
  * masking only example outputs
- Each span should be 4–20 whitespace-separated tokens and contain at least one alphabetic word.

[EXAMPLE ORIGINAL TEXT]
<TEXT START>
You are given a broken clock. You know, that it is supposed to show time in 12- or 24-hours HH:MM format. In 12-hours format hours change from 1 to 12, while in 24-hours it changes from 0 to 23. In both formats minutes change from 0 to 59.

You are given a time in format HH:MM that is currently displayed on the broken clock. Your goal is to change minimum number of digits in order to make clocks display the correct time in the given format.

For example, if 00:99 is displayed, it is enough to replace the second 9 with 3 in order to get 00:39 that is a correct time in 24-hours format. However, to make 00:99 correct in 12-hours format, one has to change at least two digits. Additionally to the first change one can replace the second 0 with 1 and obtain 01:39.


-----Input-----

The first line of the input contains one integer 12 or 24, that denote 12-hours or 24-hours format respectively.

The second line contains the time in format HH:MM, that is currently displayed on the clock. First two characters stand for the hours, while next two show the minutes.


-----Output-----

The only line of the output should contain the time in format HH:MM that is a correct time in the given format. It should differ from the original in as few positions as possible. If there are many optimal solutions you can print any of them.


-----Examples-----
Input
24
17:30

Output
17:30

Input
12
17:30

Output
07:30

Input
24
99:99

Output
09:09
<TEXT END>

[EXAMPLE OUTPUT]
<TEXT START>
{{
  "masked_token_ratio: 0.29,
  "spans": [
    {{
      "mask_id":"MASK_1","original_text":"HH:MM",
      "mask_id":"MASK_2","original_text":"1 to 12",
      "mask_id":"MASK_3","original_text":"0 to 23",
      "mask_id":"MASK_4","original_text":"0 to 59",
      "mask_id":"MASK_5","original_text":"HH:MM",
      "mask_id":"MASK_6","original_text":"change minimum number of digits",
      "mask_id":"MASK_7","original_text":"in the given format",
      "mask_id":"MASK_8","original_text":"replace the second 9 with 3",
      "mask_id":"MASK_9","original_text":"in 24-hours format",
      "mask_id":"MASK_10","original_text":"in 12-hours format",
      "mask_id":"MASK_11","original_text":"two",
      "mask_id":"MASK_12","original_text":"replace the second 0 with 1",
      "mask_id":"MASK_13","original_text":"01:39",
      "mask_id":"MASK_14","original_text":"one integer 12 or 24",
      "mask_id":"MASK_15","original_text":"the time in format HH:MM",
      "mask_id":"MASK_16","original_text":"the hours",
      "mask_id":"MASK_17","original_text":"the minutes",
      "mask_id":"MASK_18","original_text":"in format HH:MM",
      "mask_id":"MASK_19","original_text":"in the given format",
      "mask_id":"MASK_20","original_text":"24",
      "mask_id":"MASK_21","original_text":"09:09"
    }}
  ],
  "masked_text":"
You are given a broken clock. You know, that it is supposed to show time in 12- or 24-hours [MASK_1] format. In 12-hours format hours change from [MASK_2], while in 24-hours it changes from [MASK_3]. In both formats minutes change from [MASK_4].

You are given a time in format [MASK_5] that is currently displayed on the broken clock. Your goal is to [MASK_6] in order to make clocks display the correct time [MASK_7].

For example, if 00:99 is displayed, it is enough to [MASK_8] in order to get 00:39 that is a correct time [MASK_9]. However, to make 00:99 correct [MASK_10], one has to change at least [MASK_11] digits. Additionally to the first change one can [MASK_12] and obtain [MASK_13].


-----Input-----

The first line of the input contains [MASK_14], that denote 12-hours or 24-hours format respectively.

The second line contains [MASK_15], that is currently displayed on the clock. First two characters stand for [MASK_16], while next two show [MASK_17].


-----Output-----

The only line of the output should contain the time [MASK_18] that is a correct time [MASK_19]. It should differ from the original in as few positions as possible. If there are many optimal solutions you can print any of them.


-----Examples-----
Input
[MASK_20]
17:30

Output
17:30

Input
12
17:30

Output
07:30

Input
24
99:99

Output
[MASK_21]"
}}
<TEXT END>

[TASK TEXT(that needs to be masked)]
<TEXT START>
{{requirement}}
<TEXT END>

# Final Reminder (CRITICAL)
You need to replace each occurrence of mask in the original text with a numbered [MASK_i], where i represents the i-th mask, and then output the modified text in the corresponding masked_text field of the required format.

Return STRICT JSON only:
{
  "masked_token_ratio": ...,
  "spans": [
    {"mask_id":"MASK_1","original_text":"..."},
    {"mask_id":"MASK_2","original_text":"..."},
    ...
    {"mask_id":"MASK_n","original_text":"..."}
  ],
  "masked_text": ...(must include [MASK_1]...[MASK_n])
}
