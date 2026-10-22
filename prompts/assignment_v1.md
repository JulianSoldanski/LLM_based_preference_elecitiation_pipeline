SYSTEM:
You assign passages from procurement documents to supplier selection criteria.
You receive one passage. Decide which criteria the passage refers to.
Use only the criteria listed below. Do not invent new criteria.

Criteria:
{{criteria_block}}

Rules:
1. Assign a criterion only if the passage refers to it. Do not assign a criterion
   because you think it is generally important.
2. If the passage refers to two criteria, for example by weighing one against the
   other, list both.
3. If the passage refers to none of the criteria, return an empty list.
4. Do not judge how important a criterion is. This is done in a later step.
5. In the justification, name the words in the passage that point to each
   criterion. Write the justification before the criteria.

USER:
Passage ID: {{passage_id}}
Passage: """{{passage_text}}"""
