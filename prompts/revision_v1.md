SYSTEM:
Du beurteilst, wie wichtig die folgenden Kriterien bei der Auswahl eines
Lieferanten sind.
Grundlage sind ausschließlich die Aussagen, die dir zu einem Beschaffungsfall
vorgelegt werden. Jede Aussage ist dem Kriterium zugeordnet, auf das sie sich
bezieht.

Kriterien:
{{criteria_block}}

Aufgabe:
Einige Paarvergleiche zu diesem Fall wurden bereits beurteilt. Du beurteilst
nur die genannten Paare erneut, anhand der Aussagen. Du kannst beim bisherigen
Urteil bleiben oder es ändern. Für jedes Paar gibst du zuerst an, welches der
beiden Kriterien auf Basis der Aussagen wichtiger ist. Sind beide gleich
wichtig, antwortest du mit "{{indifference}}". Ist eines wichtiger, gibst du
zusätzlich an, wie viel wichtiger es ist.

Skala ({{scale_source}}):
{{scale_block}}

Der Skalenwert 1 entspricht der Antwort "{{indifference}}"; die Intensität ist
dann {{indifference_intensity}}. Wird ein Kriterium bevorzugt, liegt die Intensität
zwischen {{min_intensity}} und {{max_intensity}}.

Antwortformat:
Antworte ausschließlich mit einem JSON-Objekt in genau dieser Form und ohne
weiteren Text:

{
  "comparisons": [
    {"a": "<Kriterium links>", "b": "<Kriterium rechts>", {{justification_field}}"preference": "<Kriterium links>, <Kriterium rechts> oder <{{indifference}}>", "intensity": <{{indifference_intensity}} oder {{min_intensity}}-{{max_intensity}}>}
  ]
}

Regeln:
- Die Liste enthält genau {{comparison_count}} Einträge, in der Reihenfolge der genannten Paare.
- "a" und "b" übernimmst du unverändert als die in Klammern angegebenen
  Kennungen aus der Frage.
- "preference" ist entweder der Wert von "a", der Wert von "b" oder "{{indifference}}".
- "intensity" ist eine ganze Zahl.
{{justification_rule}}

USER:
{{scenario_intro}}

{{case_title}}

Aussagen aus dem Beschaffungsteam zum Bauteil {{component}}:
{{statements_block}}

Bitte beurteile diese {{comparison_count}} Paare anhand der Aussagen erneut:
{{revision_block}}
