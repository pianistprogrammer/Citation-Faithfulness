from collections import defaultdict
from math import sqrt

def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0: return (float("nan"), float("nan"))
    p = successes / total
    d = 1 + z*z/total
    center = (p + z*z/(2*total))/d
    radius = z*sqrt(p*(1-p)/total + z*z/(4*total*total))/d
    return center-radius, center+radius

def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows: groups[(row["model_id"], row["condition"])].append(row)
    output = []
    for (model, condition), items in sorted(groups.items()):
        available = [x for x in items if x["condition_available"]]
        recovered = [x for x in available if x["statement_recovered"]]
        positive = [x for x in recovered if x["adversarial_doc_cited"]]
        output.append({"model_id": model, "condition": condition, "target_count": len(items), "condition_available_count": len(available), "statement_recovery_count": len(recovered), "statement_recovery_rate": len(recovered)/len(available) if available else None, "post_rationalized_count": len(positive), "post_rationalization_rate_conditional": len(positive)/len(recovered) if recovered else None, "wilson_95": wilson(len(positive), len(recovered)) if recovered else None})
    return output
