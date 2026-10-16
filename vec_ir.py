from dataclasses import dataclass, field


@dataclass
class VECClause:
    clause_id: str
    regime: str
    tree_index: int
    node_id: str
    feature: str
    threshold: float
    lower: float
    upper: float
    coverage: float
    branch_upper_bound: float


@dataclass
class VECIR:
    clauses: list[VECClause] = field(default_factory=list)

    def add_clause(self, clause: VECClause):
        self.clauses.append(clause)
    
    @classmethod
    def from_vec_json(cls, filename: str):
        import json

        with open(filename, "r") as f:
            data = json.load(f)

        ir = cls()

        coverage = data["coverage_level"]

        for item in data["branch_clauses"]:
            clause_id = (
                f"{item['regime']}-"
                f"{item['tree_index']}-"
                f"{item['node_id']}"
            )

            ir.add_clause(
                VECClause(
                    clause_id=clause_id,
                    regime=item["regime"],
                    tree_index=item["tree_index"],
                    node_id=item["node_id"],
                    feature=item["feature"],
                    threshold=item["threshold"],
                    lower=float(data["feature_intervals"][item["regime"]][item["feature"]]["lower_bound"]),
                    upper=float(data["feature_intervals"][item["regime"]][item["feature"]]["upper_bound"]),
                    coverage=coverage,
                    branch_upper_bound=item["firing_rate_upper_bound"],
                )
            )

        return ir
    def find_clause(self, clause_id: str):
        for clause in self.clauses:
            if clause.clause_id == clause_id:
                return clause
        return None

    def can_eliminate_branch(self, clause_id: str, threshold: float):
        clause = self.find_clause(clause_id)

        if clause is None:
            raise ValueError(f"Unknown VEC clause: {clause_id}")

        if clause.upper < threshold:
            return "ALWAYS_TRUE"

        if clause.lower >= threshold:
            return "ALWAYS_FALSE"

        return "KEEP"

    def transform_branch(self, clause_id: str, threshold: float):
        decision = self.can_eliminate_branch(clause_id, threshold)

        if decision == "ALWAYS_TRUE":
            return "if (1) { /* branch eliminated: always true */ }"

        if decision == "ALWAYS_FALSE":
            return "if (0) { /* branch eliminated: always false */ }"

        return "if (x < threshold) { /* branch retained */ }"
