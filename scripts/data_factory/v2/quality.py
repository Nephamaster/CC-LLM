"""Token-weighted checks shared by mixture and final training data."""

from collections import Counter, defaultdict


def sub_bucket(row):
    domain = row.get("domain")
    return "math_science" if domain in {"math", "scientific"} else domain


class MixtureMetrics:
    def __init__(self, config):
        self.config = config
        self.total = 0
        self.buckets = Counter()
        self.sources = defaultdict(Counter)
        self.subs = defaultdict(Counter)
        self.attributes = Counter()
        self.enhancement_tags = Counter()

    def add(self, row, tokens):
        bucket = row["candidate_bucket"]
        self.total += tokens
        self.buckets[bucket] += tokens
        self.sources[bucket][row["source"]] += tokens
        self.subs[bucket][sub_bucket(row)] += tokens
        tags = set(row.get("tags") or [])
        if "long_doc" in tags:
            self.attributes["long_document"] += tokens
        if "classical" in tags or "classical_candidate" in tags:
            self.attributes["classical_chinese"] += tokens
        if bucket == self.config.enhancement.bucket:
            self.enhancement_tags["classical" if tags & {"classical", "classical_candidate"} else "modern"] += tokens
            if "traditional" in tags:
                self.enhancement_tags["traditional"] += tokens

    def report(self):
        checks = {}
        for bucket in self.config.buckets:
            denominator = max(1, self.buckets[bucket.name])
            for source, weight in bucket.source_weights.items():
                actual = self.sources[bucket.name][source] / denominator
                checks[f"{bucket.name}/source/{source}"] = {
                    "target_fraction": weight, "actual_fraction": actual,
                    "passed": abs(actual - weight) <= 0.01,
                }
            for name, weight in bucket.sub_buckets.items():
                actual = self.subs[bucket.name][name] / denominator
                checks[f"{bucket.name}/domain/{name}"] = {
                    "target_fraction": weight, "actual_fraction": actual,
                    "passed": abs(actual - weight) <= 0.01,
                }
        for name, limits in self.config.attributes.items():
            actual = self.attributes[name] / max(1, self.total)
            checks[f"attribute/{name}"] = {
                **limits, "actual_fraction": actual,
                "passed": limits.get("min_fraction", 0) <= actual <= limits.get("max_fraction", 1),
            }
        denominator = max(1, self.buckets[self.config.enhancement.bucket])
        for name, limit in self.config.enhancement.constraints.items():
            if name == "single_source_max_fraction":
                actual = max(self.sources[self.config.enhancement.bucket].values(), default=0) / denominator
            else:
                actual = self.enhancement_tags[name.split("_")[0]] / denominator
            checks[f"enhancement/{name}"] = {
                "limit": limit, "actual_fraction": actual,
                "passed": actual <= limit + .01 if "max" in name else actual >= limit - .01,
            }
        return {"passed": all(v["passed"] for v in checks.values()), "checks": checks}
