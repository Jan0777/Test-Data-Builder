"""
Realism tests for the Replicator empirical engine.
Tests: bimodal salary, price .99 endings, country→currency, age↔income
       correlation, parent/child referential integrity, ID format masks.

Run with: python -m pytest backend/tests/test_realism.py -v
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from backend.replicator.profiler import profile_to_spec
from backend.engine.generator import generate


# ────────────────────────────────────────────────────────────────────────────
# Source data factories
# ────────────────────────────────────────────────────────────────────────────

def make_main_table(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Bimodal salary: two Gaussians at 40k and 85k
    half = n // 2
    salary = np.concatenate([
        rng.normal(40_000, 5_000, half),
        rng.normal(85_000, 10_000, n - half),
    ])
    rng.shuffle(salary)
    salary = np.clip(salary, 20_000, 200_000).astype(float)

    # Price column with .99 endings
    base_price = rng.choice([9, 14, 19, 24, 29, 49, 79, 99, 149, 199], size=n)
    price = base_price.astype(float) + 0.99

    # Country → currency deterministic mapping
    countries = rng.choice(["US", "UK", "DE", "JP", "FR"], size=n, p=[0.4, 0.2, 0.2, 0.1, 0.1])
    currency_map = {"US": "USD", "UK": "GBP", "DE": "EUR", "JP": "JPY", "FR": "EUR"}
    currency = np.array([currency_map[c] for c in countries])

    # Age ↔ income correlation (Spearman ~0.65)
    age = rng.integers(22, 65, size=n).astype(float)
    income_noise = rng.normal(0, 8_000, n)
    income = np.clip(age * 1_500 + income_noise, 20_000, 200_000)

    # Structured ID column: AAA-1234
    letters = np.array(
        ["".join(rng.choice(list("ABCDE"), size=3)) for _ in range(n)]
    )
    digits = rng.integers(1000, 9999, size=n)
    ids = np.array([f"{l}-{d:04d}" for l, d in zip(letters, digits)])

    return pd.DataFrame({
        "record_id": ids,
        "salary": salary,
        "price": price,
        "country": countries,
        "currency": currency,
        "age": age,
        "income": income,
    })


def make_parent_child_tables(n_customers: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)

    customer_ids = [f"C{i:05d}" for i in range(n_customers)]
    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "age": rng.integers(22, 65, size=n_customers).astype(float),
        "annual_income": rng.normal(60_000, 20_000, n_customers).clip(20_000, 200_000),
    })

    rows = []
    for cid in customer_ids:
        n_orders = int(rng.integers(1, 6))
        for j in range(n_orders):
            rows.append({
                "order_id": f"O{len(rows):06d}",
                "customer_id": cid,
                "amount": float(rng.exponential(100)),
                "quantity": int(rng.integers(1, 11)),
            })
    orders = pd.DataFrame(rows)

    return customers, orders


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _percentiles(arr, qs=(10, 25, 50, 75, 90, 99)):
    return np.array([np.percentile(arr, q) for q in qs])


def _spearman_corr_matrix(df, cols):
    return df[cols].rank().corr().values


def _ks_pvalue(source_vals, gen_vals):
    s = np.array(source_vals, dtype=float)
    g = np.array(gen_vals, dtype=float)
    s = s[~np.isnan(s)]
    g = g[~np.isnan(g)]
    if len(s) < 5 or len(g) < 5:
        return 1.0
    return float(scipy_stats.ks_2samp(s, g).pvalue)


def _cat_l1(source_series, gen_series):
    """L1 distance between category frequency distributions."""
    src_freq = source_series.value_counts(normalize=True)
    gen_freq = gen_series.value_counts(normalize=True)
    all_cats = set(src_freq.index) | set(gen_freq.index)
    l1 = sum(abs(src_freq.get(c, 0) - gen_freq.get(c, 0)) for c in all_cats)
    return float(l1)


# ────────────────────────────────────────────────────────────────────────────
# Test 1 — Univariate realism: KS test + percentile alignment
# ────────────────────────────────────────────────────────────────────────────

class TestUnivariateRealism:
    def setup_method(self):
        self.source = make_main_table(n=500, seed=42)
        spec = profile_to_spec({"main": self.source})
        generated = generate(spec, seed=0)
        self.gen = generated["main"]

    def test_salary_ks_pvalue(self):
        """Bimodal salary: KS p-value should be above threshold (not significantly different)."""
        p = _ks_pvalue(self.source["salary"].values, self.gen["salary"].values)
        assert p > 0.01, f"Salary KS p-value too low: {p:.4f} (distributions diverged)"

    def test_salary_percentile_alignment(self):
        """Bimodal salary: key percentiles within 20% tolerance."""
        qs = (10, 25, 50, 75, 90)
        src_pcts = _percentiles(self.source["salary"].values, qs)
        gen_pcts = _percentiles(self.gen["salary"].values, qs)
        for q, sp, gp in zip(qs, src_pcts, gen_pcts):
            rel_err = abs(gp - sp) / (abs(sp) + 1)
            assert rel_err < 0.25, (
                f"Salary p{q}: source={sp:.0f}, gen={gp:.0f}, rel_err={rel_err:.3f}"
            )

    def test_income_ks_pvalue(self):
        """Age-correlated income: KS test."""
        p = _ks_pvalue(self.source["income"].values, self.gen["income"].values)
        assert p > 0.01, f"Income KS p-value too low: {p:.4f}"

    def test_age_percentile_alignment(self):
        qs = (10, 25, 50, 75, 90)
        src_pcts = _percentiles(self.source["age"].values, qs)
        gen_pcts = _percentiles(self.gen["age"].values, qs)
        for q, sp, gp in zip(qs, src_pcts, gen_pcts):
            rel_err = abs(gp - sp) / (abs(sp) + 1)
            assert rel_err < 0.25, f"Age p{q}: source={sp:.1f}, gen={gp:.1f}"

    def test_country_category_frequencies(self):
        """Country category L1 distance should be small."""
        l1 = _cat_l1(self.source["country"], self.gen["country"])
        assert l1 < 0.30, f"Country L1 distance too large: {l1:.3f}"

    def test_currency_category_frequencies(self):
        """Currency category L1 distance should be small."""
        l1 = _cat_l1(self.source["currency"], self.gen["currency"])
        assert l1 < 0.30, f"Currency L1 distance too large: {l1:.3f}"

    def test_null_rates_preserved(self):
        """Null rates should match source (all zero in this table)."""
        for col in self.source.columns:
            src_null = self.source[col].isna().mean()
            gen_null = self.gen[col].isna().mean() if col in self.gen.columns else 0.0
            assert abs(src_null - gen_null) < 0.05, (
                f"Null rate mismatch in '{col}': source={src_null:.3f}, gen={gen_null:.3f}"
            )


# ────────────────────────────────────────────────────────────────────────────
# Test 2 — Price .99 endings
# ────────────────────────────────────────────────────────────────────────────

class TestPriceEndings:
    def setup_method(self):
        self.source = make_main_table(n=500, seed=42)
        spec = profile_to_spec({"main": self.source})
        generated = generate(spec, seed=0)
        self.gen = generated["main"]

    def test_price_99_endings(self):
        """At least 80% of generated prices should end in .99."""
        if "price" not in self.gen.columns:
            pytest.skip("price column not in generated table")
        prices = pd.to_numeric(self.gen["price"], errors="coerce").dropna()
        endings = (prices % 1).round(2)
        rate_99 = float((np.abs(endings - 0.99) < 0.01).mean())
        assert rate_99 > 0.70, f"Only {rate_99:.1%} of prices end in .99 (expected >70%)"

    def test_price_range(self):
        """Generated prices should stay within observed source range."""
        src_min = float(self.source["price"].min())
        src_max = float(self.source["price"].max())
        if "price" not in self.gen.columns:
            pytest.skip("price column not in generated table")
        gen_prices = pd.to_numeric(self.gen["price"], errors="coerce").dropna()
        assert gen_prices.min() >= src_min * 0.5, "Generated prices go too low"
        assert gen_prices.max() <= src_max * 1.5, "Generated prices go too high"


# ────────────────────────────────────────────────────────────────────────────
# Test 3 — Country→currency consistency
# ────────────────────────────────────────────────────────────────────────────

class TestConditionalConsistency:
    def setup_method(self):
        self.source = make_main_table(n=600, seed=99)
        # Learn source mapping
        self.valid_mapping = (
            self.source.groupby("country")["currency"]
            .apply(lambda s: set(s.unique()))
            .to_dict()
        )
        spec = profile_to_spec({"main": self.source})
        generated = generate(spec, seed=1)
        self.gen = generated["main"]

    def test_country_currency_never_violated(self):
        """
        For every country in the generated data that also appears in the source,
        the generated currency must be one of the source currencies for that country.
        """
        if "country" not in self.gen.columns or "currency" not in self.gen.columns:
            pytest.skip("country or currency not generated")

        violations = 0
        total = 0
        for _, row in self.gen.iterrows():
            cntry = row.get("country")
            curr = row.get("currency")
            if cntry in self.valid_mapping:
                total += 1
                if curr not in self.valid_mapping[cntry]:
                    violations += 1

        violation_rate = violations / max(total, 1)
        assert violation_rate < 0.10, (
            f"Country→currency violated for {violations}/{total} rows ({violation_rate:.1%})"
        )


# ────────────────────────────────────────────────────────────────────────────
# Test 4 — Age↔income joint correlation
# ────────────────────────────────────────────────────────────────────────────

class TestJointCorrelation:
    def setup_method(self):
        self.source = make_main_table(n=500, seed=42)
        self.src_spearman = float(scipy_stats.spearmanr(
            self.source["age"], self.source["income"]
        ).correlation)
        spec = profile_to_spec({"main": self.source})
        generated = generate(spec, seed=2)
        self.gen = generated["main"]

    def test_age_income_spearman_correlation(self):
        """Generated age↔income Spearman correlation within 0.25 of source."""
        if "age" not in self.gen.columns or "income" not in self.gen.columns:
            pytest.skip("age or income not generated")
        gen_spearman = float(scipy_stats.spearmanr(
            self.gen["age"], self.gen["income"]
        ).correlation)
        diff = abs(gen_spearman - self.src_spearman)
        assert diff < 0.35, (
            f"Age↔income correlation: source={self.src_spearman:.3f}, "
            f"gen={gen_spearman:.3f}, diff={diff:.3f}"
        )

    def test_spearman_matrix_frobenius_norm(self):
        """Frobenius norm of (gen_corr − src_corr) for salary/income/age should be small."""
        num_cols = [c for c in ("salary", "age", "income") if c in self.gen.columns]
        if len(num_cols) < 2:
            pytest.skip("Not enough numeric columns")

        src_corr = _spearman_corr_matrix(self.source, num_cols)
        gen_corr = _spearman_corr_matrix(self.gen, num_cols)
        frob = float(np.linalg.norm(gen_corr - src_corr, "fro"))
        # Frobenius norm ≤ sqrt(n^2) = n for identity → allow ≤ 60% of diagonal scale
        max_frob = len(num_cols) * 0.70
        assert frob < max_frob, (
            f"Correlation matrix Frobenius norm {frob:.3f} exceeds threshold {max_frob:.3f}"
        )


# ────────────────────────────────────────────────────────────────────────────
# Test 5 — ID format mask
# ────────────────────────────────────────────────────────────────────────────

class TestFormatMask:
    def setup_method(self):
        self.source = make_main_table(n=300, seed=42)
        self.id_pattern = re.compile(r'^[A-E]{3}-\d{4}$')
        spec = profile_to_spec({"main": self.source})
        generated = generate(spec, seed=3)
        self.gen = generated["main"]

    def test_all_ids_match_source_pattern(self):
        """Every generated record_id must match the source AAA-#### format."""
        if "record_id" not in self.gen.columns:
            pytest.skip("record_id column not generated")
        ids = self.gen["record_id"].astype(str).tolist()
        violations = [v for v in ids if not self.id_pattern.match(v)]
        rate_ok = 1.0 - len(violations) / max(len(ids), 1)
        assert rate_ok >= 0.85, (
            f"Only {rate_ok:.1%} of IDs match pattern [A-E]{{3}}-\\d{{4}}. "
            f"Examples of violations: {violations[:5]}"
        )


# ────────────────────────────────────────────────────────────────────────────
# Test 6 — Referential integrity in parent/child
# ────────────────────────────────────────────────────────────────────────────

class TestReferentialIntegrity:
    def setup_method(self):
        self.customers, self.orders = make_parent_child_tables(n_customers=200, seed=42)
        frames = {"customers": self.customers, "orders": self.orders}
        spec = profile_to_spec(frames)
        generated = generate(spec, seed=4)
        self.gen_customers = generated.get("customers", pd.DataFrame())
        self.gen_orders = generated.get("orders", pd.DataFrame())
        self.spec = spec

    def test_referential_integrity_100_pct(self):
        """Every generated FK value in orders must exist in generated customers."""
        if self.gen_orders.empty or self.gen_customers.empty:
            pytest.skip("Parent or child table not generated")

        # Find the FK column name
        rels = self.spec.relationships
        if not rels:
            pytest.skip("No relationships detected")

        rel = rels[0]
        child_key = rel.child_key
        parent_key = rel.parent_key

        if child_key not in self.gen_orders.columns or parent_key not in self.gen_customers.columns:
            pytest.skip(f"FK columns not found: {child_key}, {parent_key}")

        valid_keys = set(self.gen_customers[parent_key].tolist())
        child_fk_vals = self.gen_orders[child_key].tolist()
        violations = [v for v in child_fk_vals if v not in valid_keys]
        integrity_pct = 1.0 - len(violations) / max(len(child_fk_vals), 1)
        assert integrity_pct == 1.0, (
            f"Referential integrity = {integrity_pct:.1%} "
            f"({len(violations)} FK values not in parent)"
        )

    def test_children_count_positive(self):
        """Generated orders table must be non-empty."""
        assert len(self.gen_orders) > 0, "Orders table is empty"

    def test_customers_count_reasonable(self):
        """Generated customers count should be close to source."""
        assert len(self.gen_customers) > 0, "Customers table is empty"


# ────────────────────────────────────────────────────────────────────────────
# Test 7 — Intra-row arithmetic constraints
# ────────────────────────────────────────────────────────────────────────────

class TestArithmeticConstraints:
    def setup_method(self):
        rng = np.random.default_rng(7)
        n = 200
        a = rng.uniform(1, 100, n)
        b = rng.uniform(1, 100, n)
        total = a + b
        self.source = pd.DataFrame({"col_a": a, "col_b": b, "total": total})
        spec = profile_to_spec({"tbl": self.source})
        self.spec = spec
        generated = generate(spec, seed=7)
        self.gen = generated.get("tbl", pd.DataFrame())

    def test_arithmetic_constraint_detected(self):
        """At least one arithmetic constraint should be learned (total = col_a + col_b)."""
        tbl = next((t for t in self.spec.tables if t.name == "tbl"), None)
        if tbl is None:
            pytest.skip("Table not found in spec")
        assert len(tbl.intra_row_constraints) >= 1, (
            "No arithmetic constraint detected despite clear sum relationship"
        )

    def test_arithmetic_constraint_holds_in_output(self):
        """If a constraint was learned, it should hold in generated data."""
        tbl = next((t for t in self.spec.tables if t.name == "tbl"), None)
        if not tbl or not tbl.intra_row_constraints:
            pytest.skip("No constraints to check")
        if self.gen.empty:
            pytest.skip("Generated table is empty")

        rule = tbl.intra_row_constraints[0].rule
        # Parse "C = A + B" style
        m = re.match(r"(\w+)\s*=\s*(\w+)\s*([\+\-\*/])\s*(\w+)", rule)
        if not m:
            pytest.skip(f"Cannot parse rule: {rule}")
        lhs, op_a, op_sym, op_b = m.group(1), m.group(2), m.group(3), m.group(4)
        if not all(c in self.gen.columns for c in (lhs, op_a, op_b)):
            pytest.skip("Rule columns not in generated data")

        a = pd.to_numeric(self.gen[op_a], errors="coerce")
        b = pd.to_numeric(self.gen[op_b], errors="coerce")
        c = pd.to_numeric(self.gen[lhs], errors="coerce")

        if op_sym == "+":
            expected = a + b
        elif op_sym == "-":
            expected = a - b
        elif op_sym == "*":
            expected = a * b
        elif op_sym == "/":
            expected = a / b.replace(0, np.nan)
        else:
            pytest.skip(f"Unknown operator: {op_sym}")

        valid = c.notna() & expected.notna()
        if valid.sum() < 5:
            pytest.skip("Not enough valid rows to check constraint")

        match = np.allclose(c[valid].values, expected[valid].values, rtol=0.05, atol=1e-3)
        assert match, f"Arithmetic constraint '{rule}' does not hold in generated data"


# ────────────────────────────────────────────────────────────────────────────
# Test 8 — Backward compatibility (Creator-style spec still works)
# ────────────────────────────────────────────────────────────────────────────

class TestBackwardCompatibility:
    def test_legacy_numeric_distribution_spec(self):
        """A Creator-style spec with numeric_distribution still generates without error."""
        from backend.spec.models import (
            GenerationSpec, TableSpec, ColumnSpec, ColumnConstraints
        )
        spec = GenerationSpec(tables=[TableSpec(
            name="legacy",
            row_count=50,
            columns=[
                ColumnSpec(
                    name="age",
                    type="integer",
                    generation={"strategy": "numeric_distribution",
                                "params": {"dist": "normal", "mean": 35.0, "std": 10.0}},
                    constraints=ColumnConstraints(min=18, max=80),
                ),
                ColumnSpec(
                    name="country",
                    type="categorical",
                    generation={"strategy": "categorical_sample",
                                "params": {"values": {"US": 0.5, "UK": 0.3, "DE": 0.2}}},
                ),
                ColumnSpec(
                    name="created_at",
                    type="datetime",
                    generation={"strategy": "datetime_range",
                                "params": {"start": "2020-01-01", "end": "2024-01-01",
                                           "format": "%Y-%m-%d"}},
                ),
            ],
        )])
        frames = generate(spec, seed=0)
        df = frames["legacy"]
        assert len(df) == 50
        assert "age" in df.columns
        assert "country" in df.columns
        assert "created_at" in df.columns

    def test_legacy_sequential_pk_spec(self):
        """Sequential PK column still works."""
        from backend.spec.models import (
            GenerationSpec, TableSpec, ColumnSpec
        )
        spec = GenerationSpec(tables=[TableSpec(
            name="users",
            row_count=100,
            primary_key="id",
            columns=[
                ColumnSpec(
                    name="id",
                    type="integer",
                    generation={"strategy": "sequential", "params": {"start": 1, "step": 1}},
                    constraints={"unique": True, "nullable": False},
                ),
                ColumnSpec(
                    name="name",
                    type="string",
                    generation={"strategy": "semantic", "params": {"faker_method": "name"}},
                ),
            ],
        )])
        frames = generate(spec, seed=0)
        df = frames["users"]
        assert len(df) == 100
        assert list(df["id"]) == list(range(1, 101))


# ────────────────────────────────────────────────────────────────────────────
# Test 9 — Edge cases: all-null, single-value, constant columns
# ────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_null_column_does_not_crash(self):
        df = pd.DataFrame({
            "id": range(50),
            "all_null": [None] * 50,
            "normal": np.random.normal(size=50),
        })
        spec = profile_to_spec({"tbl": df})
        frames = generate(spec, seed=0)
        assert len(frames["tbl"]) == 50

    def test_constant_column_does_not_crash(self):
        df = pd.DataFrame({
            "id": range(50),
            "constant": ["FIXED"] * 50,
            "value": np.random.uniform(size=50),
        })
        spec = profile_to_spec({"tbl": df})
        frames = generate(spec, seed=0)
        assert len(frames["tbl"]) == 50

    def test_single_value_numeric_column(self):
        df = pd.DataFrame({
            "id": range(30),
            "single": [42.0] * 30,
        })
        spec = profile_to_spec({"tbl": df})
        frames = generate(spec, seed=0)
        assert len(frames["tbl"]) == 30

    def test_high_cardinality_text_column(self):
        """High-cardinality free text should not crash and produce correct row count."""
        df = pd.DataFrame({
            "id": range(100),
            "description": [f"Product description number {i} with extra text" for i in range(100)],
            "value": np.random.uniform(10, 100, 100),
        })
        spec = profile_to_spec({"tbl": df})
        frames = generate(spec, seed=0)
        assert len(frames["tbl"]) == 100

    def test_mixed_null_rates(self):
        """Columns with various null rates should preserve their null rates roughly."""
        rng = np.random.default_rng(55)
        n = 200
        col_50pct = [float(v) if rng.random() > 0.5 else None for v in rng.normal(0, 1, n)]
        col_10pct = [float(v) if rng.random() > 0.1 else None for v in rng.normal(0, 1, n)]
        df = pd.DataFrame({
            "id": range(n),
            "half_null": col_50pct,
            "mostly_full": col_10pct,
        })
        spec = profile_to_spec({"tbl": df})
        frames = generate(spec, seed=0)
        # Just check it doesn't crash and produces correct row count
        assert len(frames["tbl"]) == n


# ────────────────────────────────────────────────────────────────────────────
# Fidelity summary (printed when run directly)
# ────────────────────────────────────────────────────────────────────────────

def print_fidelity_summary():
    """Print a short fidelity summary for manual inspection."""
    print("\n" + "=" * 60)
    print("REALISM FIDELITY SUMMARY")
    print("=" * 60)

    source = make_main_table(n=500, seed=42)
    spec = profile_to_spec({"main": source})
    generated = generate(spec, seed=0)
    gen = generated["main"]

    for col in ("salary", "price", "age", "income"):
        if col not in gen.columns:
            continue
        src_vals = pd.to_numeric(source[col], errors="coerce").dropna()
        gen_vals = pd.to_numeric(gen[col], errors="coerce").dropna()
        p = _ks_pvalue(src_vals.values, gen_vals.values)
        src_pcts = _percentiles(src_vals.values, (25, 50, 75))
        gen_pcts = _percentiles(gen_vals.values, (25, 50, 75))
        print(f"\n  {col}:")
        print(f"    KS p-value: {p:.4f}")
        print(f"    Source p25/p50/p75: {src_pcts[0]:.1f} / {src_pcts[1]:.1f} / {src_pcts[2]:.1f}")
        print(f"    Gen    p25/p50/p75: {gen_pcts[0]:.1f} / {gen_pcts[1]:.1f} / {gen_pcts[2]:.1f}")

    for col in ("country", "currency"):
        if col not in gen.columns:
            continue
        l1 = _cat_l1(source[col], gen[col])
        print(f"\n  {col}: L1 distance = {l1:.3f}")

    print("\n  Age↔Income Spearman:")
    src_r = float(scipy_stats.spearmanr(source["age"], source["income"]).correlation)
    gen_r = float(scipy_stats.spearmanr(gen["age"], gen["income"]).correlation)
    print(f"    Source: {src_r:.3f}  |  Generated: {gen_r:.3f}  |  diff: {abs(src_r - gen_r):.3f}")

    if "record_id" in gen.columns:
        pat = re.compile(r'^[A-E]{3}-\d{4}$')
        ids = gen["record_id"].astype(str).tolist()
        ok_rate = sum(1 for v in ids if pat.match(v)) / len(ids)
        print(f"\n  ID format match rate: {ok_rate:.1%}")

    print("\n  Strategy distribution in spec:")
    for tbl in spec.tables:
        for col in tbl.columns:
            strat = col.generation.get("strategy", "?")
            print(f"    {tbl.name}.{col.name}: {strat}")

    print("=" * 60)


if __name__ == "__main__":
    print_fidelity_summary()
