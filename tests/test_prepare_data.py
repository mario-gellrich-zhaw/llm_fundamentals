(
    (
        "Local checks for the exact Yelp subset "
        "extraction; no downloads or API "
        "calls."
    )
)

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_data", ROOT / "scripts/prepare_data.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class SelectionTests(unittest.TestCase):
    """Check exact selection and preservation of source reviews."""

    def test_exact_mcdonalds_spellings_and_rejected_names(self):
        """Check exact mcdonalds spellings and rejected names."""
        accepted = [
            "McDonald's",
            "McDonalds",
            "McDonald’s",
            "McDonalds's",
            "MCDONALD’S",
        ]
        rejected = [
            "Ronald McDonald House",
            "McDonald Park",
            "McDonald Pest Control",
            "McDonald's Express",
            " McDonalds",
            None,
        ]
        names = pd.Series(accepted + rejected)
        self.assertEqual(
            names.str.match(
                prepare.MCDONALDS_PATTERN, case=False, na=False
            ).tolist(),
            [True] * len(accepted) + [False] * len(rejected),
        )

    @unittest.skipUnless(
        importlib.util.find_spec("pyarrow"),
        "Install pyarrow to test Parquet extraction",
    )
    def test_row_groups_text_preservation_duplicates_and_date_sort(self):
        """Check row groups text preservation duplicates and date sort."""
        names = [
            "McDonald's",
            "McDonalds",
            "McDonald’s",
            "McDonalds's",
            "IKEA",
            "IKEA Restaurant",
            "Ronald McDonald House",
            "McDonald Park",
            "McDonald Pest Control",
        ]
        businesses = pd.DataFrame(
            {
                "business_id": [f"b{i}" for i in range(len(names))],
                "name": names,
                "city": ["Test city"] * len(names),
                "state": ["FL"] * len(names),
                "stars": [3.0] * len(names),
                "review_count": [10] * len(names),
            }
        )
        texts = [
            "  Café, 'quoted'\nsecond line  ",
            "repeat",
            "repeat",
            "NA",
            "unchanged",
            "IKEA duplicate",
            "IKEA duplicate",
            "excluded one",
            "excluded two",
            "excluded three",
        ]
        reviews = pd.DataFrame(
            {
                "review_id": [f"r{i}" for i in range(10)],
                "business_id": [
                    "b0",
                    "b1",
                    "b2",
                    "b3",
                    "b0",
                    "b4",
                    "b5",
                    "b6",
                    "b7",
                    "b8",
                ],
                "stars": [1, 2, 3, 4, 5, 4, 5, 1, 2, 3],
                "date": [
                    f"2021-01-{day:02d}"
                    for day in [10, 8, 6, 4, 2, 9, 1, 3, 5, 7]
                ],
                "text": texts,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bp, rp = root / "business.parquet", root / "review.parquet"
            businesses.to_parquet(bp, index=False)
            reviews.to_parquet(rp, index=False, row_group_size=2)
            before = [bp.read_bytes(), rp.read_bytes()]
            report = prepare.extract_subsets(bp, rp, root / "out")
            self.assertEqual([bp.read_bytes(), rp.read_bytes()], before)
            bus = pd.read_csv(root / "out/businesses_mcdonalds_ikea.csv")
            self.assertEqual(
                bus.business_id.tolist(), [f"b{i}" for i in range(6)]
            )
            for chain, filename, ids in [
                (
                    "McDonald's",
                    "mcdonalds_reviews.csv",
                    [f"b{i}" for i in range(4)],
                ),
                ("IKEA", "ikea_reviews.csv", ["b4", "b5"]),
            ]:
                actual = pd.read_csv(
                    root / "out" / filename, keep_default_na=False
                )
                expected = reviews[reviews.business_id.isin(ids)].sort_values(
                    "date"
                )
                self.assertEqual(
                    list(actual.columns), prepare.REVIEW_OUTPUT_COLUMNS
                )
                self.assertEqual(
                    actual.review_id.tolist(), expected.review_id.tolist()
                )
                self.assertEqual(actual.text.tolist(), expected.text.tolist())
                self.assertTrue(actual.date.is_monotonic_increasing)
                self.assertEqual(
                    report["actual"][chain]["review_rows"], len(expected)
                )
            self.assertEqual(
                report["actual"]["IKEA"]["distinct_review_texts"], 1
            )
            # Miniature inputs deliberately differ; the script must report, not
            # fix, them.
            self.assertFalse(report["all_checks_match"])
            self.assertTrue(
                any(d["scope"] == "input" for d in report["differences"])
            )


if __name__ == "__main__":
    unittest.main()
