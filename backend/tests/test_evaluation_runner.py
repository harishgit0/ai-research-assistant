import json

import pytest

from app.services.evaluation.run_evaluation import run_evaluation


def test_evaluation_requires_nonempty_dataset(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"version": 1, "cases": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="No evaluation cases"):
        run_evaluation(dataset)
