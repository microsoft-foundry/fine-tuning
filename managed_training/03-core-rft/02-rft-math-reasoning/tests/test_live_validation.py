from types import SimpleNamespace

from live_validation import assess_completed_job_reuse, summarize_result_csv


def test_completed_job_reuse_requires_every_recipe_check():
    job = SimpleNamespace(
        status="succeeded",
        model="qwen3.6-35b-a3b",
        training_file="train",
        validation_file="valid",
        method=SimpleNamespace(
            type="reinforcement",
            reinforcement=SimpleNamespace(
                hyperparameters=SimpleNamespace(
                    batch_size=128,
                    eval_interval=25,
                    eval_samples=2,
                    learning_rate_multiplier=2.0,
                    n_epochs=2,
                    reasoning_effort="medium",
                )
            ),
        ),
    )
    files = {
        "train": SimpleNamespace(bytes=8),
        "valid": SimpleNamespace(bytes=10),
    }

    result = assess_completed_job_reuse(
        job,
        base_model="qwen3.6-35b-a3b",
        retrieve_file=files.__getitem__,
        retrieve_content={"train": b"training", "valid": b"validation"}.__getitem__,
        training_sha256=(
            "c2fb788c7deedbeaa296e424d4c2921b871a4f6cb4cf393c1c1105653ab399b4"
        ),
        validation_sha256=(
            "98c41dcd20b86b86830ec0794559835614458ceaae0f0ec77a3ed1cd3a1f7d55"
        ),
        local_regrade_matches_service=True,
    )

    assert result["reusable"] is True
    assert all(result["checks"].values())


def test_completed_job_reuse_rejects_unverified_regrade():
    job = SimpleNamespace(
        status="succeeded",
        model="qwen3.6-35b-a3b",
        training_file="train",
        validation_file="valid",
        method=SimpleNamespace(
            type="reinforcement",
            reinforcement=SimpleNamespace(
                hyperparameters=SimpleNamespace(
                    batch_size=128,
                    eval_interval=25,
                    eval_samples=2,
                    learning_rate_multiplier=2.0,
                    n_epochs=2,
                    reasoning_effort="medium",
                )
            ),
        ),
    )
    files = {
        "train": SimpleNamespace(bytes=8),
        "valid": SimpleNamespace(bytes=10),
    }

    result = assess_completed_job_reuse(
        job,
        base_model="qwen3.6-35b-a3b",
        retrieve_file=files.__getitem__,
        retrieve_content={"train": b"training", "valid": b"validation"}.__getitem__,
        training_sha256=(
            "c2fb788c7deedbeaa296e424d4c2921b871a4f6cb4cf393c1c1105653ab399b4"
        ),
        validation_sha256=(
            "98c41dcd20b86b86830ec0794559835614458ceaae0f0ec77a3ed1cd3a1f7d55"
        ),
        local_regrade_matches_service=False,
    )

    assert result["reusable"] is False
    assert result["checks"]["local_regrade_matches_service"] is False


def test_result_csv_summary_ignores_empty_metric_cells():
    summary = summarize_result_csv(
        "step,train_mean_reward,full_valid_mean_reward,pg_loss_sum,"
        "total_loss_sum,entropy_loss_sum\n"
        "1,0.095,0.05,262307.9,262307.9,70.7\n"
        "2,0.22,,304185.9,304185.9,55.1\n"
        "3,,0.2,,,\n"
    )

    assert summary == {
        "training_reward": [0.095, 0.22],
        "validation_reward": [0.05, 0.2],
        "policy_gradient_loss_sum": [262307.9, 304185.9],
        "total_loss_sum": [262307.9, 304185.9],
        "entropy_loss_sum": [70.7, 55.1],
    }
