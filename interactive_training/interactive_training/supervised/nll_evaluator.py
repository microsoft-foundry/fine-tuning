import itertools

from azure.ai.finetuningsessions.models import Datum, TensorData

from interactive_training.eval.evaluators import TrainingClientEvaluator
from interactive_training.supervised.common import compute_mean_nll
from interactive_training.supervised.types import SupervisedDataset


def _datum_token_count(datum: Datum) -> int:
    tokens = datum.loss_fn_inputs.get("target_tokens")
    if isinstance(tokens, dict):
        return len(tokens.get("data", []))
    return len(getattr(tokens, "data", []) or [])


def _datum_loss_token_count(datum: Datum) -> int:
    weights = datum.loss_fn_inputs.get("weights")
    if isinstance(weights, dict):
        values = weights.get("data", [])
    else:
        values = getattr(weights, "data", []) or []
    return int(sum(values))


class NLLEvaluator(TrainingClientEvaluator):
    def __init__(self, data: list[Datum], name: str = "test"):
        self.name = name
        self.data = data

    async def __call__(self, training_client) -> dict[str, float]:
        future = await training_client.forward_async(self.data, loss_fn="cross_entropy")
        result = await future.result_async()
        logprobs = [x["logprobs"] for x in result.loss_fn_outputs]
        weights = [TensorData(**datum.loss_fn_inputs["weights"]) if isinstance(datum.loss_fn_inputs["weights"], dict) else datum.loss_fn_inputs["weights"] for datum in self.data]
        nll = compute_mean_nll(logprobs, weights)
        prefix = self.name
        return {
            f"{prefix}/nll": nll,
            f"{prefix}/num_sequences": len(self.data),
            f"{prefix}/num_tokens": sum(_datum_token_count(datum) for datum in self.data),
            f"{prefix}/num_loss_tokens": sum(_datum_loss_token_count(datum) for datum in self.data),
            f"{prefix}/num_steps": 1,
        }

    @classmethod
    def from_dataset(cls, dataset: SupervisedDataset, name: str = "test") -> "NLLEvaluator":
        all_data = list(itertools.chain(*[dataset.get_batch(i) for i in range(len(dataset))]))
        return cls(all_data, name=name)
