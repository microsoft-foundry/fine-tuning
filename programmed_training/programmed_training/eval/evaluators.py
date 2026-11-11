import logging
from typing import Callable

from azure.ai.finetuningsessions import FineTuningSession

# Set up logger
logger = logging.getLogger(__name__)


class TrainingClientEvaluator:
    """
    An evaluator that takes in a TrainingClient
    """

    async def __call__(self, training_client: FineTuningSession) -> dict[str, float]:
        raise NotImplementedError


class SamplingClientEvaluator:
    """
    An evaluator that takes in a TokenCompleter
    """

    async def __call__(
        self,
        sampling_client: FineTuningSession,
        *,
        step: int | None = None,
    ) -> dict[str, float]:
        raise NotImplementedError


EvaluatorBuilder = Callable[[], TrainingClientEvaluator | SamplingClientEvaluator]
SamplingClientEvaluatorBuilder = Callable[[], SamplingClientEvaluator]
Evaluator = TrainingClientEvaluator | SamplingClientEvaluator
