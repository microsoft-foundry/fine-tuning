from typing import Any, Literal, cast

import chz
from chz.tiepin import CastError

TrainingType = Literal["GlobalStandard"]

TRAINING_TYPES: tuple[TrainingType, ...] = ("GlobalStandard",)

_TRAINING_TYPES_BY_CASEFOLD = {
    training_type.casefold(): training_type for training_type in TRAINING_TYPES
}
_UNAVAILABLE_TRAINING_TYPES = {"datazonestandard", "developertier"}


def normalize_training_type(training_type: str | None) -> TrainingType | None:
    if training_type is None:
        return None

    if training_type.casefold() in _UNAVAILABLE_TRAINING_TYPES:
        raise ValueError(
            f"training_type {training_type!r} is currently unavailable; "
            "omit it or use GlobalStandard"
        )
    normalized = _TRAINING_TYPES_BY_CASEFOLD.get(training_type.casefold())
    if normalized is None:
        expected = ", ".join(TRAINING_TYPES)
        raise ValueError(
            f"Unsupported training_type {training_type!r}; expected one of: {expected}"
        )
    return cast(TrainingType, normalized)


def _parse_training_type(training_type: str) -> TrainingType:
    try:
        normalized = normalize_training_type(training_type)
    except ValueError as error:
        raise CastError(str(error)) from None
    assert normalized is not None
    return normalized


def training_type_field() -> Any:
    return chz.field(
        default=None,
        blueprint_cast=_parse_training_type,
        doc="Training service tier. Omit to let the service select the tier.",
    )
