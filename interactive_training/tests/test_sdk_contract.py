"""SDK contract smoke test: catch cookbook <-> SDK staleness at PR time.

The cookbook calls ``azure-ai-finetuningsessions`` (the SDK) through the
adapters in :mod:`interactive_training.rl.train_azure` (``self._client.<method>(...)``).
Local SDK snapshots can share a published version string, and new cookbook code
can require a newer beta. An installed SDK can therefore lag the cookbook -- a recipe fails deep in a
training/checkpoint call (e.g. ``TypeError: save_weights_async() got an
unexpected keyword argument 'step_number'``, an ``AttributeError`` on a renamed
method, or an ``ImportError`` from ``azure.ai.finetuningsessions.models``).

Interactive Training CI installs sibling SDK source; public validation installs the pinned PyPI
release in a clean environment. This test checks either installation directly:
every SDK method the cookbook calls must exist and accept its keyword arguments.

**Zero manual upkeep.** The contract is not hand-maintained -- it is discovered
by statically parsing ``train_azure.py`` for ``self._client.<method>(...)`` call
sites and the keyword arguments they pass (via the :mod:`ast` module). When a
contributor adds or changes an SDK call, it is checked automatically with
nothing to remember.
"""

import ast
import inspect
from pathlib import Path

import pytest

from azure.ai.finetuningsessions.aio import FineTuningSessionClient

# The adapter module(s) whose ``self._client.*`` calls define the SDK contract.
# Paths are relative to the cookbook root (the parent of this tests/ dir).
ADAPTER_SOURCES = [
    Path(__file__).resolve().parent.parent / "interactive_training" / "rl" / "train_azure.py",
]

# The attribute the adapters hold the SDK client in (``self._client = client``).
CLIENT_ATTR = "_client"

# Core methods that must always be discovered. This is a tiny, stable anchor --
# NOT a hand-maintained contract -- so that an accidental rename of the client
# attribute (which would make discovery return nothing) can't let the
# parametrized checks pass vacuously.
ANCHOR_METHODS = frozenset(
    {"forward_backward_async", "optim_step_async", "save_weights_async", "sample"}
)


def _discover_contract() -> dict[str, set[str]]:
    """Parse the adapter source(s) for ``self.<CLIENT_ATTR>.<method>(...)`` calls.

    Returns a mapping ``method_name -> set(keyword argument names passed)``,
    unioned across every call site. ``**kwargs`` splats (``kw.arg is None``) are
    ignored -- only explicitly-named keyword arguments are part of the contract.
    """
    contract: dict[str, set[str]] = {}
    for source in ADAPTER_SOURCES:
        tree = ast.parse(source.read_text(), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Match ``self.<CLIENT_ATTR>.<method>`` : Attribute(attr=method,
            # value=Attribute(attr=CLIENT_ATTR, value=Name(id="self"))).
            if not isinstance(func, ast.Attribute):
                continue
            client = func.value
            if (
                isinstance(client, ast.Attribute)
                and client.attr == CLIENT_ATTR
                and isinstance(client.value, ast.Name)
                and client.value.id == "self"
            ):
                kwargs = {kw.arg for kw in node.keywords if kw.arg is not None}
                contract.setdefault(func.attr, set()).update(kwargs)
    return contract


CONTRACT = _discover_contract()


def _accepts_keyword(sig: inspect.Signature, name: str) -> bool:
    """True if a call may pass ``name=...`` -- explicit param or **kwargs."""
    for param in sig.parameters.values():
        if param.name == name and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def test_contract_discovery_is_sane() -> None:
    """Discovery actually found the SDK call sites.

    Guards against a refactor (e.g. renaming ``self._client``) silently making
    the contract empty, which would let every parametrized check pass vacuously.
    """
    assert CONTRACT, (
        f"No self.{CLIENT_ATTR}.<method>(...) calls were discovered in "
        f"{[str(p) for p in ADAPTER_SOURCES]}. Did the client attribute name "
        f"change? Update CLIENT_ATTR / ADAPTER_SOURCES in this test."
    )
    missing_anchors = ANCHOR_METHODS - CONTRACT.keys()
    assert not missing_anchors, (
        f"Expected core SDK calls not discovered: {sorted(missing_anchors)}. "
        f"The adapter changed shape -- check CLIENT_ATTR / ADAPTER_SOURCES."
    )


@pytest.mark.parametrize("method_name", sorted(CONTRACT))
def test_sdk_method_exists(method_name: str) -> None:
    """Every SDK method the cookbook calls is present on the client."""
    assert hasattr(FineTuningSessionClient, method_name), (
        f"FineTuningSessionClient is missing {method_name!r}. The installed "
        "azure-ai-finetuningsessions is out of date or incompatible with this "
        "cookbook -- follow this repository's SDK update instructions "
        "(see docs/troubleshooting.md, 'Stale SDK')."
    )


@pytest.mark.parametrize(
    ("method_name", "kwarg"),
    sorted(
        (name, kw) for name, kwargs in CONTRACT.items() for kw in kwargs
    ),
)
def test_sdk_method_accepts_kwarg(method_name: str, kwarg: str) -> None:
    """Each SDK method accepts the keyword arguments the cookbook passes."""
    method = getattr(FineTuningSessionClient, method_name, None)
    assert method is not None, (
        f"FineTuningSessionClient is missing {method_name!r} "
        "(see test_sdk_method_exists)."
    )
    sig = inspect.signature(method)
    assert _accepts_keyword(sig, kwarg), (
        f"FineTuningSessionClient.{method_name}() does not accept "
        f"{kwarg!r}. The installed azure-ai-finetuningsessions predates the "
        "cookbook code that calls it -- follow this repository's SDK update "
        "instructions (see docs/troubleshooting.md, 'Stale SDK')."
    )


def test_cookbook_adapter_imports() -> None:
    """The SDK-facing adapter imports cleanly against the installed SDK."""
    import interactive_training.rl.train_azure as train_azure

    assert hasattr(train_azure, "AzureSDKTrainingClient")
    assert hasattr(train_azure, "AzureSDKSamplingClient")
