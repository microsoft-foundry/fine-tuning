import copy
import json
import subprocess
import sys

import pytest

from interactive_training.recipes.tool_ner_rl.ner_task import (
    ENTITY_LABELS,
    NerEpisode,
    NerTask,
)
from interactive_training.renderers import get_renderer
from interactive_training.renderers.gpt_oss import (
    _format_tool_definition,
    _resolve_tool_schema,
)


def _definition(schema):
    return _format_tool_definition({"name": "probe", "parameters": schema})


class ReversibleTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8")


@pytest.mark.parametrize("labels", [("GIVENNAME", "CITY"), ENTITY_LABELS])
@pytest.mark.parametrize(
    ("tool_name", "array_name", "fields"),
    [
        ("ground_entities", "proposals", ("text: string", "label:")),
        ("finalize", "entities", ("label:", "span_ids: string[]")),
    ],
)
def test_ner_tool_references_preserve_nested_fields(
    tool_name, array_name, fields, labels
):
    task = NerTask(
        uid="schema-probe",
        source_text="Ada lives in London.",
        labels=labels,
        gold_entities=frozenset(),
    )
    specs = [tool.to_spec() for tool in NerEpisode(task).tools_for_variant("fast")]
    spec = next(spec for spec in specs if spec["name"] == tool_name)

    definition = _format_tool_definition(spec)

    assert f"{array_name}: any[]" not in definition
    for field in fields:
        assert field in definition
    assert " | ".join(json.dumps(label) for label in labels) in definition


def test_reference_free_definition_is_unchanged():
    schema = {
        "type": "object",
        "description": "Input",
        "properties": {
            "query": {"type": "string", "description": "Search terms"},
            "limit": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    }
    assert _resolve_tool_schema(schema) is schema
    assert _definition(schema) == (
        "type probe = (_: // Input\n{\n// Search terms\nquery: string,\n"
        "limit?: number, // default: 10\n}) => any;"
    )


@pytest.mark.parametrize("container", ["$defs", "definitions"])
def test_root_reference_and_repeated_references_do_not_mutate_input(container):
    schema = {
        "$id": "https://example.invalid/root.json",
        container: {
            "Entry": {
                "type": "object",
                "properties": {
                    "span_ids": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["span_ids"],
            },
            "Args": {
                "type": "object",
                "properties": {
                    "first": {"$ref": f"#/{container}/Entry"},
                    "second": {"$ref": f"#/{container}/Entry"},
                },
                "required": ["first"],
            },
        },
        "$ref": f"#/{container}/Args",
    }
    original = copy.deepcopy(schema)
    rendered = _definition(schema)
    assert "first: { span_ids: string[] }" in rendered
    assert "second?: { span_ids: string[] }" in rendered
    assert schema == original


@pytest.mark.parametrize(
    ("name", "pointer"),
    [
        ("Entry/Name", "Entry~1Name"),
        ("Entry~Name", "Entry~0Name"),
        ("Entry Name", "Entry%20Name"),
        ("Entry~1Name", "Entry~01Name"),
        ("Entry%20Name", "Entry%2520Name"),
        ("Entry/Name", "Entry%7E1Name"),
        ("", ""),
        ("0", "0"),
    ],
)
def test_escaped_json_pointers(name, pointer):
    schema = {
        "$defs": {name: {"type": "string"}},
        "type": "object",
        "properties": {"value": {"$ref": f"#/$defs/{pointer}"}},
    }
    assert "value?: string" in _definition(schema)


@pytest.mark.parametrize("index", ["0", "1"])
def test_local_pointer_can_traverse_array_indices(index):
    schema = {
        "$defs": {"Choice": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
        "type": "object",
        "properties": {"value": {"$ref": f"#/$defs/Choice/anyOf/{index}"}},
    }
    expected = "string" if index == "0" else "number"
    assert f"value?: {expected}" in _definition(schema)


@pytest.mark.parametrize("index", ["-1", "-", "00", "01", "+1", "2", "", "1.0"])
def test_local_pointer_rejects_invalid_array_indices(index):
    schema = {
        "$defs": {"Choice": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
        "$ref": f"#/$defs/Choice/anyOf/{index}",
    }
    with pytest.raises(ValueError, match="GPT-OSS tool 'probe'.*Unresolvable"):
        _definition(schema)


@pytest.mark.parametrize("name", ["Entry~2Name", "Entry~", "Entry~~0Name"])
def test_local_pointer_rejects_invalid_escapes_even_when_key_exists(name):
    with pytest.raises(ValueError, match="GPT-OSS tool 'probe'.*Invalid escape"):
        _definition({"$defs": {name: {"type": "string"}}, "$ref": f"#/$defs/{name}"})


@pytest.mark.parametrize("target", [None, True, "string", [], 1])
def test_local_pointer_rejects_non_object_targets(target):
    with pytest.raises(ValueError, match="GPT-OSS tool 'probe'.*object-valued"):
        _definition({"$defs": {"Value": target}, "$ref": "#/$defs/Value"})


def test_sibling_annotations_and_constraints_are_preserved():
    schema = {
        "$defs": {"Text": {"type": "string", "description": "Base", "minLength": 1}},
        "type": "object",
        "properties": {
            "value": {"$ref": "#/$defs/Text", "description": "Field", "maxLength": 8},
        },
    }
    assert _resolve_tool_schema(schema)["properties"]["value"] == {
        "type": "string",
        "description": "Field",
        "minLength": 1,
        "maxLength": 8,
    }
    assert "// Field\nvalue?: string" in _definition(schema)


def test_sibling_properties_and_required_fields_are_conjoined():
    schema = {
        "$defs": {
            "Entry": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": ["CITY", "GIVENNAME"]},
                    "span_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["label"],
            },
        },
        "$ref": "#/$defs/Entry",
        "properties": {"label": {"enum": ["CITY"]}},
        "required": ["span_ids"],
    }
    rendered = _definition(schema)
    assert 'label: ("CITY" | "GIVENNAME") & "CITY"' in rendered
    assert "span_ids: string[]" in rendered
    assert "label?:" not in rendered and "span_ids?:" not in rendered


def test_conflicting_sibling_constraints_are_not_overwritten():
    schema = {
        "$defs": {"Text": {"type": "string", "maxLength": 8}},
        "$ref": "#/$defs/Text",
        "maxLength": 12,
    }
    resolved = _resolve_tool_schema(schema)
    assert resolved["allOf"][0] == {"type": "string", "maxLength": 8}
    assert resolved["allOf"][1]["maxLength"] == 12
    assert resolved["allOf"][1]["$defs"] == schema["$defs"]


@pytest.mark.parametrize("has_properties", [False, True])
def test_closed_reference_does_not_gain_allowed_sibling_properties(has_properties):
    target = {"type": "object", "additionalProperties": False}
    if has_properties:
        target["properties"] = {"original": {"type": "string"}}
    schema = {
        "$defs": {"Closed": target},
        "$ref": "#/$defs/Closed",
        "properties": {"added": {"type": "string"}},
    }

    resolved = _resolve_tool_schema(schema)

    assert resolved["allOf"][0] == target
    assert "added" not in resolved["allOf"][0].get("properties", {})
    assert resolved["allOf"][1]["properties"] == schema["properties"]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("composition", [None, "allOf", "anyOf", "oneOf"])
def test_closed_reference_preserves_sibling_required_fields(nested, composition):
    schema = {
        "$defs": {
            "Entry": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "span_ids": {"type": "array", "items": {"type": "string"}},
                    "label": {"type": "string"},
                },
            },
        },
    }
    if composition:
        schema["$defs"]["Entry"] = {composition: [schema["$defs"]["Entry"]]}
    reference = {"$ref": "#/$defs/Entry", "required": ["span_ids"]}
    if nested:
        schema.update(type="object", properties={"entry": reference})
    else:
        schema.update(reference)

    rendered = _definition(schema)

    assert "span_ids: string[]" in rendered
    assert "span_ids?:" not in rendered
    assert "label?: string" in rendered
    resolved = _resolve_tool_schema(schema)
    reference_schema = resolved["properties"]["entry"] if nested else resolved
    assert reference_schema["allOf"][0] == schema["$defs"]["Entry"]
    assert reference_schema["allOf"][1]["required"] == ["span_ids"]


@pytest.mark.parametrize("composition", ["anyOf", "oneOf"])
def test_requiredness_does_not_leak_into_nested_objects_or_other_union_branches(
    composition,
):
    span_ids = {"type": "array", "items": {"type": "string"}}
    entry = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "span_ids": span_ids,
            "label": {"type": "string"},
            "nested": {"type": "object", "properties": {"span_ids": span_ids}},
        },
    }
    schema = {
        "$defs": {"Entry": {composition: [{**entry, "required": ["label"]}, entry]}},
        "$ref": "#/$defs/Entry",
        "required": ["span_ids"],
    }

    rendered = _definition(schema)

    assert "span_ids: string[], label: string, nested?: { span_ids?: string[] }" in rendered
    assert "span_ids: string[], label?: string, nested?: { span_ids?: string[] }" in rendered


def test_renderer_package_imports_without_jsonpointer():
    script = """
import importlib.abc
import sys

class NoJsonPointer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "jsonpointer" or fullname.startswith("jsonpointer."):
            raise ModuleNotFoundError("jsonpointer deliberately unavailable")

sys.meta_path.insert(0, NoJsonPointer())
from interactive_training.renderers.gpt_oss import _format_tool_definition
definition = _format_tool_definition({
    "name": "probe",
    "parameters": {
        "$defs": {"Text": {"type": "string"}},
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/Text"}},
    },
})
assert "value?: string" in definition
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, "(string | number)[]"),
        ({"oneOf": [{"type": "string"}, {"type": "null"}]}, "(string | null)[]"),
        ({"type": "string", "enum": ["CITY", "GIVENNAME"]}, '("CITY" | "GIVENNAME")[]'),
        ({"type": ["string", "null"]}, "(string | null)[]"),
    ],
)
def test_referenced_array_union_is_parenthesized(item, expected):
    schema = {
        "$defs": {"Value": item},
        "type": "object",
        "properties": {"values": {"type": "array", "items": {"$ref": "#/$defs/Value"}}},
    }
    assert f"values?: {expected}" in _definition(schema)


@pytest.mark.parametrize(
    "reference",
    ["https://example.invalid/schema", "file:///unread", "other.json", "#anchor", 1],
)
def test_external_and_non_pointer_references_are_rejected_without_io(
    reference, monkeypatch
):
    def unexpected_io(*args, **kwargs):
        pytest.fail("Reference resolution attempted external I/O")

    monkeypatch.setattr("urllib.request.urlopen", unexpected_io)
    monkeypatch.setattr("builtins.open", unexpected_io)
    with pytest.raises(ValueError, match="GPT-OSS tool 'probe'.*document-local"):
        _definition({"$ref": reference})


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "#"},
        {
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {"child": {"$ref": "#/$defs/Node"}},
                }
            },
            "$ref": "#/$defs/Node",
        },
        {
            "$defs": {
                "First": {"$ref": "#/$defs/Second"},
                "Second": {"$ref": "#/$defs/First"},
            },
            "$ref": "#/$defs/First",
        },
    ],
)
def test_recursive_references_fail_clearly(schema):
    with pytest.raises(ValueError, match="Cyclic tool schema reference"):
        _definition(schema)


def test_missing_reference_fails_clearly():
    with pytest.raises(ValueError, match="Unresolvable local tool schema reference"):
        _definition({"$ref": "#/$defs/Missing"})


def test_unused_definitions_and_example_data_are_not_resolved():
    schema = {
        "$defs": {
            "Text": {"type": "string"},
            "Unused": {"$ref": "https://example.invalid"},
        },
        "type": "object",
        "properties": {
            "value": {"$ref": "#/$defs/Text", "examples": [{"$ref": "literal data"}]}
        },
    }
    assert '"$ref": "literal data"' in _definition(schema)
    assert "value?: string" in _definition(schema)


def test_nested_id_scope_is_not_misresolved():
    schema = {
        "$defs": {"Text": {"$id": "other.json", "type": "string"}},
        "$ref": "#/$defs/Text",
    }
    with pytest.raises(ValueError, match=r"Nested \$id"):
        _definition(schema)


@pytest.mark.parametrize("branching", [False, True])
def test_expansion_has_depth_and_size_limits(branching):
    definitions = {"Value0": {"type": "string"}}
    for index in range(1, 70 if not branching else 14):
        reference = {"$ref": f"#/$defs/Value{index - 1}"}
        definitions[f"Value{index}"] = (
            {"type": "object", "properties": {"left": reference, "right": reference}}
            if branching
            else reference
        )
    schema = {"$defs": definitions, "$ref": f"#/$defs/Value{index}"}
    with pytest.raises(ValueError, match="reference expansion limit"):
        _definition(schema)


def test_sampling_and_training_share_the_resolved_tool_prompt():
    renderer = get_renderer("gpt_oss_no_sysprompt", ReversibleTokenizer())
    schema = {
        "$defs": {
            "Entry": {
                "type": "object",
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
            }
        },
        "type": "object",
        "properties": {
            "entities": {"type": "array", "items": {"$ref": "#/$defs/Entry"}}
        },
        "required": ["entities"],
    }
    messages = renderer.create_conversation_prefix_with_tools(
        [{"name": "finalize", "parameters": schema}], system_prompt="Call finalize."
    )
    messages.append({"role": "user", "content": "Extract entities."})
    sampling = renderer.build_generation_prompt(messages)
    training, _ = renderer.build_supervised_example(
        [*messages, {"role": "assistant", "content": json.dumps({"entities": []})}]
    )
    sampling_tokens = [token for chunk in sampling.chunks for token in chunk.tokens]
    training_tokens = [token for chunk in training.chunks for token in chunk.tokens]
    assert training_tokens[: len(sampling_tokens)] == sampling_tokens
    assert "entities: { label: string }[]" in renderer.tokenizer.decode(sampling_tokens)
