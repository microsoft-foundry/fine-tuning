"""Synthetic alien-word color tasks shared by the MCP server and RL environment."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import random
from typing import Literal

Color = Literal[
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "black",
    "white",
    "brown",
]

ALIEN_WORD_TO_COLOR: dict[str, Color] = {
    "varkesh": "red",
    "tolmari": "orange",
    "senquor": "yellow",
    "irevax": "green",
    "nalmek": "blue",
    "quorali": "purple",
    "bexith": "pink",
    "zundra": "black",
    "kethuun": "white",
    "morvane": "brown",
}
COLOR_TO_ALIEN_WORD = {color: word for word, color in ALIEN_WORD_TO_COLOR.items()}
COLORS = tuple(COLOR_TO_ALIEN_WORD)


@dataclass(frozen=True)
class AlienColorTask:
    task_id: int
    colors: tuple[Color, ...]
    question: str
    answers: tuple[str, ...]


def build_mixed_tasks(count: int = 30, *, seed: int = 0) -> list[AlienColorTask]:
    """Build interleaved single/pair/triple prompts without evaluation wording."""
    if not 1 <= count <= 180:
        raise ValueError("mixed task count must be between 1 and 180")
    rng = random.Random(seed)
    single_count = min((count + 2) // 3, len(COLORS))
    remaining = count - single_count
    singles = [(color,) for color in rng.sample(list(COLORS), single_count)]
    pairs = rng.sample(list(itertools.permutations(COLORS, 2)), (remaining + 1) // 2)
    triples = rng.sample(list(itertools.permutations(COLORS, 3)), remaining // 2)
    colors_in_order = [
        colors for group in itertools.zip_longest(singles, pairs, triples)
        for colors in group if colors is not None
    ]
    tasks = []
    for index, colors in enumerate(colors_in_order):
        color_list = (
            colors[0]
            if len(colors) == 1
            else ", ".join(colors[:-1]) + f" and {colors[-1]}"
        )
        answer_fields = (
            "<alien-word>"
            if len(colors) == 1
            else ", ".join(f"{color}=<alien-word>" for color in colors)
        )
        tasks.append(AlienColorTask(
            task_id=1000 + index,
            colors=colors,
            question=(
                f"Recall the alien words for {color_list}, then verify each with the tool. "
                f"Then reply with exactly `Answer: {answer_fields}` using the requested color order."
            ),
            answers=tuple(COLOR_TO_ALIEN_WORD[color] for color in colors),
        ))
    return tasks


def build_tasks(count: int = 60, *, offset: int = 0) -> list[AlienColorTask]:
    """Build deterministic prompt variants whose alien-word mapping is hidden."""
    single_templates = (
        "Recall the alien word for {color}, then verify it with the tool.",
        "Use the tool to identify and verify the alien word for {color}.",
        "Which alien word summons {color}? Verify your choice with the tool.",
        "Recall which alien word produces {color}, then verify that choice.",
        "Identify the learned alien term for {color} and verify it with the tool.",
        "Verify the alien word for {color} using as few tool calls as possible.",
    )
    multi_templates = (
        "Recall the alien words for {colors}, then verify each with the tool.",
        "Use the tool to identify and verify the alien words for {colors}.",
        "Which alien words summon {colors}? Verify each choice with the tool.",
        "Recall which alien words produce {colors}, then verify those choices.",
        "Identify the learned alien terms for {colors} and verify each with the tool.",
        "Verify the alien words for {colors} using as few tool calls as possible.",
    )
    tasks: list[AlienColorTask] = []
    for task_id in range(offset, offset + count):
        target_count = (task_id // len(COLORS)) % 3 + 1
        template_index = (task_id // (3 * len(COLORS))) % len(single_templates)
        colors = tuple(
            COLORS[(task_id + 3 * index) % len(COLORS)]
            for index in range(target_count)
        )
        answers = tuple(COLOR_TO_ALIEN_WORD[color] for color in colors)
        if target_count == 1:
            instruction = single_templates[template_index].format(color=colors[0])
            answer_instruction = "Then reply with exactly `Answer: <alien-word>`."
        else:
            color_list = ", ".join(colors[:-1]) + f" and {colors[-1]}"
            instruction = multi_templates[template_index].format(colors=color_list)
            answer_fields = ", ".join(
                f"{color}=<alien-word>" for color in colors
            )
            answer_instruction = (
                f"Then reply with exactly `Answer: {answer_fields}` using the "
                "requested color order."
            )
        tasks.append(
            AlienColorTask(
                task_id=task_id,
                colors=colors,
                question=f"{instruction} {answer_instruction}",
                answers=answers,
            )
        )
    return tasks


def build_curriculum_tasks(
    train_count: int = 40,
    test_count: int = 30,
) -> tuple[list[AlienColorTask], list[AlienColorTask]]:
    """Build balanced recall-first training and held-out composition tasks."""
    template_width = 3 * len(COLORS)
    tasks = build_tasks(6 * template_width)
    by_template_and_size = {
        (template_index, target_count): [
            task
            for task in tasks
            if task.task_id // template_width == template_index
            and len(task.colors) == target_count
        ]
        for template_index in range(6)
        for target_count in (1, 2, 3)
    }
    atomic_candidates = [
        task
        for template_index in range(5)
        for task in by_template_and_size[(template_index, 1)]
    ]
    def composition_candidates(target_count: int) -> list[AlienColorTask]:
        return [
            by_template_and_size[
                ((color_index + template_offset) % 5, target_count)
            ][color_index]
            for template_offset in range(5)
            for color_index in range(len(COLORS))
        ]

    pair_candidates = composition_candidates(2)
    triple_candidates = composition_candidates(3)
    atomic_count = min(len(atomic_candidates), (train_count + 1) // 2)
    composition_count = train_count - atomic_count
    pair_count = min(len(pair_candidates), (composition_count + 1) // 2)
    triple_count = composition_count - pair_count
    train_candidates = [
        *atomic_candidates[:atomic_count],
        *pair_candidates[:pair_count],
        *triple_candidates[:triple_count],
    ]
    test_candidates = [
        task
        for target_count in (1, 2, 3)
        for task in by_template_and_size[(5, target_count)]
    ]
    if train_count > len(train_candidates):
        raise ValueError(
            f"train_count cannot exceed {len(train_candidates)}, got {train_count}"
        )
    if test_count > len(test_candidates):
        raise ValueError(
            f"test_count cannot exceed {len(test_candidates)}, got {test_count}"
        )
    return train_candidates[:train_count], test_candidates[:test_count]
