import asyncio
import random
from dataclasses import dataclass
from typing import Literal, Optional
import copy

import json
import numpy as np
import pandas as pd

from constants import MUTATION_PROMPTS, THINKING_STYLES
from embeddings.onnx_backend import ONNXEmbeddingModel
from openai_utils import get_completion_simple_async
from task import Task, run_task

# Mutation Operator Percentage
# Zero-order Hyper-Mutation 42%
# Lineage Based Mutation 26%
# First-order Hyper-Mutation 23%
# EDA Rank and Index Mutation 12.7%
# Direct Mutation 12%
# - 0-order: description -> new prompt
# - first order: prompt, mutation op -> new prompt
# EDA Mutation 10.7%
# Lamarckian Mutation 6.3%


@dataclass
class Unit:
    """
    One evolutionary unit. Includes task prompt & mutation prompt.
    """
    task_prompt: str
    mutation_prompt: str
    generation: int = 0
    fitness: Optional[float] = None
    origin: Optional[str] = None


@dataclass
class Generation:
    """
    Entire generation of evolutionary units.
    """
    step: int
    units: list[Unit]
    lineage: list[Unit]  # historical best in each generation

    def save_to_file(self, file_name: str):
        result = {
            "step": self.step,
            "lineage": [u.__dict__ for u in self.lineage],
            "units": [u.__dict__ for u in self.units],
        }
        with open(file_name, "a") as f:
            f.write(json.dumps(result) + "\n")



# All mutation operators take the Task, the Generation, and the Unit to mutate.
# They return a new Unit.


async def fresh_prompt(
    unit: Unit,
    generation: Optional[Generation],
    task: Task,
    model_name: Literal[
        "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4", "mistral"
    ] = "gpt-3.5-turbo",
):
    description = task.description
    meta_prompt = (
        f"I have the following instructions to solve a problem: {description}\n\n"
        "Provide a few hints/tips on how to approach this."
    )
    new_prompt = await get_completion_simple_async(
        meta_prompt,
        model_name=model_name,
        max_new_tokens=256,
        temperature=0.5,
    )

    # we mutate the task prompt, and retain the mutation prompt
    return Unit(
        task_prompt=new_prompt,
        mutation_prompt=unit.mutation_prompt,
        generation=unit.generation + 1,
        origin="fresh_prompt",
    )


async def direct_mutation(unit: Unit, generation: Generation, task: Task):
    """
    Directly mutate the prompt with the mutation prompt.
    """
    meta_prompt = (
        unit.mutation_prompt
        + "\n\n INSTRUCTION: "
        + unit.task_prompt
        + "\n\nProvide the new instruction."
    )
    new_prompt = await get_completion_simple_async(
        meta_prompt,
        model_name="gpt-3.5-turbo",
        max_new_tokens=256,
        temperature=0.5,
    )
    # we mutate the task prompt, and retain the mutation prompt
    return Unit(
        task_prompt=new_prompt,
        mutation_prompt=unit.mutation_prompt,
        generation=unit.generation + 1,
        origin="direct_mutation",
    )


async def eda_mutation(unit: Unit, generation: Generation, task: Task):
    """
    Use EDA to mutate the prompt.
    """
    # get diverse list of current generation
    embedding_model = ONNXEmbeddingModel(model_name="bge-micro-v2")
    all_task_prompts = [u.task_prompt for u in generation.units]
    random.shuffle(all_task_prompts)
    filtered_task_prompts = []
    filtered_task_embeddings = []
    for p in all_task_prompts:
        emb = embedding_model.embed(p)
        if len(filtered_task_embeddings) > 0:
            similarities = [np.dot(emb, g) for g in filtered_task_embeddings]
            if np.max(similarities) > 0.9:
                continue
        filtered_task_prompts.append(p)
        filtered_task_embeddings.append(emb)

    meta_prompt = (
        "Here is a variety of instructions for how to complete a task:\n"
        + "\n".join([" - " + p for p in filtered_task_prompts])
        + "\n\nGenerate a new, substantially different, better instruction based on these."
    )
    new_prompt = await get_completion_simple_async(
        meta_prompt,
        model_name="gpt-3.5-turbo",
        max_new_tokens=256,
        temperature=0.5,
    )
    # we mutate the task prompt, and retain the mutation prompt
    return Unit(
        task_prompt=new_prompt,
        mutation_prompt=unit.mutation_prompt,
        generation=unit.generation + 1,
        origin="eda_mutation",
    )


async def eda_ranked_mutation(unit: Unit, generation: Generation, task: Task):
    # get diverse list of current generation
    embedding_model = ONNXEmbeddingModel(model_name="bge-micro-v2")
    all_task_prompts = [(u.task_prompt, u.fitness) for u in generation.units]
    random.shuffle(all_task_prompts)
    filtered_task_prompts = []
    filtered_task_embeddings = []
    for p in all_task_prompts:
        emb = embedding_model.embed(p[0])
        if len(filtered_task_embeddings) > 0:
            similarities = [np.dot(emb, g) for g in filtered_task_embeddings]
            if np.max(similarities) > 0.9:
                continue
        filtered_task_prompts.append(p)
        filtered_task_embeddings.append(emb)

    # order by fitness
    filtered_task_prompts = sorted(filtered_task_prompts, key=lambda x: x[1])

    meta_prompt = (
        "Instructions ranked in ascending order of quality:\n"
        + "\n".join([" - " + p[0] for p in filtered_task_prompts])
        + "\n\nCreate a unique new instruction that's even better than the last one."
    )
    new_prompt = await get_completion_simple_async(
        meta_prompt,
        model_name="gpt-3.5-turbo",
        max_new_tokens=256,
        temperature=0.5,
    )
    # we mutate the task prompt, and retain the mutation prompt
    return Unit(
        task_prompt=new_prompt,
        mutation_prompt=unit.mutation_prompt,
        generation=unit.generation + 1,
        origin="eda_ranked_mutation",
    )


async def lineage_mutation(unit: Unit, generation: Generation, task: Task):
    # if there's no history, this won't work well
    if len(generation.lineage) < 3:
        # YOU BIG DUMMY YOU NEEDED TO AWAIT THIS
        unit = await eda_ranked_mutation(unit, generation, task)
        return unit
    else:
        meta_prompt = (
            "The following list shows how a task instruction was honed and improved over time. "
            + "Each instruction is better than the last.\n"
            + "\n".join([" - " + u.task_prompt for u in generation.lineage])
            + "\n\nContinue the improvement by sharing a unique new instruction that's even better than the last one."
        )
    new_prompt = await get_completion_simple_async(
        meta_prompt,
        model_name="gpt-3.5-turbo",
        max_new_tokens=256,
        temperature=0.5,
    )
    # we mutate the task prompt, and retain the mutation prompt
    return Unit(
        task_prompt=new_prompt,
        mutation_prompt=unit.mutation_prompt,
        generation=unit.generation + 1,
        origin="lineage_mutation",
    )


async def fresh_mutation_prompt(unit: Unit, generation: Generation, task: Task):
    thinking_style = random.choice(THINKING_STYLES)
    sampled_mutation_prompt = random.choice(MUTATION_PROMPTS)
    new_mutation_prompt = sampled_mutation_prompt + " " + thinking_style

    meta_prompt = (
        new_mutation_prompt
        + "\n\nINSTRUCTION: "
        + unit.task_prompt
        + "\n\nProvide the revised instruction."
    )

    new_prompt = await get_completion_simple_async(
        meta_prompt,
        model_name="gpt-3.5-turbo",
        max_new_tokens=256,
        temperature=0.5,
    )

    return Unit(
        task_prompt=new_prompt,
        mutation_prompt=new_mutation_prompt,
        generation=unit.generation + 1,
        origin="fresh_mutation_prompt",
    )


async def first_order_hypermutation(unit: Unit, generation: Generation, task: Task):
    meta_mutation_prompt = (
        "Please paraphrase and improve the following instruction, adding any additional "
        "explanation or details that could be helpful for someone trying to follow it.\n\n"
        f"INSTRUCTION: {unit.mutation_prompt}\n\nNow provide your revision."
    )
    new_mutation_prompt = await get_completion_simple_async(
        meta_mutation_prompt,
        model_name="gpt-3.5-turbo",
        max_new_tokens=256,
        temperature=0.5,
    )

    # use the new mutation prompt to mutate the task prompt
    meta_prompt = (
        new_mutation_prompt
        + "\n\n INSTRUCTION: "
        + unit.task_prompt
        + "\n\nProvide the new instruction."
    )

    new_prompt = await get_completion_simple_async(
        meta_prompt,
        model_name="gpt-3.5-turbo",
        max_new_tokens=256,
        temperature=0.5,
    )

    # both the mutation and task prompt have been mutated
    return Unit(
        task_prompt=new_prompt,
        mutation_prompt=new_mutation_prompt,
        generation=unit.generation + 1,
        origin="first_order_hypermutation",
    )


async def lamarckian_mutation(unit: Unit, generation: Generation, task: Task):
    # get an examplar for the task
    df = pd.read_csv(f"data/exemplars/{task.data_file}")
    exemplar = df.sample(1).iloc[0]
    meta_prompt = (
        "I gave a friend some really great instructions to solve a problem. "
        "Here is the problem and my friend's working-out of the answer:\n\n"
        f"PROBLEM: {exemplar['question']}\n\nSOLUTION: {exemplar['answer']}\n\n"
        "Please provide the instructions I gave to my friend that led to this correct working-out."
    )

    new_prompt = await get_completion_simple_async(
        meta_prompt,
        model_name="gpt-3.5-turbo",
        max_new_tokens=256,
        temperature=0.5,
    )

    # only the task prompt has been mutated
    return Unit(
        task_prompt=new_prompt,
        mutation_prompt=unit.mutation_prompt,
        generation=unit.generation + 1,
        origin="lamarckian_mutation",
    )


async def apply_random_mutation(unit: Unit, generation: Generation, task: Task):
    """
    Apply a random mutation operator.
    """
    mutation_operator = random.choice(
        [
            fresh_prompt,
            direct_mutation,
            eda_mutation,
            eda_ranked_mutation,
            lineage_mutation,
            fresh_mutation_prompt,
            first_order_hypermutation,
            lamarckian_mutation,
        ]
    )
    new_unit = await mutation_operator(unit, generation, task)
    return new_unit


async def score_generation(
    generation: Generation,
    task: Task,
    num_samples: int,
    model_name: Literal[
        "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4", "mistral"
    ] = "gpt-3.5-turbo",
):
    """
    Scores the generation by running the task on each unit.
    """
    # run the task on each unit
    prompts = [u.task_prompt for u in generation.units]
    scores = await run_task(
        task=task,
        prompts=prompts,
        model=model_name,
        split="train",
        num_samples=num_samples,
    )
    for i in range(len(generation.units)):
        generation.units[i].fitness = scores[generation.units[i].task_prompt]


async def mutate_units(units: list[Unit], generation: Generation, task: Task):
    new_unit_tasks = [apply_random_mutation(u, generation, task) for u in units]
    new_units = await asyncio.gather(*new_unit_tasks)
    return new_units


async def step_generation(generation: Generation, task: Task):
    """
    Runs the tournament selection and mutates the generation in parallel.
    Assumes fitness has been calculated.
    """
    # get the top unit (for lineage)
    generation.lineage.append(max(generation.units, key=lambda x: x.fitness))

    # shuffle units and pair them up
    random.shuffle(generation.units)
    pairs = list(zip(generation.units[::2], generation.units[1::2]))
    units_to_mutate = []
    units_to_keep = []
    for pair in pairs:
        if pair[0].fitness < pair[1].fitness:
            units_to_mutate.append(pair[0])
            units_to_keep.append(pair[1])
        else:
            units_to_mutate.append(pair[1])
            units_to_keep.append(pair[0])

    new_units = await mutate_units(units_to_mutate, generation, task)

    new_generation = Generation(
        step=generation.step + 1,
        units=new_units + units_to_keep,
        lineage=generation.lineage,
    )
    return new_generation
