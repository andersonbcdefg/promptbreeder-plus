# import asyncio
import fire
import random
from dataclasses import dataclass
from typing import Literal, Union

import yaml
from rich.status import Status
from constants import MUTATION_PROMPTS, THINKING_STYLES
from evolution import Generation, Unit, score_generation, step_generation
from logger import logger
from openai_utils import instructions_to_message_lists, run_chat_queries_async
from task import TASKS
from heuristic_classifier import ScoringModel


@dataclass
class PromptBreederConfig:
    task_name: str
    model_name: Literal["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4", "mistral"]
    generations: int
    population_size: int
    num_scoring_samples: int
    random_seed: int

    @classmethod
    def from_yaml(cls, file_name: str):
        return cls(**yaml.safe_load(open(file_name)))


async def initialize(
    config: PromptBreederConfig,
):
    task = TASKS[config.task_name]
    random.seed(config.random_seed)
    initial_task_meta_prompts = []
    for i in range(config.population_size):
        mutation_prompt = random.choice(MUTATION_PROMPTS)
        thinking_style = random.choice(THINKING_STYLES)
        initial_task_meta_prompts.append(
            f"{mutation_prompt} {thinking_style}\n\nINSTRUCTION: {task.description}"
        )
    with Status("Creating initial task prompts...") as status:
        initial_task_prompts, usage = await run_chat_queries_async(
            prompts=instructions_to_message_lists(initial_task_meta_prompts),
            max_tokens_per_minute=400_000,
            max_requests_per_minute=1_000,
            model_name="gpt-3.5-turbo",
            max_new_tokens=512,
        )
        initial_task_prompts = [
            p if p is not None else task.description for p in initial_task_prompts
        ]

        status.update("Creating initial mutation prompts...")
        initial_mutation_prompts = random.sample(MUTATION_PROMPTS, config.population_size)
        initial_generation = Generation(
            step=0,
            units=[
                Unit(task_prompt=t, mutation_prompt=m, origin="initial")
                for t, m in zip(initial_task_prompts, initial_mutation_prompts)
            ],
            lineage=[],
        )
        return initial_generation

async def run_promptbreeder(
    initial_generation: Generation,
    config: PromptBreederConfig,
):
    task = TASKS[config.task_name]
    scoring_model = ScoringModel()

    # Run the evolution
    with Status("Starting PromptBreeder...") as status:
        current_generation = initial_generation
        for i in range(config.generations):
            status.update(f"Scoring generation {i}...")
            await score_generation(
                current_generation,
                task,
                num_samples=config.num_scoring_samples,
                model_name=config.model_name,
            )
            mae = scoring_model.update(current_generation.units)
            # current_generation.save_to_file
            status.update(f"Evolving generation {i + 1}...")
            current_generation = await step_generation(current_generation, task)

        # score the last generation
        status.update(f"Scoring final generation...")
        logger.log_to_file("=== FINAL GENERATION ===")
        await score_generation(
            current_generation,
            task,
            num_samples=config.num_scoring_samples,
            model_name=config.model_name,
        )

    # print the top prompts and their fitness
    logger.info("Top prompts:")
    units = sorted(current_generation.units, key=lambda u: u.fitness, reverse=True)
    for unit in units:
        logger.info(f"Origin: {unit.origin} | Fitness: {unit.fitness}")
        logger.info(f"Task prompt: {unit.task_prompt}")
        logger.info("=====================")

    return current_generation


async def main(config: Union[PromptBreederConfig, str]):
    # Set up config
    if isinstance(config, str):
        config = PromptBreederConfig.from_yaml(config)

    # Initialize
    initial_generation = await initialize(config)

    # Run
    results = await run_promptbreeder(initial_generation, config)


if __name__ == "__main__":
    fire.Fire(main)
