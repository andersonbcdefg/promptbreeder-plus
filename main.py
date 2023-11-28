# import asyncio
import os
import json
import fire
import random
from dataclasses import dataclass
from typing import Literal, Union, Optional

import yaml
from rich.status import Status
from prompts import MUTATION_PROMPTS, THINKING_STYLES
from evolution import Generation, Unit, score_generation, step_generation
from logger import logger
from openai_utils import instructions_to_message_lists, run_chat_queries_async
from task import TASKS, run_task
from heuristic_classifier import ScoringModel
from dotenv import load_dotenv
load_dotenv()

MAX_TOKENS_PER_MINUTE = int(os.environ.get("MAX_TOKENS_PER_MINUTE", 400_000))
MAX_REQUESTS_PER_MINUTE = int(os.environ.get("MAX_REQUESTS_PER_MINUTE", 1000))

@dataclass
class PromptBreederConfig:
    experiment_name: str
    task_name: str
    model_name: Literal["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4", "mistral"]
    generations: int
    population_size: int
    num_scoring_samples: int
    num_final_scoring_samples: int
    random_seed: int
    use_heuristic_model: bool = True
    diversity_factor: Optional[float] = 0.5
    oversample_factor: Optional[int] = 1

    @classmethod
    def from_yaml(cls, file_name: str):
        return cls(**yaml.safe_load(open(file_name)))
    
    def __post_init__(self):
        if self.task_name not in TASKS:
            raise ValueError(f"Invalid task name: {self.task_name}")
        if self.model_name not in ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4", "mistral"]:
            raise ValueError(f"Invalid model name: {self.model_name}")
        if self.population_size % 2 != 0:
            raise ValueError(f"Population size must be even, but got {self.population_size}")
        if self.oversample_factor and self.oversample_factor < 1:
            raise ValueError(f"Oversample factor must be at least 1, but got {self.oversample_factor}")
        if self.oversample_factor and not self.use_heuristic_model:
            logger.warning("Oversample factor is set, but heuristic model is not being used. Oversampling will be ignored.")
        if self.diversity_factor and self.diversity_factor < 0 or self.diversity_factor > 1:
            raise ValueError(f"Diversity factor must be between 0 and 1, but got {self.diversity_factor}")
        if self.diversity_factor and not self.use_heuristic_model:
            logger.warning("Diversity factor is set, but heuristic model is not being used. Diversity will be ignored.")


async def calibrate_model(
    config: PromptBreederConfig,
):
    print("Calibrating scoring model with random prompts...")
    task = TASKS[config.task_name]
    scoring_model = ScoringModel()
    initial_task_meta_prompts = []
    for i in range(225):
        mutation_prompt = random.choice(MUTATION_PROMPTS)
        thinking_style = random.choice(THINKING_STYLES)
        initial_task_meta_prompts.append(
            f"{mutation_prompt} {thinking_style}\n\nINSTRUCTION: {task.description}\n\nRespond with JUST your modified instruction (no need to label it as INSTRUCTION)."
        )
    with Status("Calibrating scoring model...") as status:
        calibration_prompts, usage = await run_chat_queries_async(
            prompts=instructions_to_message_lists(initial_task_meta_prompts),
            max_tokens_per_minute=MAX_TOKENS_PER_MINUTE,
            max_requests_per_minute=MAX_REQUESTS_PER_MINUTE,
            model_name="gpt-3.5-turbo",
            max_new_tokens=512,
            max_attempts=10,
            callback=lambda task_id, messages, result, status_tracker: status.update(
                f"Calibrating scoring model... ({status_tracker.num_tasks_succeeded}/{len(initial_task_meta_prompts)})"
            ),
        )
        calibration_prompts = [
            p for p in calibration_prompts if p is not None
        ]
        mutation_prompts = ["" for _ in calibration_prompts]
        calibration_generation = Generation(
            step=0,
            units=[
                Unit(task_prompt=t, mutation_prompt=m, origin="calibration")
                for t, m in zip(calibration_prompts, mutation_prompts)
            ],
            lineage=[],
        )
        print("Scoring calibration prompts...")
        await score_generation(
            calibration_generation,
            task,
            num_samples=8,
            model_name=config.model_name,
            status=status,
        )
        
        scoring_model.update(calibration_generation.units, log_dir=f"logs/{config.experiment_name}", status=status)
        assert scoring_model.Xs is not None, "Xs should not be None after calibration"
        
        return scoring_model

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
            f"{mutation_prompt} {thinking_style}\n\nINSTRUCTION: {task.description}\n\nRespond with JUST your modified instruction (no need to label it as INSTRUCTION)."
        )
    with Status("Creating initial task prompts...") as status:
        initial_task_prompts, usage = await run_chat_queries_async(
            prompts=instructions_to_message_lists(initial_task_meta_prompts),
            max_tokens_per_minute=MAX_TOKENS_PER_MINUTE,
            max_requests_per_minute=MAX_REQUESTS_PER_MINUTE,
            model_name="gpt-3.5-turbo",
            max_new_tokens=512,
            max_attempts=10,
            callback=lambda task_id, messages, result, status_tracker: status.update(
                f"Creating initial task prompts... ({status_tracker.num_tasks_succeeded}/{len(initial_task_meta_prompts)})"
            ),
        )
        initial_task_prompts = [
            p if p is not None else task.description for p in initial_task_prompts
        ]

        status.update("Creating initial mutation prompts...")
        initial_mutation_prompts = random.choices(MUTATION_PROMPTS, k=config.population_size)
        initial_generation = Generation(
            step=0,
            units=[
                Unit(task_prompt=t, mutation_prompt=m, origin="initial")
                for t, m in zip(initial_task_prompts, initial_mutation_prompts)
            ],
            lineage=[],
        )
        return initial_generation

async def rank_final_prompts(
    config: PromptBreederConfig
):
    task = TASKS[config.task_name]
    with Status("Ranking final prompts...") as status:
        top_prompts = set()
        for file in os.listdir(f"logs/{config.experiment_name}"):
            if file.startswith("gen_") and file.endswith(".json"):
                units = json.load(open(f"logs/{config.experiment_name}/{file}"))["units"]
                top_prompts.update([u["task_prompt"] for u in units[:2]])
        # evaluate the prompts
        print(f"Evaluating {len(top_prompts)} prompts...")
        result = await run_task(
            task,
            list(top_prompts),
            split="dev",
            model=config.model_name,
            num_samples=config.num_final_scoring_samples,
            status=status,
            base_status="Evaluating top prompts on held-out dev set...",
        )
        scores = result["scored_prompts"]
        
        for prompt, score in scores.items():
            logger.log_to_file(f"Score: {score}, Prompt: {prompt}")
            print(f"Score: {score}, Prompt: {prompt}\n\n====================\n\n")

async def run_promptbreeder(
    initial_generation: Generation,
    scoring_model: Optional[ScoringModel],
    config: PromptBreederConfig,
):
    task = TASKS[config.task_name]

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
                status=status,
            )
            print(f"=== GENERATION {i} ===")
            print(f"Best score: {max([u.fitness for u in current_generation.units]):.3f}")
            print(f"Average score: {sum([u.fitness for u in current_generation.units]) / len(current_generation.units):.3f}")
            if scoring_model is not None:
                metrics = scoring_model.update(current_generation.units, log_dir=f"logs/{config.experiment_name}", status=status)
                if metrics is not None:
                    logger.info(f"Metrics: {metrics}")
            current_generation.save_to_file(f"logs/{config.experiment_name}/gen_{i}.json")
            status.update(f"Evolving generation {i + 1}...")
            current_generation = await step_generation(
                current_generation, 
                task, 
                scoring_model, 
                oversample_factor = config.oversample_factor
            )

        # score the last generation
        status.update(f"Scoring final generation...")
        await score_generation(
            current_generation,
            task,
            num_samples=config.num_scoring_samples,
            model_name=config.model_name,
            status=status,
        )
        print (f"=== FINAL GENERATION ===")
        print(f"Best score: {max([u.fitness for u in current_generation.units]):.3f}")
        print(f"Average score: {sum([u.fitness for u in current_generation.units]) / len(current_generation.units):.3f}")
        if scoring_model is not None:
            metrics = scoring_model.update(current_generation.units, log_dir=f"logs/{config.experiment_name}", status=status)
            if metrics is not None:
                logger.info(f"Metrics: {metrics}")
        current_generation.save_to_file(f"logs/{config.experiment_name}/gen_{i + 1}_final.json")

        # rank the final prompts
        await rank_final_prompts(config)

    


async def main(config: Union[PromptBreederConfig, str]):
    # Set up config
    if isinstance(config, str):
        config = PromptBreederConfig.from_yaml(config)

    # create log directory
    if not os.path.exists(f"logs/{config.experiment_name}"):
        os.makedirs(f"logs/{config.experiment_name}")

    # Calibrate model, if using
    if config.use_heuristic_model:
        scoring_model = await calibrate_model(config)
    else:
        scoring_model = None

    # Initialize
    initial_generation = await initialize(config)

    # Run
    results = await run_promptbreeder(
        initial_generation, 
        scoring_model,
        config
    )


if __name__ == "__main__":
    fire.Fire(main)
