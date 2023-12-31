# import asyncio
import copy
import json
import os
import random
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import fire
import yaml
from dotenv import load_dotenv
from rich.status import Status

from .evolution import Generation, Unit, score_generation, step_generation
from .logger import logger
from .openai_utils import instructions_to_message_lists, run_chat_queries_async
from .prompts import MUTATION_PROMPTS, THINKING_STYLES, get_meta_mutation_prompt
from .scoring_model import ScoringModel
from .task import TASKS, Task, run_task

load_dotenv()

@dataclass
class PromptBreederConfig:
    experiment_name: str
    task: Union[str, Task]
    model_name: Literal["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4", "mistral"]
    generations: int
    population_size: int
    num_scoring_samples: int
    num_final_scoring_samples: int
    random_seed: int
    use_heuristic_model: bool = True
    few_shot_examples: int = 0
    diversity_factor: Optional[float] = 0.5
    oversample_factor: Optional[int] = 4
    max_requests_per_minute: int = field(default_factory=lambda: int(os.environ.get("MAX_REQUESTS_PER_MINUTE", 1000)))
    max_tokens_per_minute: int = field(default_factory=lambda: int(os.environ.get("MAX_TOKENS_PER_MINUTE", 400_000)))
    delete_data_on_exit: bool = False

    @classmethod
    def from_yaml(cls, file_name: str):
        return cls(**yaml.safe_load(open(file_name)))
    
    def __post_init__(self):
        if isinstance(self.task, str):
            if self.task not in TASKS:
                raise ValueError(f"Invalid task name: {self.task}")
            else:
                self.task = TASKS[self.task]
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

    def to_dict(self):
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_") and k != "task"}
        d["task"] = self.task.name
        return d

@dataclass
class ExperimentTracker:
    """
    Tracks all objects over the course of an experiment (so we don't have to log everything to disk on server,
    or pass around multiple objects between functions).
    """
    config: PromptBreederConfig
    scoring_model: Optional[ScoringModel] = None
    generations: list[Generation] = field(default_factory=list)
    scored_prompts: list[dict] = field(default_factory=list)
    final_prompts: list[dict] = field(default_factory=list)
    positive_exemplars: list[dict] = field(default_factory=list)
    negative_exemplars: list[dict] = field(default_factory=list)

    def save(self, file_name: str):
        obj = {}
        obj["config"] = self.config.to_dict()
        obj["final_prompts"] = self.final_prompts
        obj["generations"] = [g.to_dict() for g in self.generations]
        obj["scored_prompts"] = self.scored_prompts
        obj["positive_exemplars"] = self.positive_exemplars
        obj["negative_exemplars"] = self.negative_exemplars

        with open(file_name, "w") as f:
            json.dump(obj, f, indent=2)

async def calibrate_model(
    tracker: ExperimentTracker,
):
    print("Calibrating scoring model with random prompts...")
    task = tracker.config.task
    if tracker.scoring_model is None:
        tracker.scoring_model = ScoringModel(diversity_factor=tracker.config.diversity_factor)
    initial_task_meta_prompts = []
    for i in range(150):
        mutation_prompt = random.choice(MUTATION_PROMPTS)
        thinking_style = random.choice(THINKING_STYLES)
        meta_prompt = get_meta_mutation_prompt(mutation_prompt, task.description, thinking_style=thinking_style)
        initial_task_meta_prompts.append(meta_prompt)

    with Status("Calibrating scoring model...") as status:
        calibration_prompts, usage = await run_chat_queries_async(
            prompts=instructions_to_message_lists(initial_task_meta_prompts),
            max_tokens_per_minute=tracker.config.max_tokens_per_minute,
            max_requests_per_minute=tracker.config.max_requests_per_minute,
            json_mode=True,
            model_name="gpt-3.5-turbo",
            max_new_tokens=512,
            max_attempts=10,
            callback=lambda task_id, messages, result, status_tracker: status.update(
                f"Calibrating scoring model... ({status_tracker.num_tasks_succeeded}/{len(initial_task_meta_prompts)})"
            ),
        )
        parsed = []
        for p in calibration_prompts:
            try:
                parsed.append(json.loads(p)["instruction"])
            except:
                parsed.append(task.description)
        calibration_prompts = parsed
        print("A few examples of prompts for calibration: \n- " + "\n- ".join(calibration_prompts[:10]))
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
        result = await score_generation(
            calibration_generation,
            task,
            num_samples=6,
            model_name=tracker.config.model_name,
            status=status,
        )

        tracker.generations.append(copy.deepcopy(calibration_generation))
        tracker.positive_exemplars.extend(result["positive_exemplars"])
        tracker.negative_exemplars.extend(result["negative_exemplars"])
        tracker.scored_prompts.extend([
            {"prompt": p, "score": s} for p, s in result["scored_prompts"].items()    
        ])
        tracker.scoring_model.update(calibration_generation.units, log_dir=f"logs/{tracker.config.experiment_name}", status=status)
        assert tracker.scoring_model.Xs is not None, "Xs should not be None after calibration"
        tracker.save(f"logs/{tracker.config.experiment_name}/experiment.json")

async def initialize(
    config: PromptBreederConfig,
):    
    task = config.task
    random.seed(config.random_seed)
    initial_task_meta_prompts = []
    for i in range(config.population_size):
        mutation_prompt = random.choice(MUTATION_PROMPTS)
        thinking_style = random.choice(THINKING_STYLES)
        meta_prompt = get_meta_mutation_prompt(mutation_prompt, task.description, thinking_style=thinking_style)
        initial_task_meta_prompts.append(meta_prompt)

    with Status("Creating initial task prompts...") as status:
        initial_task_prompts, usage = await run_chat_queries_async(
            prompts=instructions_to_message_lists(initial_task_meta_prompts),
            max_tokens_per_minute=config.max_tokens_per_minute,
            max_requests_per_minute=config.max_requests_per_minute,
            json_mode=True,
            model_name="gpt-3.5-turbo",
            max_new_tokens=512,
            max_attempts=10,
            callback=lambda task_id, messages, result, status_tracker: status.update(
                f"Creating initial task prompts... ({status_tracker.num_tasks_succeeded}/{len(initial_task_meta_prompts)})"
            ),
        )

        parsed = []
        for p in initial_task_prompts:
            try:
                parsed.append(json.loads(p)["instruction"])
            except:
                parsed.append(task.description)
        initial_task_prompts = parsed

        print("Initial task prompts:\n-" + "\n -".join(initial_task_prompts))

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
    tracker: ExperimentTracker,
    status: Optional[Status] = None
):
    config = tracker.config
    task = config.task
    
    top_prompts = set()
    # get the top prompts from each generation
    for generation in tracker.generations:
        units = sorted(generation.units, key=lambda x: x.fitness, reverse=True)
        top_prompts.update([u.task_prompt for u in units[:2]])
    # also add the initial task description--we can measure how much we improved performance
    top_prompts.add(task.description)
    top_prompts = list(top_prompts)

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
    items = []
    for prompt, score in scores.items():
        logger.log_to_file(f"Score: {score}, Prompt: {prompt}")
        print(f"Score: {score}, Prompt: {prompt}\n\n====================\n\n")
        items.append({"prompt": prompt, "score": score})

    # get the best score/prompt and compare to baseline
    best_prompt = max(items, key=lambda x: x["score"])
    best_score = best_prompt["score"]
    original_prompt = next(i for i in items if i["prompt"] == task.description)
    original_score = original_prompt["score"]
    improvement = best_score - original_score
    print(f"Best prompt: {best_prompt['prompt']}")
    print(f"Best score: {best_score}")
    print("Improved {} over baseline prompt!".format(improvement))

    # save the results
    with open(f"logs/{config.experiment_name}/final_ranking.json", "w") as f:
        json.dump(items, f, indent=2)

    return items

    

async def run_promptbreeder(
    initial_generation: Generation,
    tracker: ExperimentTracker,
):
    task = tracker.config.task
    config = tracker.config

    # Run the evolution
    with Status("Starting PromptBreeder...") as status:
        current_generation = initial_generation
        for i in range(config.generations):
            status.update(f"Scoring generation {i}...")
            result = await score_generation(
                current_generation,
                task,
                num_samples=config.num_scoring_samples,
                model_name=config.model_name,
                status=status,
            )
            tracker.positive_exemplars.extend(result["positive_exemplars"])
            tracker.negative_exemplars.extend(result["negative_exemplars"])
            tracker.scored_prompts.extend([
                {"prompt": p, "score": s} for p, s in result["scored_prompts"].items()    
            ])
            tracker.generations.append(copy.deepcopy(current_generation))
            print(f"=== GENERATION {i} ===")
            print(f"Best score: {max([u.fitness for u in current_generation.units]):.3f}")
            print(f"Average score: {sum([u.fitness for u in current_generation.units]) / len(current_generation.units):.3f}")
            if tracker.scoring_model is not None:
                metrics = tracker.scoring_model.update(current_generation.units, log_dir=f"logs/{config.experiment_name}", status=status)
                if metrics is not None:
                    logger.info(f"Metrics: {metrics}")
            tracker.save(f"logs/{config.experiment_name}/experiment.json")
            status.update(f"Evolving generation {i + 1}...")
            provide_exemplars = True if len(tracker.positive_exemplars) > 0 and len(tracker.negative_exemplars) > 0 else False
            current_generation = await step_generation(
                current_generation, 
                task, 
                exemplars=({
                    "positive": tracker.positive_exemplars,
                    "negative": tracker.negative_exemplars,
                } if provide_exemplars else None),
                scoring_model=tracker.scoring_model, 
                oversample_factor = config.oversample_factor
            )

        # score the last generation
        status.update("Scoring final generation...")
        result = await score_generation(
            current_generation,
            task,
            num_samples=config.num_scoring_samples,
            model_name=config.model_name,
            status=status,
        )
        tracker.positive_exemplars.extend(result["positive_exemplars"])
        tracker.negative_exemplars.extend(result["negative_exemplars"])
        tracker.scored_prompts.extend([
            {"prompt": p, "score": s} for p, s in result["scored_prompts"].items()    
        ])
        tracker.generations.append(copy.deepcopy(current_generation))
        print ("=== FINAL GENERATION ===")
        print(f"Best score: {max([u.fitness for u in current_generation.units]):.3f}")
        print(f"Average score: {sum([u.fitness for u in current_generation.units]) / len(current_generation.units):.3f}")
        if tracker.scoring_model is not None:
            metrics = tracker.scoring_model.update(current_generation.units, log_dir=f"logs/{config.experiment_name}", status=status)
            if metrics is not None:
                logger.info(f"Metrics: {metrics}")
        
        # rank the final prompts
        items = await rank_final_prompts(tracker, status=status)
        tracker.final_prompts = items

        tracker.save(f"logs/{config.experiment_name}/experiment.json")
    
    return items

async def main(config: Union[PromptBreederConfig, str]):
    # Set up config
    if isinstance(config, str):
        config = PromptBreederConfig.from_yaml(config)

    # create log directory
    if not os.path.exists(f"logs/{config.experiment_name}"):
        os.makedirs(f"logs/{config.experiment_name}")

    # set up experiment tracker
    tracker = ExperimentTracker(config=config)

    # Calibrate model, if using
    if config.use_heuristic_model:
        await calibrate_model(tracker)

    # Initialize
    initial_generation = await initialize(tracker.config)

    # Run
    items = await run_promptbreeder(
        initial_generation, 
        tracker
    )


if __name__ == "__main__":
    fire.Fire(main)
