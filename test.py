import asyncio
from main import initialize, PromptBreederConfig, TASKS
from evolution import Generation, Unit, score_generation, step_generation
from rich.console import Console
from rich.status import Status
from heuristic_classifier import ScoringModel
config = PromptBreederConfig.from_yaml("config/test.yaml")

async def test():
    console = Console()
    task = config.task
    scoring_model = ScoringModel()

    with Status("Initializing...", console=console) as status:
        # set up 0-th generation
        status.update("Creating initial task prompts...")
        initial_generation = await initialize(config)
        console.rule("Initial Generation")
        for unit in initial_generation.units:
            console.print(unit)

        # Score the generation
        status.update("Scoring initial generation...")
        await score_generation(initial_generation, task, config.num_scoring_samples, config.model_name)
        scoring_model.update(initial_generation.units)

        console.rule("Initial Generation With Scores")
        for unit in initial_generation.units:
            console.print(unit)

        # Step the generation
        status.update("Stepping initial generation...")
        generation_one = await step_generation(initial_generation, task)

        console.rule("Generation One")
        for unit in generation_one.units:
            console.print(unit)

        # Score the generation
        status.update("Scoring generation one...")
        await score_generation(generation_one, task, config.num_scoring_samples, config.model_name)
        preds = scoring_model.predict(generation_one.units)
        for i, pred in enumerate(preds):
            console.print(f"Predicted fitness: {pred:.3f} | Actual fitness: {generation_one.units[i].fitness}")
        scoring_model.update(generation_one.units)
    
        console.rule("Generation One With Scores")
        for unit in generation_one.units:
            console.print(unit)

        # Step the generation
        status.update("Stepping generation one...")
        generation_two = await step_generation(generation_one, task)

        console.rule("Generation Two")
        for unit in generation_two.units:
            console.print(unit)

        # Score the generation
        status.update("Scoring generation two...")
        await score_generation(generation_two, task, config.num_scoring_samples, config.model_name)

        console.rule("Generation Two With Scores")
        for unit in generation_two.units:
            console.print(unit)

        # Step the generation
        status.update("Stepping generation two...")
        generation_three = await step_generation(generation_two, task)

        console.rule("Generation Three")
        for unit in generation_three.units:
            console.print(unit)
        
        # Score the generation
        status.update("Scoring generation three...")
        await score_generation(generation_three, task, config.num_scoring_samples, config.model_name)

        console.rule("Generation Three With Scores")
        for unit in generation_three.units:
            console.print(unit)

        # Step the generation
        status.update("Stepping generation three...")
        generation_four = await step_generation(generation_three, task)

        console.rule("Generation Four")
        for unit in generation_four.units:
            console.print(unit)

        # Score the generation
        status.update("Scoring generation four...")
        await score_generation(generation_four, task, config.num_scoring_samples, config.model_name)

if __name__ == "__main__":
    initial_generation = asyncio.run(test())
    print(initial_generation)
    