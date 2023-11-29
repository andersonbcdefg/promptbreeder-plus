import os
import json
import uuid
import pandas as pd
import requests
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import tempfile
from fastapi import BackgroundTasks
from main import (
    ExperimentTracker,
    PromptBreederConfig,
    initialize,
    calibrate_model,
    run_promptbreeder
)
from task import Task, GRADE_FNS
from typing import Optional

async def get_promptbreeder_config(
    experiment: str,
    task_description: str,
    model: str,
    population: int,
    generations: int,
    input_field: str,
    output_field: str,
    grade_fn: str,
    heuristic: Optional[bool],
    diversity_factor: float,
    file: UploadFile
) -> PromptBreederConfig:
    """
    Version of PromptBreederConfig with strict limits on hyperparams to prevent
    insanely large jobs from eating all our OpenAI credits. Also automatically selects
    some hyperparameters based on the training data.
    """
    if population < 10 or population > 32:
        raise ValueError("Population size must be between 10 and 32")
    if population % 2 != 0:
        raise ValueError("Population size must be even")
    if generations < 4 or generations > 12:
        raise ValueError("Generations must be between 4 and 12")
    if diversity_factor < 0.0 or diversity_factor > 1.0:
        raise ValueError("Diversity factor must be between 0.0 and 1.0")
    
    # save data file & read to dataframe
    contents = await file.read()
    if file.filename.endswith(".csv"):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
            f.write(contents.decode("utf-8"))
            f.flush()
            df = pd.read_csv(f.name)
    elif file.filename.endswith(".jsonl"):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as f:
            f.write(contents.decode("utf-8"))
            f.flush()
            df = pd.read_json(f.name, orient="records", lines=True)
    else:
        raise ValueError("File must be a CSV or JSONL")
    
    # check that the input and output fields are valid
    if input_field not in df.columns:
        raise ValueError(f"Input field {input_field} not found in file")
    if output_field not in df.columns:
        raise ValueError(f"Output field {output_field} not found in file")
    
    # split data
    num_test = int(len(df) * 0.1)
    if num_test < 10:
        num_test = 10
    elif num_test > 100:
        num_test = 100
    num_train = len(df) - num_test
    df = df.sample(frac=1.0, random_state=42)
    df_train = df.iloc[:num_train]
    df_test = df.iloc[num_train:]
    random_string = str(uuid.uuid4())

    # save data
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/train"):
        os.makedirs("data/train")
    if not os.path.exists("data/dev"):
        os.makedirs("data/dev")
    df_train.to_csv(f"data/train/{random_string}_{experiment}.csv", index=False)
    df_test.to_csv(f"data/dev/{random_string}_{experiment}.csv", index=False)



    task = Task(
        name=file.filename,
        description=task_description,
        data_file=f"{random_string}_{experiment}.csv",
        input_column=input_field,
        output_column=output_field,
        negative_column=None,
        grade_fn=GRADE_FNS[grade_fn.replace(" ", "_")],
    )

    config = PromptBreederConfig(
        experiment_name=random_string + "_" + experiment,
        task=task,
        model_name=model,
        population_size=population,
        generations=generations,
        num_scoring_samples=min(40, num_train),
        num_final_scoring_samples=min(100, num_test),
        random_seed=42,
        use_heuristic_model=heuristic,
        diversity_factor=diversity_factor,
    )

    return config



app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def home(request: Request):
    context = {"request": request}
    return templates.TemplateResponse("index.html", context)


async def run_promptbreeder_in_background(
    config: PromptBreederConfig,
    email: str
):
    # create log directory
    if not os.path.exists(f"logs/{config.experiment_name}"):
        os.makedirs(f"logs/{config.experiment_name}")

    # tracker
    tracker = ExperimentTracker(config)

    # Calibrate model, if using
    if config.use_heuristic_model:
        scoring_model = await calibrate_model(tracker)

    # Initialize
    initial_generation = await initialize(tracker.config)

    # Run
    items = await run_promptbreeder(
        initial_generation,
        tracker,
    )

    # get best prompt, baseline, improvement
    items = sorted(items, key=lambda x: x["score"], reverse=True)
    best_prompt = items[0]["prompt"]
    best_score = items[0]["score"]
    baseline_score = [i for i in items if i["prompt"] == config.task.description][0]["score"]

    items = [
        {"prompt": i["prompt"], "score": round(i["score"], 2)} for i in items
    ]


    # TODO: email results to user
    loops_api_key = os.environ.get("LOOPS_API_KEY")
    loops_transactional_id = os.environ.get("LOOPS_TRANSACTIONAL_ID")

    res = requests.post(
        "https://app.loops.so/api/v1/transactional",
        # auth token
        headers={
            "Authorization": "Bearer " + loops_api_key
        },
        json={
            "transactionalId": loops_transactional_id,
            "email": email,
            "dataVariables": {
                "experiment_name": config.experiment_name.split("_", 1)[1],
                "num_generations": config.generations,
                "population_size": config.population_size,
                "best_score": round(best_score, 2),
                "best_prompt": best_prompt,
                "improvement_over_baseline": round(best_score - baseline_score, 2),
                "all_prompts": json.dumps({"prompts": items}, indent=4)
            }
        }
    )
    print(f"Tried to send email to {email}, got response: ", res.json())

    if config.delete_data_on_exit:
        os.remove(f"data/train/{config.experiment_name}.csv")
        os.remove(f"data/dev/{config.experiment_name}.csv")


@app.post("/run_promptbreeder")
async def promptbreeder_endpoint(
    background_tasks: BackgroundTasks,
    email: str = Form(...),
    experiment: str = Form(...),
    task_description: str = Form(...),
    model: str = Form(...),
    population: int = Form(...),
    generations: int = Form(...),
    input_field: str = Form(...),
    output_field: str = Form(...),
    grade_fn: str = Form(...),
    heuristic: Optional[bool] = Form(None),
    diversity_factor: float = Form(...),
    file: UploadFile = File(...),
):
    # Printing out the received data
    config = await get_promptbreeder_config(
        experiment=experiment,
        task_description=task_description,
        model=model,
        population=int(population),
        generations=int(generations),
        input_field=input_field,
        output_field=output_field,
        grade_fn=grade_fn,
        heuristic=bool(heuristic),
        diversity_factor=float(diversity_factor),
        file=file
    )
    print(config)

    background_tasks.add_task(run_promptbreeder_in_background, config, email)

    return {"message": "We've started your PromptBreeder job! We'll email you when it's done."}
