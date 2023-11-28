# A task consists of a set of inputs, expected outputs, an initial prompt,
# a problem description, and a routine to run to grade the task that returns a score
# between 0 (worst) and 1 (best). Several evaluation methods are provided out-of-the-box:
# - String match (exact match)
# - Multiple choice match (model answer begins with same letter as gold answer)
# - Semantic similarity (cosine similarity of embeddings
# - Model-graded (does the model output match the gold answer according to GPT-3.5/4)
import re
import os
import json
from dataclasses import dataclass
from typing import Callable, Literal, Optional
from functools import partial

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from embeddings.onnx_backend import ONNXEmbeddingModel

# import matplotlib.pyplot as plt
from logger import logger
from openai_utils import run_chat_queries_async, StatusTracker
from rich.status import Status

load_dotenv()
MAX_TOKENS_PER_MINUTE = int(os.environ.get("MAX_TOKENS_PER_MINUTE", 400_000))
MAX_REQUESTS_PER_MINUTE = int(os.environ.get("MAX_REQUESTS_PER_MINUTE", 1000))


async def grade_exact_match(inputs, model_outputs, gold_outputs, negative_outputs=None):
    """Grades a task by exact string matching (case-insensitive)"""
    grades = [m.lower() == g.lower() for m, g in zip(model_outputs, gold_outputs)]
    return np.mean(grades), np.array(grades).astype(int).tolist()


async def grade_substring_match(
    inputs, model_outputs, gold_outputs, negative_outputs=None
):
    """
    Grades a task by checking if the gold output occurs in the model output.
    Should work well for integer arithmetic where it's relatively hard to just guess.
    Split by spaces first so that we don't match 42 if it's part of 420, e.g.
    and also removes punctuation (don't use for floating point arithmetic!)
    """
    grades = []
    for m, g in zip(model_outputs, gold_outputs):
        m = m.strip()
        g = str(g).strip()
        if len(m) == 0:
            grades.append(False)
        else:
            correct = False
            pieces = m.split(" ")
            for p in pieces:
                # remove punctuation e.g. commas, periods, etc.
                if re.sub(r"[^\w\s]", "", p).lower() == g.lower():
                    correct = True
                    break
            grades.append(correct)
    
    return np.mean(grades), np.array(grades).astype(int).tolist()

async def grade_multiple_choice(
    inputs, model_outputs, gold_outputs, negative_outputs=None
):
    """Grades a task by multiple choice matching (model answer begins with same letter as gold answer)"""
    grades = []
    for m, g in zip(model_outputs, gold_outputs):
        m = m.strip()
        g = g.strip()
        # also get rid of punctuation in model output
        m = re.sub(r"[^\w\s]", "", m)
        if len(m) == 0:
            grades.append(False)
        elif m.lower()[0] == g.lower()[0]:
            grades.append(True)
        else:
            grades.append(False)
    return np.mean(grades), np.array(grades).astype(int).tolist()

async def grade_semantic_similarity(
    inputs, model_outputs, gold_outputs, negative_outputs=None
):
    embeddings_model = ONNXEmbeddingModel(model_name="bge-micro-v2")
    model_embeddings = embeddings_model.embed_batch(model_outputs, normalize=True)
    gold_embeddings = embeddings_model.embed_batch(gold_outputs, normalize=True)
    similarities = [
        np.dot(m, g) for m, g in zip(model_embeddings, gold_embeddings)
    ]  # cosine similarity (range -1 to 1)
    return (np.mean(similarities) + 1) / 2,  similarities.tolist()


async def grade_semantic_similarity_with_negatives(
    inputs, model_outputs, gold_outputs, negative_outputs
):
    embeddings_model = ONNXEmbeddingModel(model_name="bge-micro-v2")
    model_embeddings = embeddings_model.embed_batch(model_outputs, normalize=True)
    gold_embeddings = embeddings_model.embed_batch(gold_outputs, normalize=True)
    negative_embeddings = embeddings_model.embed_batch(negative_outputs, normalize=True)
    gold_similarities = [
        np.dot(m, g) for m, g in zip(model_embeddings, gold_embeddings)
    ]  # cosine similarity (range -1 to 1)
    negative_similarities = [
        np.dot(m, g) for m, g in zip(model_embeddings, negative_embeddings)
    ]  # cosine similarity (range -1 to 1)
    differences = [g - n for g, n in zip(gold_similarities, negative_similarities)]
    # histogram = plt.hist(differences, bins=8)
    # plt.savefig("histogram.png")
    # plt.close()
    # score is fraction of gold similarities that are greater than negative similarities
    return np.mean([d > 0 for d in differences]), [int(d > 0) for d in differences]

async def grade_json(inputs, model_outputs, gold_outputs, negative_outputs=None):
    """
    Grades a task by exact match. Answer must be in the "answer" field of JSON.
    This will be counted incorrect if model fails to produce valid JSON.
    """
    grades = []
    for m, g in zip(model_outputs, gold_outputs):
        try:
            m = str(json.loads(m.strip())["answer"]).strip()
        except:
            m = ""
        m = m.strip()
        g = g.strip()
        if len(m) == 0:
            grades.append(False)
        elif m.lower() == g.lower():
            grades.append(True)
        else:
            grades.append(False)
    return np.mean(grades), np.array(grades).astype(int).tolist()

async def grade_llm(inputs, model_outputs, gold_outputs, negative_outputs=None):
    """Grades a task by whether the model output matches the gold answer according to GPT-3.5/4"""
    logger.info(f"Grading {len(model_outputs)} outputs with LLM...")
    prompt_template = (
        "Is the candidate response correct according to the reference answer? It doesn't have to "
        "match exactly (e.g. it might be a bit subjective, or the model might get to the right answer "
        "a different way).\n\n"
        "Question: {question}\n\n"
        "Reference response: {reference_answer}\n\n"
        "Candidate response: {candidate_answer}\n\n"
        "Answer simply yes or no: is the candidate answer correct? (yes/no):"
    )

    prompts = [
        prompt_template.format(
            question=inp, reference_answer=gold, candidate_answer=model
        )
        for inp, gold, model in zip(inputs, gold_outputs, model_outputs)
    ]
    prompts = [[{"role": "user", "content": p}] for p in prompts]
    responses, usage = await run_chat_queries_async(
        prompts=prompts,
        max_tokens_per_minute=MAX_TOKENS_PER_MINUTE,
        max_requests_per_minute=MAX_REQUESTS_PER_MINUTE,
        max_new_tokens=2,
    )

    grades = []
    for r in responses:
        if r.strip().lower().startswith("y"):
            grades.append(True)
        elif r.strip().lower().startswith("n"):
            grades.append(False)
        else:
            grades.append(False)

    return np.mean(grades), np.array(grades).astype(int).tolist()


@dataclass
class Task:
    name: str
    description: str
    data_file: str
    input_column: str
    output_column: str
    grade_fn: Callable[[list, list, list], float]
    negative_column: Optional[str] = None
    max_tokens: Optional[int] = 512
    exemplars: Optional[list[dict[str, str]]] = None


gsm8k = Task(
    name="GSM8K",
    description="Solve the math word problem, giving your answer as an arabic numeral.",
    data_file="gsm8k.csv",
    input_column="question",
    output_column="answer",
    max_tokens=512,
    grade_fn=grade_substring_match,
    exemplars = [
        {
            "input": "It takes Matt 2 minutes per problem to do his math homework with a calculator and 5 minutes per problem without a calculator. If Matt's assignment has 20 problems, how much time will using a calculator save?",
            "output": (
                "First find the time difference per problem: 5 minutes/problem - 2 minutes/problem = <<5-2=3>>3 minutes/problem\n"
                "Then multiply that number by the number of problems to find the total time difference: 3 minutes/problem * 20 problems = <<3*20=60>>60 minutes"
            )
        },
        {
            "input": "Jacob is building ladders. One of his clients needs 10 ladders with 50 rungs, and 20 ladders with 60 rungs. Jacob has to charge $2 for every rung. How much does the client have to pay for the ladders?",
            "output": (
                "10 ladders with 50 rungs have a total of 10*50=<<10*50=500>>500 rungs.\n"
                "20 ladders with 60 rungs have a total of 20*60=<<20*60=1200>>1200 rungs.\n"
                "In total there are 500+1200=<<500+1200=1700>>1700 rungs.\n"
                "The client must pay 2*1700=$<<2*1700=3400>>3400."
            )
        },
        {
            "input": "Chris bought 8 movies on DVD for $12 each and 4 movies on Blu-ray for $18 each. What is the average price he paid per movie?",
            "output": (
                "He spent a total of 8 * $12 + 4 * $18 = $<<8*12+4*18=168>>168\n"
                "He bought a total of 8 + 4 = <<8+4=12>>12 movies.\n"
                "The average price is $168 / 12 = $<<168/12=14>>14 per movie."
            )
        },
        {
            "input": "Freddy is 2 years younger than Stephanie. Stephanie is 4 times as old as Job. If Job is 5, how old is Freddy?",
            "output": (
                "Stephanie is 5 * 4 = <<5*4=20>>20 years old\n"
                "Freddy is 20 - 2 = <<20-2=18>>18 years old"
            )
        },
        {
            "input": "John has a large water collection tank.  The tank can hold 200 gallons.  It weighs 80 pounds empty.  A rainstorm fills it to 80% of capacity.  If a gallon of water weighs 8 pounds, how much does it weigh now?",
            "output": (
                "The tank has 200*.8=<<200*.8=160>>160 gallons of water in it\n"
                "That water weighs 160*8=<<160*8=1280>>1280 pounds\n"
                "So in total it weighs 1280+80=<<1280+80=1360>>1360 pounds"
            )
        }
    ]
)

mmlu = Task(
    name="MMLU",
    description="Answer the multiple-choice question, with the letter only.",
    data_file="mmlu.csv",
    input_column="question",
    output_column="answer",
    max_tokens=2,
    grade_fn=grade_multiple_choice,
    exemplars = [
        {
            "input": (
                "Miller is tried for armed robbery of the First Bank of City."
                "At the request of police, the teller who was robbed prepared a sketch "
                "bearing a strong likeness to Miller, but the teller died in an automobile "
                "accident before Miller was arrested. At trial the prosecution offers the sketch. The sketch is\n"
                "(A) admissible as an identification of a person after perceiving him.\n"
                "(B) admissible as past recollection recorded.\n"
                "(C) inadmissible as hearsay, not within any exception. \n"
                "(D) inadmissible as an opinion of the teller"
            ),
            "output": "C"
        },
        {
            "input": (
                "Jenny was a five-year-old girl. One day, while she was shopping with her mother, "
                "she saw a plastic pearl necklace   and loved it so much. So she asked her mother to "
                "buy it for her. Every night, before Jenny went to bed, her dad would read stories to her. "
                'One night, when he finished the story, he asked, ""Jenny, do you love me?"" '
                '""Dad, you know I love you,"" Jenny answered. ""Well, give me your necklace,"" Dad said. '
                '""No, Dad. But you can have my favorite doll."" Several times, when her father asked her to give '
                "him the plastic necklace, Jenny would give him something else instead. One evening, after Jenny's "
                'father read her a story, Jenny said, ""Here, Dad."" She put her plastic pearl necklace into her '
                "father's hand. Her father hold the necklace in one hand and opened the other hand. There was a real "
                "pearl necklace in it. He had had it for a long time, and waited for Jenny to give up the cheap one so "
                "that he could give her the real one. So, don't be _ . If we are generous  , maybe we will get something "
                "better. What did Jenny do when her father asked her for the necklace for the first time?\n"
                "(A) She asked him to buy a doll for her.\n"
                "(B) She gave him her favorite doll.\n"
                "(C) She said she didn't love him at all.\n"
                "(D) She gave the necklace to him."
            ),
            "output": "B"
        },
    ]
)

theoremqa = Task(
    name="TheoremQA",
    description="Answer the following question.",
    data_file="theoremqa.csv",
    input_column="question",
    output_column="answer",
    max_tokens=512,
    grade_fn=grade_llm,
)

logiqa = Task(
    name="LogiQA",
    description="Answer the following reasoning question, with the letter only.",
    data_file="logiqa.csv",
    input_column="question",
    output_column="answer",
    max_tokens=2,
    grade_fn=grade_multiple_choice,
)

truthfulqa = Task(
    name="TruthfulQA",
    description="Answer the following question, making sure to stay factual.",
    data_file="truthfulqa.csv",
    input_column="question",
    output_column="answer",
    negative_column="negative",
    max_tokens=512,
    grade_fn=grade_semantic_similarity_with_negatives,
)


async def run_task(
    task: Task,
    prompts: list[str],
    model: str = "gpt-3.5-turbo",
    split: Literal["train", "dev", "test"] = "train",
    num_samples: int = 50,
    seed: Optional[int] = None,
    status: Optional[Status] = None,
    base_status: Optional[str] = None
):
    """
    Runs task with list of provided prompts, and returns score for each prompt.
    """
    if seed is None:
        seed = np.random.randint(0, 1_000_000)
    df = pd.read_csv(f"data/{split}/{task.data_file}").sample(
        num_samples, random_state=seed
    )
    queries = []
    for prompt in prompts:
        for inp in df[task.input_column]:
            queries.append([{"role": "user", "content": "INSTRUCTION: " + prompt + "\n\nTASK: " + inp}])

    def cb(task_id: int, messages: list, result: str, status_tracker: StatusTracker, base_status: str, num_queries: int):
        num_finished = status_tracker.num_tasks_succeeded
        new_status = base_status + f" ({num_finished}/{num_queries})"
        status.update(new_status)

    responses, usage = await run_chat_queries_async(
        prompts=queries,
        max_tokens_per_minute=100_000,
        max_requests_per_minute=500,
        model_name=model,
        max_new_tokens=task.max_tokens,
        callback=partial(cb, base_status=base_status, num_queries=len(queries))
    )

    # replace None with empty string -- robust to occasional model failures
    responses = [r if r is not None else "" for r in responses]

    # partition responses into groups of num_samples
    responses = [
        responses[i : i + num_samples] for i in range(0, len(responses), num_samples)
    ]

    # compute score for each prompt -- TODO: parallelize if the grading function is llm
    scored_prompts = {}
    positive_exemplars = []
    negative_exemplars = []
    for i, prompt in enumerate(prompts):
        inputs = df[task.input_column].tolist()
        score, grades = await task.grade_fn(
            inputs=df[task.input_column].tolist(),
            model_outputs=responses[i],
            gold_outputs=df[task.output_column].tolist(),
            negative_outputs=df[task.negative_column].tolist()
            if task.negative_column
            else None,
        )
        # if grades are binary, correct answers are positive exemplars and incorrect answers are negative exemplars
        if len(set(grades)) <= 2:
            for inp, model_output, grade in zip(inputs, responses[i], grades):
                if grade == 1:
                    positive_exemplars.append({"input": inp, "prompt": prompt, "output": model_output, "grade": grade})
                else:
                    negative_exemplars.append({"input": inp, "prompt": prompt, "output": model_output, "grade": grade})
        # otherwise, the top 5% of grades are positive exemplars and the bottom 5% are negative exemplars
        else:
            for inp, model_output, grade in zip(inputs, responses[i], grades):
                if grade >= np.percentile(grades, 95):
                    positive_exemplars.append({"input": inp, "prompt": prompt, "output": model_output, "grade": grade})
                elif grade <= np.percentile(grades, 5):
                    negative_exemplars.append({"input": inp, "prompt": prompt, "output": model_output, "grade": grade})

        scored_prompts[prompt] = score

    return {
        "scored_prompts": scored_prompts,
        "positive_exemplars": positive_exemplars,
        "negative_exemplars": negative_exemplars,
    }


TASKS = {
    "gsm8k": gsm8k,
    "mmlu": mmlu,
    "theoremqa": theoremqa,
    "logiqa": logiqa,
    "truthfulqa": truthfulqa,
}


async def test(model="gpt-3.5-turbo"):
    for task in TASKS.values():
        result = await run_task(
            task=task,
            prompts=[
                task.description,
                "Please state the wrong answer to the question on purpose.",
            ],
            model=model,
            split="train",
            num_samples=50,
            seed=42,
        )
    print("done")
