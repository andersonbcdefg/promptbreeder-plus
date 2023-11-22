# A task consists of a set of inputs, expected outputs, an initial prompt,
# a problem description, and a routine to run to grade the task that returns a score
# between 0 (worst) and 1 (best). Several evaluation methods are provided out-of-the-box:
# - String match (exact match)
# - Multiple choice match (model answer begins with same letter as gold answer)
# - Semantic similarity (cosine similarity of embeddings
# - Model-graded (does the model output match the gold answer according to GPT-3.5/4)
import re
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from embeddings.onnx_backend import ONNXEmbeddingModel

# import matplotlib.pyplot as plt
from logger import logger
from openai_utils import run_chat_queries_async

load_dotenv()


async def grade_exact_match(inputs, model_outputs, gold_outputs, negative_outputs=None):
    """Grades a task by exact string matching (case-insensitive)"""
    return np.mean(
        [m.lower() == g.lower() for m, g in zip(model_outputs, gold_outputs)]
    )


async def grade_substring_match(
    inputs, model_outputs, gold_outputs, negative_outputs=None
):
    """
    Grades a task by checking if the gold output occurs in the model output.
    Should work well for integer arithmetic where it's relatively hard to just guess.
    Split by spaces first so that we don't match 42 if it's part of 420, e.g.
    and also removes punctuation (don't use for floating point arithmetic!)
    """
    num_correct = 0
    for m, g in zip(model_outputs, gold_outputs):
        m = m.strip()
        g = str(g).strip()
        if len(m) == 0:
            continue
        pieces = m.split(" ")
        for p in pieces:
            # remove punctuation e.g. commas, periods, etc.
            if re.sub(r"[^\w\s]", "", p).lower() == g.lower():
                num_correct += 1
                break

    return num_correct / len(model_outputs)


async def grade_multiple_choice(
    inputs, model_outputs, gold_outputs, negative_outputs=None
):
    """Grades a task by multiple choice matching (model answer begins with same letter as gold answer)"""
    num_correct = 0
    for m, g in zip(model_outputs, gold_outputs):
        m = m.strip()
        g = g.strip()
        # also get rid of punctuation in model output
        m = re.sub(r"[^\w\s]", "", m)
        if len(m) == 0:
            continue
        if m.lower()[0] == g.lower()[0]:
            num_correct += 1

    return num_correct / len(model_outputs)


async def grade_semantic_similarity(
    inputs, model_outputs, gold_outputs, negative_outputs=None
):
    embeddings_model = ONNXEmbeddingModel(model_name="bge-micro-v2")
    model_embeddings = embeddings_model.embed_batch(model_outputs, normalize=True)
    gold_embeddings = embeddings_model.embed_batch(gold_outputs, normalize=True)
    similarities = [
        np.dot(m, g) for m, g in zip(model_embeddings, gold_embeddings)
    ]  # cosine similarity (range -1 to 1)
    return (np.mean(similarities) + 1) / 2  # normalize to 0 to 1


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
    return np.mean([d > 0 for d in differences])


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
        max_tokens_per_minute=400_000,
        max_requests_per_minute=1_000,
        max_new_tokens=2,
    )

    num_correct = 0
    for r in responses:
        if r.strip().lower().startswith("y"):
            num_correct += 1

    return num_correct / len(responses)


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


gsm8k = Task(
    name="GSM8K",
    description="Solve the math word problem, giving your answer as an arabic numeral.",
    data_file="gsm8k.csv",
    input_column="question",
    output_column="answer",
    max_tokens=512,
    grade_fn=grade_substring_match,
)

mmlu = Task(
    name="MMLU",
    description="Answer the multiple-choice question, with the letter only.",
    data_file="mmlu.csv",
    input_column="question",
    output_column="answer",
    max_tokens=2,
    grade_fn=grade_multiple_choice,
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
            queries.append([{"role": "user", "content": prompt + "\n\n" + inp}])

    responses, usage = await run_chat_queries_async(
        prompts=queries,
        max_tokens_per_minute=100_000,
        max_requests_per_minute=500,
        model_name=model,
        max_new_tokens=task.max_tokens,
    )

    # replace None with empty string
    responses = [r if r is not None else "" for r in responses]

    # partition responses into groups of num_samples
    responses = [
        responses[i : i + num_samples] for i in range(0, len(responses), num_samples)
    ]

    # compute score for each prompt -- TODO: parallelize if the grading function is llm
    result = {}
    for i, prompt in enumerate(prompts):
        score = await task.grade_fn(
            inputs=df[task.input_column].tolist(),
            model_outputs=responses[i],
            gold_outputs=df[task.output_column].tolist(),
            negative_outputs=df[task.negative_column].tolist()
            if task.negative_column
            else None,
        )
        result[prompt] = score

    for prompt, score in result.items():
        logger.log_to_file(f"Score: {score}, Prompt: {prompt}")
    return result


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
