import asyncio
import json

### Code here adapted from openai cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import aiohttp
import tiktoken
import xxhash
from tqdm.auto import tqdm

from .logger import logger

## TODO: Make a Queue where we can append API requests as-needed in other parts of the application, and they can be
## processed in the background in parallel.


@dataclass
class StatusTracker:
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    time_of_last_rate_limit_error: int = 0
    total_requests = 0


@dataclass
class SqliteCache:
    path: str

    def __post_init__(self):
        self.conn = sqlite3.connect(self.path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY, content TEXT)"
        )
        self.conn.commit()

    @staticmethod
    def get_hash(messages):
        hasher = xxhash.xxh64()
        hasher.update(json.dumps(messages).encode())
        hash_key = hasher.hexdigest()
        return hash_key

    def get_from_cache(self, messages):
        hash_key = self.get_hash(messages)
        self.cursor.execute("SELECT content FROM cache WHERE hash=?", (hash_key,))
        return self.cursor.fetchone()

    def set_to_cache(self, messages, content):
        hash_key = self.get_hash(messages)
        try:
            self.cursor.execute(
                "INSERT INTO cache (hash, content) VALUES (?, ?)", (hash_key, content)
            )
        # if failed due to unique constraint, update instead
        except sqlite3.IntegrityError:
            self.cursor.execute(
                "UPDATE cache SET content=? WHERE hash=?", (content, hash_key)
            )
        except Exception as e:
            logger.error(f"Error setting cache: {json.dumps(e)}")
        self.conn.commit()


@dataclass
class APIModel:
    name: str
    api_base: str
    api_key_env_var: str
    request_timeout: int

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
GPT3_TURBO = APIModel(
    name="gpt-3.5-turbo-1106",
    api_base="https://api.openai.com/v1",
    api_key_env_var="OPENAI_API_KEY",
    request_timeout=20,
)
GPT4_TURBO = APIModel(
    name="gpt-4-1106-preview",
    api_base="https://api.openai.com/v1",
    api_key_env_var="OPENAI_API_KEY",
    request_timeout=45,
)
GPT4 = APIModel(
    name="gpt-4",
    api_base="https://api.openai.com/v1",
    api_key_env_var="OPENAI_API_KEY",
    request_timeout=45,
)
MISTRAL = APIModel(
    name="mistralai/Mistral-7B-Instruct-v0.1",
    api_base="https://api.endpoints.anyscale.com/v1",
    api_key_env_var="ANYSCALE_API_KEY",
    request_timeout=45,
)


@dataclass
class APIRequest:
    task_id: int
    messages: list[dict]
    attempts_left: int
    temperature: float = 0.0
    json_mode: bool = False
    max_new_tokens: Optional[int] = None
    model: Optional[APIModel] = None
    result: list = field(default_factory=list)

    def __post_init__(self):
        # automatically select model if not specified
        tokens = tokenizer.encode(json.dumps(self.messages))
        self.num_tokens = len(tokens)
        if self.model is None:
            if self.num_tokens < 12000:
                self.model = GPT3_TURBO
            else:
                self.model = GPT4_TURBO

        self.request_header = {
            "Authorization": f"Bearer {os.getenv(self.model.api_key_env_var)}",
        }
        self.request_json = {
            "model": self.model.name,
            "messages": self.messages,
            "temperature": self.temperature,
        }
        if self.max_new_tokens is not None:
            self.request_json["max_tokens"] = self.max_new_tokens
        if self.json_mode and "1106" in self.model.name:
            self.request_json["response_format"] = {"type": "json_object"}

    async def call_api(
        self,
        retry_queue: asyncio.Queue,
        status_tracker: StatusTracker,
        callback: Optional[Callable] = None,
        cache: Optional[SqliteCache] = None,
        pbar: Optional[tqdm] = None,
    ):
        # first try to get from cache
        if cache is not None:
            cached_result = cache.get_from_cache(self.messages)
            if cached_result:
                logger.info(f"Found cached completion for prompt {self.task_id}")
                self.result.append(json.loads(cached_result[0]))
                if pbar is not None:
                    pbar.update(1)
                if callback is not None:
                    callback(
                        self.task_id,
                        self.messages,
                        self.result[-1]["choices"][0]["message"]["content"],
                        status_tracker,
                    )
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_succeeded += 1
                return

        # if not in cache, call the API
        try:
            status_tracker.total_requests += 1
            timeout = aiohttp.ClientTimeout(total=self.model.request_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url=self.model.api_base + "/chat/completions",
                    headers=self.request_header,
                    json=self.request_json,
                ) as response:
                    # make sure it's a 200-level status code
                    if response.status >= 200 and response.status < 300:
                        try:
                            # # get response mimetype
                            # 
                            # # if not json, print the response, this is unexpected
                            # if "json" not in mimetype:
                            #     logger.error(f"Response with unexpected mimetype... status code {response.status}")
                            #     logger.error(f"Mimetype: {mimetype}, Task id: {self.task_id}, input was {self.messages}")
                            #     logger.error(f"Got response: {await response.text()}")
                            #     raise Exception("Unexpected mimetype")
                            response = await response.json()
                        except Exception as e:
                            logger.error(f"Exception parsing response: {e}")
                            raise e
                    else:
                        mimetype = response.headers["Content-Type"]
                        if "json" in mimetype:
                            response = await response.json()
                            logger.log_to_file(f"Error response: {json.dumps(response)}")
                        else:
                            text = await response.text()
                            response = {"error": {"message": text, "status_code": response.status}}
                            logger.log_to_file(f"Error response: {response}")

            if "error" in response:
                logger.log_to_file(f"'error' key in response: {json.dumps(response)}")
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                if "context length" in response["error"].get("message", ""):
                    logger.error("context length exceeded, retrying won't help")
                    self.attempts_left = 0
                self.result.append(response)
                if self.attempts_left > 0:
                    self.attempts_left -= 1
                    retry_queue.put_nowait(self)
                else:
                    logger.error("out of tries")
                    status_tracker.num_tasks_in_progress -= 1
                    status_tracker.num_tasks_failed += 1
            else:
                if pbar is not None:
                    pbar.update(1)
                if callback is not None:
                    callback(
                        self.task_id,
                        self.messages,
                        response["choices"][0]["message"]["content"],
                        status_tracker,
                    )
                self.result.append(response)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_succeeded += 1
                if cache is not None:
                    cache.set_to_cache(self.messages, json.dumps(response))
        except Exception as e:
            logger.log_to_file(f"{type(e).__name__}")

            self.result.append({"error": str(e)})
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1


# assumes the API keys are already stored in env variables
async def process_api_requests_from_list(
    prompts: list[list[dict]],  # each prompt is just a list of messages
    max_attempts: int,
    max_tokens_per_minute: int,  # you're gonna need to specify these, don't break everything lol
    max_requests_per_minute: int,
    model: Optional[APIModel] = None,
    cache: Optional[SqliteCache] = None,
    callback: Optional[Callable] = None,  # should take in (id, messages, response)
    temperature: float = 0.0,
    json_mode: bool = False,
    max_new_tokens: Optional[int] = None,
    show_progress: bool = False,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.01  # so concurrent tasks can run

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    prompts_not_finished = True
    # logger.debug(f"Initialization complete.")

    # turn the texts into an iterator
    if show_progress:
        pbar = tqdm(total=len(prompts))
    else:
        pbar = None
    prompts = iter(enumerate(prompts))
    results = []
    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            if not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
                logger.log_to_file(
                    f"Retrying request {next_request.task_id}."  #: {next_request}"
                )
            elif prompts_not_finished:
                try:
                    # get new request
                    idx, messages = next(prompts)
                    next_request = APIRequest(
                        task_id=idx,
                        messages=messages,
                        attempts_left=max_attempts,
                        temperature=temperature,
                        json_mode=json_mode,
                        max_new_tokens=max_new_tokens,
                        model=model,
                    )
                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    results.append(next_request)
                    # logger.debug(
                    #     f"Reading request {next_request.task_id}: {next_request}"
                    # )
                except StopIteration:
                    # if prompts run out, set flag to stop reading it
                    # logger.info("Finished prompts. Only retries left.")
                    prompts_not_finished = False

        # update available capacity
        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity
            + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )
        available_token_capacity = min(
            available_token_capacity
            + max_tokens_per_minute * seconds_since_update / 60.0,
            max_tokens_per_minute,
        )
        last_update_time = current_time

        # if enough capacity available, call API
        if next_request:
            next_request_tokens = next_request.num_tokens
            if (
                available_request_capacity >= 1
                and available_token_capacity >= next_request_tokens
            ):
                # update counters
                available_request_capacity -= 1
                available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1

                # call API
                asyncio.create_task(
                    next_request.call_api(
                        retry_queue=queue_of_requests_to_retry,
                        status_tracker=status_tracker,
                        callback=callback,
                        cache=cache,
                        pbar=pbar,
                    )
                )
                # logger.debug(
                #     f"Called API for request {next_request.task_id}: {next_request}"
                # )
                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        seconds_since_rate_limit_error = (
            time.time() - status_tracker.time_of_last_rate_limit_error
        )
        if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
            remaining_seconds_to_pause = (
                seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
            )
            await asyncio.sleep(remaining_seconds_to_pause)
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            logger.warn(
                f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
            )

    # after finishing, log final status
    # logger.info(f"""Parallel processing complete.""")
    if status_tracker.num_tasks_failed > 0:
        logger.warning(
            f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed."
        )
    if status_tracker.num_rate_limit_errors > 0:
        logger.warning(
            f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
        )
    return results

async def run_chat_queries_async(
    prompts: list[list[dict]],  # each prompt is just a list of messages
    max_tokens_per_minute: int,
    max_requests_per_minute: int,
    temperature: float = 0.0,
    json_mode: bool = False,
    model_name: Literal["gpt-3.5-turbo", "gpt-4-turbo", "gpt4", "mistral"] = None,
    callback: Optional[Callable] = None,  # should take in (id, messages, response)
    max_new_tokens: Optional[int] = None,
    max_attempts: int = 5,
    cache_file: str = None,
    show_progress: bool = False,
):
    model = {
        "gpt-3.5-turbo": GPT3_TURBO,
        "gpt-4-turbo": GPT4_TURBO,
        "gpt4": GPT4,
        "mistral": MISTRAL,
    }.get(model_name, None)
    if cache_file is not None:
        cache = SqliteCache(cache_file)
    else:
        cache = None
    results = await process_api_requests_from_list(
        prompts=prompts,
        max_attempts=max_attempts,
        max_tokens_per_minute=max_tokens_per_minute,
        max_requests_per_minute=max_requests_per_minute,
        temperature=temperature,
        json_mode=json_mode,
        max_new_tokens=max_new_tokens,
        show_progress=show_progress,
        cache=cache,
        model=model,
        callback=callback,
    )
    # extract the replies
    replies = [None for _ in range(len(prompts))]
    usage = [None for _ in range(len(prompts))]
    for result in results:
        if len(result.result) == 0:
            logger.error(f"Result is empty: {result}")
            raise Exception("Result is empty")
        if isinstance(result.result[-1], str):
            logger.error(f"Result is a string instead of the expected dict: {result}")
            raise Exception("Result is a string")
        if "error" in result.result[-1].keys():
            replies[result.task_id] = None
        else:
            replies[result.task_id] = result.result[-1]["choices"][0]["message"][
                "content"
            ]
        usage[result.task_id] = {
            "model": result.model.name,
            "input_tokens": result.num_tokens,
            "completion_tokens": len(tokenizer.encode(replies[result.task_id])) if replies[result.task_id] is not None else None,
            "attempts": max_attempts - result.attempts_left,
        }
        
    return replies, usage


def instructions_to_message_lists(prompts: list[str], system_prompt: str = None):
    """
    Convert a list of instructions into a list of lists of messages.
    """
    result = []
    for p in prompts:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": p})
        result.append(messages)
    return result

async def get_completion_simple_async(
    prompt: str,
    model_name: Literal[
        "gpt-3.5-turbo", "gpt-4-turbo", "gpt4", "mistral"
    ] = "gpt-3.5-turbo",
    temperature: float = 0.0,
    json_mode: bool = False,
    max_new_tokens: Optional[int] = None,
):
    """
    Get a single completion from a prompt.
    """
    result, usage = await run_chat_queries_async(
        prompts=instructions_to_message_lists([prompt]),
        max_tokens_per_minute=25000,
        max_requests_per_minute=100,
        temperature=temperature,
        json_mode=json_mode,
        model_name=model_name,
        cache_file=None,
        max_new_tokens=max_new_tokens,
        max_attempts=5,
        show_progress=False,
    )
    return result[0]
