import aiohttp
import argparse
import asyncio
import backoff
import datetime
import os
import json
import hashlib
import random
import re
import requests
import secrets
import signal
import string
import sys
import threading
import yaml
import concurrent.futures
from collections import defaultdict
from functools import partial
from loguru import logger
from queue import Queue, Empty
from time import sleep
from typing import List, Dict, Any
from uuid import uuid4
from exceptions import (
    RateLimitError,
    TooManyRequestsError,
    TokensExhaustedError,
    ServerOverloadedError,
    ServerError,
    ContextLengthExceededError,
    BadResponseError,
)
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Defaults and constants.
MAX_DOCSTORE_SIZE = 15000
OPENAI_API_BASE_URL = "https://api.openai.com"
MODEL_ENDPOINTS = {
    "completions": [
        "text-davinci-003",
        "text-davinci-002",
        "text-curie-001",
        "text-babbage-001",
        "text-ada-001",
    ],
    "chat_completions": [
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-32k",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
    ],
}


class SelfInstructor:
    """Class and methods used to generate instructions, based on self-instruct paper/code."""
    CLI_ARGS = {

        # The updated code with several instructors has way too many options to support
        # as CLI args, so we just accept the config file path now.
        "--config": {
            "type": str,
            "default": "config.yaml",
            "help": "path to the airobors configuration file",
        },
    }

    def __init__(self, *, config_path: str = "config.yaml"):
        """Constructor."""
        self.used_tokens = 0
        self.config_path = config_path
        self.load_config()
        self.instructor_counts = defaultdict(int)
        self.initialize_docstores()

    def load_config(self):
        """Load an advanced configuration from a YAML file."""
        raw_config = self.raw_config = yaml.safe_load(open(self.config_path).read())
        self.model = raw_config.get("model") or "gpt-4"
        self.openai_api_key = raw_config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable or openai_api_key must be provided"
            )
        self.organization_id = raw_config.get("organization_id")
        self.topics_path = raw_config.get("topics_path") or "topics.txt"
        self.output_path = raw_config.get("output_path") or "instructions.jsonl"
        self.overwrite = str(raw_config.get("overwrite")).lower() == "true"
        self.append = str(raw_config.get("append")).lower() == "true"
        self.topic_avoidance = raw_config.get("topic_avoidance", "")
        self.response_filters = []
        for val in raw_config.get("response_filters") or []:
            self.response_filters.append(re.compile(val, re.I))
        self.max_tokens = int(raw_config["max_tokens"]) if raw_config.get("max_tokens") else None
        self.min_docsearch_score = float(raw_config.get("min_docsearch_score") or 0.35)
        api_params = raw_config.get("api_params") or {}
        self.api_params = {
            "temperature": float(api_params.get("temperature") or 0.7),
            "top_p": float(api_params.get("top_p") or 0.5),
            "frequency_penalty": float(api_params.get("frequency_penalty") or 0.0),
            "presence_penalty": float(api_params.get("presence_penalty") or 2.0),
        }
        self.topic_prompt = raw_config["topic_prompt"].format(topic_avoidance=self.topic_avoidance)
        self.topic_request_count = int(raw_config.get("topic_request_count") or 20)
        self.default_count = int(raw_config.get("default_count") or 100)
        self.default_batch_size = int(raw_config.get("default_batch_size") or 5)

        # Validate the model for each generator.
        self.instructors = raw_config.get("instructors")
        self.validate_model(self.model)
        valid_models = {self.model: True}
        for key, config in self.instructors.items():
            if config.get("model") and config["model"] not in valid_models:
                self.validate_model(config["model"])
                valid_models[config["model"]] = True

    def initialize_docstores(self):
        """Initialize the in-memory vector databases used to check prompt uniqueness."""
        docs = []
        if os.path.exists(self.output_path):
            if self.overwrite:
                result = input("Remove and overwrite {output_path} (Y/N)? ")
                if result.strip().lower() == "y":
                    os.remove(self.output_path)
                else:
                    raise RuntimeError("Overwrite aborted.")
            elif self.append:
                with open(self.output_path, "r") as infile:
                    for line in infile.readlines():
                        task = json.loads(line)
                        self.instructor_counts[task.get("category", "general")] += 1
                        docs.append(task["instruction"])
                logger.info(
                    f"Found {len(docs)} existing machine-generated instruction(s)."
                )
                for category, count in self.instructor_counts.items():
                    logger.info(f"  - category {category}: {count}")
            else:
                raise RuntimeError(
                    f"{self.output_path} already exists, but overwrite and append are false!"
                )
        logger.info(
            "Initializing in-memory document store for similarity comparison..."
        )
        if not docs:
            docs = ["__initialize__"]
        self.embeddings = HuggingFaceEmbeddings()
        self.docstores = [Chroma.from_texts(docs, self.embeddings)]
        self.docstore_rotated_at = 0
        self.topic_index = 0
        if len(docs) >= MAX_DOCSTORE_SIZE:
            logger.info("Initializing fresh docstore due to doc count...")
            self.docstore_rotated_at = len(docs)
            self.docstores.append(
                Chroma.from_texts(["__initialize__"], self.embeddings)
            )

    def validate_model(self, model):
        """Ensure the specified model is available, and configure the endpoint
        to use accordingly (chat completions or completions).
        """
        if model in MODEL_ENDPOINTS["completions"]:
            self._completions = True
        elif model in MODEL_ENDPOINTS["chat_completions"]:
            self._completions = False
        else:
            raise ValueError(f"Model is not currently supported: {model}")
        # Ensure the model is authorized for this key.
        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        if self.organization_id:
            headers["OpenAI-Organization"] = self.organization_id
        result = requests.get(f"{OPENAI_API_BASE_URL}/v1/models", headers=headers)
        if result.status_code != 200:
            raise ValueError(
                f"Invalid openai API key [{result.status_code}: {result.text}]"
            )
        available = {item["id"] for item in result.json()["data"]}
        if model not in available:
            raise ValueError(f"Model is not available to your API key: {model}")
        logger.success(f"Successfully validated model: {model}")

    async def initialize_topics(self) -> List[str]:
        """Ensure topics are initialized, i.e. topics already exist and are read,
        or a new list of topics is generated.
        """
        if os.path.exists(self.topics_path):
            self.topics = list(
                {line.strip() for line in open(self.topics_path).readlines() if line.strip()}
            )
            logger.info(
                f"Using {len(self.topics)} topics from {self.topics_path}..."
            )
            return

        logger.info("Generating random topics to use in prompts...")
        seen = set([])
        self.topics = []
        with open(self.topics_path, "w") as outfile:
            count = self.topic_request_count
            while count > 0:
                todo = 8 if count >= 8 else count
                responses = await asyncio.gather(*[
                    self.generate_response(self.topic_prompt, **self.api_params)
                    for _ in range(todo)
                ])
                count -= todo
                for response in responses:
                    if not response:
                        continue
                    for topic in re.findall(r"(?:^|\n)\d+\. (.*?)(?:$|(?=\n\d+\. ))", response, re.DOTALL):
                        if not topic or topic.lower().strip() in seen:
                            continue
                        seen.add(topic.lower().strip())
                        self.topics.append(topic)
                        outfile.write(topic.strip() + "\n")
        logger.success(
            f"Successfully generated {len(self.topics)} topics, stored in {self.topics_path}..."
        )

    def get_instructor_topics(self, instructor_config):
        """Get the topics for a specific instructor, defaulting to main topics.

        :param instructor_config: Dict containing the target instructor's config.
        :type instructor_config: dict

        :return: List of topic strings.
        :rtype: list[str]
        """
        if not instructor_config.get("topics_path"):
            return self.topics
        with open(instructor_config["topics_path"]) as infile:
            topics = list(
                {line.strip() for line in infile.readlines() if line.strip()}
            )
            if not topics:
                raise ValueError(f"Found empty topics file: {instructor_config['topics_path']}")
        return topics

    def generate_prompt(self, template: str):
        """Generate a single prompt, inserting random topics.

        :param template: The prompt template to use.
        :type template: str

        :return: The prompt, including a list of random topics.
        :rtype: str
        """
        self.topic_lock.acquire()
        topics = []
        for _ in range(self.batch_size):
            topics.append(self.topics[self.topic_index])
            self.topic_index += 1
            if self.topic_index >= len(self.topics):
                self.topic_index = 0
        self.topic_lock.release()
        topics = "\n".join(
            [
                f" * instruction {idx + 1} must be related to topic: {json.dumps(topic)}"
                for idx, topic in enumerate(topics)
            ]
        )
        return template.format(topics=topics, batch_size=self.batch_size)

    def extract_instructions_from_response(self, text: str) -> List[str]:
        """Extract the list of instructions from the OpenAI response.

        :param text: The OpenAI response text.
        :type text: str

        :return: List of instructions.
        :rtype: List[str]
        """
        if not text:
            return []
        instructions = []
        for instruction in re.findall(r"(?:^|\n)\d+\. (.*?)(?:$|(?=\n\d+\. ))", text):
            # Skip various prompts that have been deemed unsuitable for language models
            # by the self-instruct team.
            if (
                not instruction.strip()
                or self.skip_instruction_re.search(instruction)
                or instruction[0] in string.punctuation
                or not instruction[0].isascii()
            ):
                logger.warning(
                    f"Skipping instruction: {instruction} [unsuitable prompt]"
                )
                continue
            instructions.append(instruction.strip())
            logger.info(f"Generated candidate task: {instruction}")
        return instructions

    @backoff.on_exception(
        backoff.expo,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            ServerError,
            RateLimitError,
            TooManyRequestsError,
            ServerOverloadedError,
        ),
    )
    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a post request to OpenAI API.

        :param path: URL path to send request to.
        :type path: str

        :param payload: Dict containing request body/payload.
        :type payload: Dict[str, Any]

        :return: Response object.
        :rtype: Dict[str, Any]
        """
        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        if self.organization_id:
            headers["OpenAI-Organization"] = self.organization_id
        request_id = str(uuid4())
        logger.debug(f"POST [{request_id}] with payload {json.dumps(payload)}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OPENAI_API_BASE_URL}{path}",
                headers=headers,
                json=payload,
                timeout=600.0,
            ) as result:
                if result.status != 200:
                    text = await result.text()
                    logger.error(f"OpenAI request error: {text}")
                    if "too many requests" in text.lower():
                        raise TooManyRequestsError(text)
                    if "rate limit reached" in text.lower():
                        sleep(30)
                        raise RateLimitError(text)
                    elif "context_length_exceeded" in text.lower():
                        raise ContextLengthExceededError(text)
                    elif "server_error" in text and "overloaded" in text.lower():
                        raise ServerOverloadedError(text)
                    elif "bad gateway" in text.lower() or "server_error" in text.lower():
                        raise ServerError(text)
                    else:
                        raise BadResponseError(text)
                result = await result.json()
                logger.debug(f"POST [{request_id}] response: {json.dumps(result)}")
                self.used_tokens += result["usage"]["total_tokens"]
                if self.max_tokens and self.used_tokens > self.max_tokens:
                    raise TokensExhaustedError(f"Max token usage exceeded: {self.used_tokens}")
                logger.debug(f"token usage: {self.used_tokens}")
                return result

    async def _post_no_exc(self, *a, **k) -> Dict[str, Any] | None:
        """Post, ignoring all exceptions."""
        try:
            return await self._post(*a, **k)
        except Exception as ex:
            logger.error(f"Error performing post: {ex}")
        return None

    async def generate_response(
        self, instruction: str, **kwargs
    ) -> str:
        """Call OpenAI with the specified instruction and return the text response.

        :param instruction: The instruction to respond to.
        :type instruction: str

        :param recurse: Allow recursive calls, e.g. to rephrase to remove AI refs.
        :type recurse: bool

        :return: Response text.
        :rtype: str
        """
        model = kwargs.get("model", self.model)
        completions = True if model in MODEL_ENDPOINTS["completions"] else False
        path = "/v1/completions" if completions else "/v1/chat/completions"
        payload = {**kwargs}
        if "model" not in payload:
            payload["model"] = model
        if completions:
            payload["prompt"] = instruction
            payload["max_tokens"] = 2000
        else:
            payload["messages"] = [{"role": "user", "content": instruction}]
        response = await self._post_no_exc(path, payload)
        if (
            not response
            or not response.get("choices")
            or response["choices"][0]["finish_reason"] == "length"
        ):
            return None
        text = None
        if self._completions:
            text = response["choices"][0]["text"]
        else:
            text = response["choices"][0]["message"]["content"]

        if any([banned.match(text, re.I) for banned in self.response_filters]):
            logger.warning(f"Banned response: {text}")
            return None
        if text.startswith(("I'm sorry,", "Apologies,", "I can't", "I won't")):
            logger.warning(f"Banned response: {text}")
            return None
        return text

    def generate_instruction_batch(self, queue: Queue) -> None:
        """Generate an set of instructions.

        :param queue: Queue to pass generated outputs to.
        :type queue: Queue
        """
        contextual = random.random() <= self.contextual_prompt_ratio
        prompt = self.generate_prompt(
            self.prompt if not contextual else self.contextual_prompt
        )
        for new_instruction in self.extract_instructions_from_response(
            self.generate_response(
                prompt, temperature=self.prompt_generation_temperature
            )
        ):
            prompt = new_instruction
            if contextual:
                injected = new_instruction + CONTEXT_TASK_INJECTION
                if random.random() <= 0.2:
                    injected += " " + FORMAT_INJECTION
                prompt = self.generate_response(injected)
                if not prompt or "=:=:=" not in prompt:
                    logger.error(
                        f"Error generating contextual prompt: {new_instruction}"
                    )
                    continue
                parts = [
                    part.strip().lstrip(":").strip()
                    for part in prompt.split("=:=:=")
                    if part.strip()
                ]
                if len(parts) != 2:
                    logger.warning(
                        f"Contextual prompt returned incorrect part count: {prompt}"
                    )
                    continue
                flip = random.random()
                if flip <= 0.7:
                    prompt = f"Using the provided text, respond to the instruction: {parts[1]}\n\n{parts[0]}"
                elif flip <= 0.85:
                    prompt = (
                        parts[0]
                        + f"\n\nUsing the text above, respond to the instruction: {parts[1]}"
                    )
                else:
                    prompt = parts[1] + f"\n\nContext:\n{parts[0]}"
            queue.put({"instruction": prompt})

    def generate_instruction_batches(self, queue: Queue) -> None:
        """Generate batches of instructions, storing new instructions in queue.

        :param queue: Queue to store new instructions in for post-processing.
        :type queue: Queue
        """
        consecutive_errors = 0
        while not self.stop_producing:
            try:
                self.generate_instruction_batch(queue)
                consecutive_errors = 0
            except TokensExhaustedError:
                logger.error("Max tokens reached, stopping...")
                break
            except Exception as exc:
                logger.error(f"Unhandled exception generating batch: {exc}")
                consecutive_errors += 1
                if consecutive_errors > 3:
                    logger.error("Too many consecutive errors, shutting down!")
                    os.kill(os.getpid(), signal.SIGKILL)

    def is_too_similar(self, instruction: str, min_score: float = None):
        """Check the similarity of a new instruction to the existing set.

        :param instruction: The instruction string to compare.
        :type instruction: str

        :param min_score: Minimum document similarity score to consider unique.
        :type min_score: float

        :return: Boolean indicating if the instruction is too similar or not.
        :rtype: bool
        """
        min_ = 1.0
        for docstore in self.docstores:
            similar = docstore.similarity_search_with_score(instruction, k=1)
            for _, score in similar:
                if score < min_:
                    min_ = score
        if min_ <= min_score:
            logger.warning(f"Skipping instruction, too similar [{min_}]: {instruction}")
            return True
        return False

    def validate_and_store_results(self, queue: Queue) -> None:
        """Dedupe based on rouge score for each new instruction and save results.

        :param queue: Queue to consume messages from.
        :type queue: Queue
        """
        with open(self.output_path, "a+") as outfile:
            while True:
                instruction = queue.get()
                if not instruction:
                    break
                min_score = 1.0
                for docstore in self.docstores:
                    similar = docstore.similarity_search_with_score(
                        instruction["instruction"], k=1
                    )
                    for _, score in similar:
                        if score < min_score:
                            min_score = score
                if min_score <= self.min_docsearch_score:
                    logger.warning(
                        f"Skipping instruction, too similar [{min_score}]: {instruction['instruction']}"
                    )
                    continue
                outfile.write(json.dumps(instruction) + "\n")
                outfile.flush()
                self.machine_task_count += 1
                if self.machine_task_count >= self.instruction_count:
                    self.stop_producing = True
                started_at = datetime.datetime.utcnow()
                if (
                    self.machine_task_count - self.docstore_rotated_at
                    >= MAX_DOCSTORE_SIZE
                ):
                    logger.info("Initializing new docstore...")
                    self.docstores.append(
                        Chroma.from_texts(["__initialize__"], self.embeddings)
                    )
                    self.docstore_rotated_at = self.machine_task_count
                self.docstores[-1].add_texts([instruction["instruction"]])
                delta = round(
                    (datetime.datetime.utcnow() - started_at).total_seconds(), 3
                )
                logger.success(
                    f"Generated unique [score={round(min_score, 4)}] instruction in {delta}s [total={self.machine_task_count}]: {instruction['instruction']}"
                )

    def inject_response(self, instruction):
        """Update the input instruction with the response from OpenAI."""
        if instruction.get("response"):
            return instruction
        result = self.generate_response(instruction["instruction"])
        if result:
            return {"instruction": instruction["instruction"], "response": result}
        return None

    def run_prompt_generation_phase(self):
        """Run the self-instruct, instruction generation (without responses)."""
        if self.machine_task_count >= self.instruction_count:
            logger.warning(
                f"Already have {self.machine_task_count} machine-generated tasks, skipping generation..."
            )
            return
        queue = Queue(maxsize=self.concurrency * BATCH_SIZE)
        producers = [
            threading.Thread(target=self.generate_instruction_batches, args=(queue,))
            for _ in range(self.concurrency)
        ]
        for producer in producers:
            producer.start()
        consumer = threading.Thread(
            target=self.validate_and_store_results, args=(queue,)
        )
        consumer.start()
        for producer in producers:
            producer.join()
        queue.put(None)
        consumer.join()

    def generate_responses(self, input_queue: Queue, output_queue: Queue):
        """Generate responses to machine-generated prompts."""
        while True:
            try:
                instruction = input_queue.get(block=True, timeout=10.0)
                if instruction is None:
                    output_queue.put(None)
                    break
            except Empty:
                continue
            if result := self.inject_response(instruction):
                output_queue.put(result)

    def store_completed_results(self, tmp_path: str, queue: Queue) -> None:
        """Store all completed instructions."""
        finished_count = 0
        with open(tmp_path, "a+") as outfile:
            while True:
                try:
                    instruction = queue.get(block=True, timeout=10.0)
                except Empty:
                    continue
                if instruction is None:
                    finished_count += 1
                    if finished_count == self.concurrency:
                        break
                else:
                    outfile.write(json.dumps(instruction) + "\n")
                    logger.success(
                        f"Generated response [{instruction['instruction'][0:100]}...]\n{instruction['response']}"
                    )

    def run_response_generation_phase(self):
        """Generate the responses for each of the generated prompts."""
        input_queue = Queue(maxsize=self.concurrency * 4)
        output_queue = Queue()
        producers = [
            threading.Thread(
                target=self.generate_responses, args=(input_queue, output_queue)
            )
            for _ in range(self.concurrency)
        ]
        for producer in producers:
            producer.start()

        # Skip over any responses that have already been generated.
        tmp_path = f"{self.output_path}.with_results.tmp"
        already_responded = set([])
        if os.path.exists(tmp_path):
            with open(f"{tmp_path}.filtered", "w") as outfile:
                with open(tmp_path, "r") as infile:
                    for line in infile:
                        instruction = json.loads(line)
                        if "response" in instruction:
                            already_responded.add(
                                hashlib.md5(
                                    instruction["instruction"].encode()
                                ).hexdigest()
                            )
                            outfile.write(line)
            os.rename(f"{tmp_path}.filtered", tmp_path)
            logger.info(
                f"Found {len(already_responded)} prompts that have already been responded to..."
            )

        # Start consumer.
        consumer = threading.Thread(
            target=self.store_completed_results, args=(tmp_path, output_queue)
        )
        consumer.start()

        # Queue up the instructions to be answered.
        with open(self.output_path, "r") as infile:
            for line in infile:
                instruction = json.loads(line)
                if (
                    hashlib.md5(instruction["instruction"].encode()).hexdigest()
                    in already_responded
                ):
                    continue
                input_queue.put(instruction)

        # Send termination queue messages to each producer.
        for _ in range(self.concurrency):
            input_queue.put(None)

        # Join all threads.
        for producer in producers:
            producer.join()
        consumer.join()
        os.rename(tmp_path, self.output_path)

    def run(self):
        """Run prompt generation and answer to completion."""
        self.initialize_topics()
        self.initialize_docstores()
        self.run_prompt_generation_phase()
        logger.success(
            f"Finished generating instructions [asked for {self.instruction_count}, created {self.machine_task_count}], generating responses..."
        )
        self.run_response_generation_phase()
        logger.success(
            f"Finished self-instruct task, total instructions: {self.machine_task_count}"
        )


def generate_instructions(args):
    random.seed(secrets.randbelow(1000000000))
    parser = argparse.ArgumentParser()
    for arg, kwargs in SelfInstructor.CLI_ARGS.items():
        parser.add_argument(arg, **kwargs)
    SelfInstructor(**vars(parser.parse_args(args))).run()


def generate_topics(args):
    random.seed(secrets.randbelow(1000000000))
    parser = argparse.ArgumentParser()
    for arg, kwargs in SelfInstructor.CLI_ARGS.items():
        parser.add_argument(arg, **kwargs)
    instructor = SelfInstructor(**vars(parser.parse_args(args)))
    asyncio.run(instructor.initialize_topics())

if __name__ == "__main__":
    generate_instructions(sys.argv[1:])
