import hashlib
import importlib.resources
import logging
import os
import pathlib
import re
from typing import List, Union

from dotenv import load_dotenv
from joblib import Memory
import requests

from ollama import Client
from openai import OpenAI

memory = Memory(location='cache', verbose=0)


def hash_input(*args):
    return hashlib.md5(str(args).encode()).hexdigest()


def _default_prompt_path(filename: str) -> pathlib.Path:
    """Return the path to a bundled prompt template file."""
    ref = importlib.resources.files("safe_spoon.prompting.prompts").joinpath(filename)
    return pathlib.Path(str(ref))


class Prompter:
    def __init__(
        self,
        model_type: str,
        llm_server: str = None,
        llm_provider: str = None,
        api_key: str = None,
        logger: logging.Logger = None,
        temperature: float = 0,
        seed: int = 1234,
        max_tokens: int = None,
    ):
        self._logger = logger if logger else logging.getLogger(__name__)
        self.model_type = model_type
        self.context = None
        self.params = {"temperature": temperature, "seed": seed}
        self._api_key = api_key

        # Determine backend; recognize gpt-like models by default
        if llm_provider is not None:
            self.backend = llm_provider
            self._logger.info(f"Using provider: {llm_provider}")
        elif model_type.startswith(("gpt-", "o1-", "o3-", "o4-")):
            self.backend = "openai"
        else:
            raise ValueError(
                f"Cannot determine backend for model '{model_type}'. "
                "Set llm_provider explicitly (e.g. 'ollama', 'vllm', 'openai')."
            )

        if max_tokens is not None:
            if self.backend == "ollama":
                self.params["num_predict"] = max_tokens
                self._logger.info(f"Setting num_predict to: {max_tokens}")
            else:
                self.params["max_tokens"] = max_tokens
                self._logger.info(f"Setting max_tokens to: {max_tokens}")

        if self.backend == "openai":
            load_dotenv()
            self.openai_base_url = llm_server
            if self.openai_base_url:
                self._validate_model(model_type, "openai", base_url=self.openai_base_url)
            self._logger.info(
                f"Using OpenAI API with model: {model_type}"
                + (f", base_url: {llm_server}" if llm_server else "")
            )

        elif self.backend == "vllm":
            self.openai_base_url = llm_server
            if not self.openai_base_url:
                raise ValueError(
                    "llm_server (base URL) is required for the vllm backend."
                )
            if self._api_key is None:
                load_dotenv()
                self._api_key = (
                    os.getenv("VLLM_API_KEY")
                    or os.getenv("OPENAI_API_KEY")
                    or "EMPTY"
                )
            self._validate_model(model_type, "vllm", base_url=self.openai_base_url)
            self._logger.info(
                f"Using vLLM (OpenAI-compatible) at {self.openai_base_url} with model: {model_type}"
            )

        elif self.backend == "ollama":
            ollama_host = llm_server or os.getenv("OLLAMA_HOST", "http://localhost:11434")
            os.environ['OLLAMA_HOST'] = ollama_host
            Prompter.ollama_client = Client(
                host=ollama_host,
                headers={'x-some-header': 'some-value'}
            )
            self._validate_model(model_type, "ollama", ollama_host=ollama_host)
            self._logger.info(f"Using OLLAMA API with host: {ollama_host}")

        elif self.backend == "llama_cpp":
            self.llama_cpp_host = llm_server or "http://localhost:11435/v1/chat/completions"
            self._logger.info(f"Using llama_cpp API with host: {self.llama_cpp_host}")

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _validate_model(
        self,
        model_type: str,
        backend: str,
        base_url: str = None,
        ollama_host: str = None,
    ) -> None:
        available = Prompter.fetch_available_models(
            backend=backend,
            base_url=base_url,
            api_key=self._api_key,
            ollama_host=ollama_host,
        )
        if model_type not in available:
            raise ValueError(
                f"Model '{model_type}' is not available on the {backend} backend. "
                f"Available models: {available}"
            )

    @staticmethod
    def fetch_available_models(
        backend: str,
        base_url: str = None,
        api_key: str = None,
        ollama_host: str = None,
    ) -> List[str]:
        """Query the backend server and return the list of available model IDs."""
        if backend in ("openai", "vllm"):
            resolved_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
            kwargs: dict = {"api_key": resolved_key}
            if base_url:
                kwargs["base_url"] = base_url
            client = OpenAI(**kwargs)
            return [m.id for m in client.models.list().data]
        elif backend == "ollama":
            host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
            return [m.model for m in Client(host=host).list().models]
        elif backend == "llama_cpp":
            base = re.sub(r'/chat/completions$', '', (base_url or "")).rstrip('/')
            client = OpenAI(api_key="EMPTY", base_url=base)
            return [m.id for m in client.models.list().data]
        else:
            raise ValueError(f"fetch_available_models not supported for backend: {backend}")

    @staticmethod
    @memory.cache
    def _cached_prompt_impl(
        template: str,
        question: str,
        model_type: str,
        backend: str,
        params: tuple,
        context=None,
        use_context: bool = False,
        openai_base_url: str = None,
        api_key: str = None,
    ) -> dict:
        print("Cache miss: computing results...")

        if backend in ("openai", "vllm"):
            result, logprobs = Prompter._call_openai_api(
                template=template,
                question=question,
                model_type=model_type,
                params=dict(params),
                base_url=openai_base_url,
                api_key=api_key,
            )
        elif backend == "ollama":
            result, logprobs, context = Prompter._call_ollama_api(
                template=template,
                question=question,
                model_type=model_type,
                params=dict(params),
                context=context,
            )
        elif backend == "llama_cpp":
            result, logprobs = Prompter._call_llama_cpp_api(
                template=template,
                question=question,
                params=dict(params),
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return {
            "inputs": {
                "template": template,
                "question": question,
                "model_type": model_type,
                "backend": backend,
                "params": dict(params),
                "context": context if use_context else None,
                "use_context": use_context,
            },
            "outputs": {
                "result": result,
                "logprobs": logprobs,
            },
        }

    @staticmethod
    def _call_openai_api(template, question, model_type, params, base_url=None, api_key=None):
        if template is not None:
            messages = [
                {"role": "system", "content": template},
                {"role": "user", "content": question},
            ]
        else:
            messages = [
                {"role": "user", "content": question},
            ]

        resolved_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        client_kwargs: dict = {"api_key": resolved_key}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        open_ai_client = OpenAI(**client_kwargs)
        max_tok = params.get("max_tokens", 1000)

        def _create(use_max_completion_tokens=False, use_temperature=True, **extra):
            kwargs = dict(model=model_type, messages=messages, stream=False)
            if use_temperature:
                kwargs["temperature"] = params["temperature"]
            if use_max_completion_tokens:
                kwargs["max_completion_tokens"] = max_tok
            else:
                kwargs["max_tokens"] = max_tok
            seed = params.get("seed")
            if seed is not None:
                kwargs["seed"] = seed
            kwargs.update(extra)
            return open_ai_client.chat.completions.create(**kwargs)

        def _create_with_fallbacks(**extra):
            """Retry stripping unsupported params until the call succeeds."""
            use_mct = False
            use_temp = True
            while True:
                try:
                    return _create(
                        use_max_completion_tokens=use_mct,
                        use_temperature=use_temp,
                        **extra,
                    )
                except Exception as e:
                    msg = str(e)
                    if ("max_completion_tokens" in msg or "max_tokens" in msg) and not use_mct:
                        use_mct = True
                    elif "temperature" in msg and use_temp:
                        use_temp = False
                    else:
                        raise

        try:
            response = _create_with_fallbacks(logprobs=True, top_logprobs=10)
            logprobs = response.choices[0].logprobs.content
        except Exception:
            # model does not support logprobs
            response = _create_with_fallbacks()
            logprobs = None
        result = response.choices[0].message.content
        return result, logprobs

    @staticmethod
    def _call_ollama_api(template, question, model_type, params, context):
        if Prompter.ollama_client is None:
            raise ValueError("OLLAMA client is not initialized.")

        if template is not None:
            response = Prompter.ollama_client.generate(
                system=template,
                prompt=question,
                model=model_type,
                stream=False,
                options=params,
                context=context,
            )
        else:
            response = Prompter.ollama_client.generate(
                prompt=question,
                model=model_type,
                stream=False,
                options=params,
                context=context,
            )
        result = response["response"]
        logprobs = None
        context = response.get("context", None)
        return result, logprobs, context

    @staticmethod
    def _call_llama_cpp_api(template, question, params, llama_cpp_host="http://localhost:11435/v1/chat/completions"):
        payload = {
            "messages": [
                {"role": "system", "content": template},
                {"role": "user", "content": question},
            ],
            "temperature": params.get("temperature", 0),
            "max_tokens": params.get("max_tokens", 100),
            "logprobs": 1,
            "n_probs": 1,
        }
        response = requests.post(llama_cpp_host, json=payload)
        response_data = response.json()

        if response.status_code == 200:
            result = response_data["choices"][0]["message"]["content"]
            logprobs = response_data.get("completion_probabilities", [])
        else:
            raise RuntimeError(f"llama_cpp API error: {response_data.get('error', 'Unknown error')}")

        return result, logprobs

    def prompt(
        self,
        system_prompt_template_path: str,
        question: str,
        use_context: bool = False,
        temperature: float = None,
    ) -> Union[str, List[str]]:
        """Execute a prompt given a system prompt template path and a question."""
        system_prompt_template = None
        if system_prompt_template_path is not None:
            system_prompt_template = pathlib.Path(str(system_prompt_template_path)).read_text(encoding="utf-8")

        if temperature is not None:
            self.params["temperature"] = temperature
        params_tuple = tuple(sorted(self.params.items()))

        print("Cache key:", hash_input(system_prompt_template, question, self.model_type, self.backend, params_tuple, self.context, use_context))
        cached_data = self._cached_prompt_impl(
            template=system_prompt_template,
            question=question,
            model_type=self.model_type,
            backend=self.backend,
            params=params_tuple,
            context=self.context if use_context else None,
            use_context=use_context,
            openai_base_url=getattr(self, "openai_base_url", None),
            api_key=getattr(self, "_api_key", None),
        )

        result = cached_data["outputs"]["result"]
        logprobs = cached_data["outputs"]["logprobs"]

        if use_context:
            self.context = cached_data["inputs"]["context"]

        return result, logprobs
