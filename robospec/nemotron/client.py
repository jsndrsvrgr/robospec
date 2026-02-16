"""Async HTTP client for Nemotron via NVIDIA NIM and OpenRouter."""

import os

import httpx
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

console = Console()


class NemotronClient:
    """Make API calls to Nemotron. Tries NIM first, falls back to OpenRouter."""

    NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"

    def __init__(self):
        self.nim_key = os.getenv("NVIDIA_API_KEY")
        self.or_key = os.getenv("OPENROUTER_API_KEY")
        self.client = httpx.AsyncClient(timeout=120.0)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> str:
        """Generate a completion. Try NIM first, fallback to OpenRouter."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.nim_key:
            try:
                return await self._call(
                    self.NIM_URL, self.nim_key, messages, temperature, max_tokens
                )
            except Exception as e:
                console.print(f"[yellow][WARN] NIM failed: {e}, trying OpenRouter...[/yellow]")

        if self.or_key:
            return await self._call(
                self.OPENROUTER_URL, self.or_key, messages, temperature, max_tokens
            )

        raise RuntimeError(
            "No API key set. Set NVIDIA_API_KEY or OPENROUTER_API_KEY in .env"
        )

    async def _call(
        self,
        url: str,
        key: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        resp = await self.client.post(
            url,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def close(self):
        """Close the underlying HTTP client."""
        await self.client.aclose()
