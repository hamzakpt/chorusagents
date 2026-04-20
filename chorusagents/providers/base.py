"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """
    Abstract interface for LLM backends.

    Implement this class to add support for any LLM provider.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        **kwargs: Any,
    ) -> str:
        """
        Send a chat completion request and return the response text.

        Parameters
        ----------
        messages:
            List of ``{"role": ..., "content": ...}`` dicts.
        system:
            Optional system prompt (injected before user messages).
        **kwargs:
            Provider-specific keyword arguments.

        Returns
        -------
        str
            The assistant's response text.
        """

    @property
    @abstractmethod
    def model(self) -> str:
        """The model identifier being used."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
