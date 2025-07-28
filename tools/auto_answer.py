"""Automatic question answering helper."""

from __future__ import annotations

from typing import Optional
from AgentModule.edu_agent import create_agent, run_agent, AgentExecutor
from tools.language_handler import LanguageHandler


def auto_answer(text: str, agent: Optional[AgentExecutor] = None) -> bool:
    """Run the agent if ``text`` looks like a question.

    This triggers when ``text`` ends with a question mark. If activated, the
    agent's reply is printed and ``True`` is returned. Otherwise ``False`` is
    returned.
    """
    if text.strip().endswith("?"):
        agent = agent or create_agent()
        language = LanguageHandler.choose_or_detect(text)
        answer = run_agent(text, executor=agent)
        answer = LanguageHandler.ensure_language(answer, language)
        print(f"\n\U0001f916 Agent Answer:\n{answer}\n")
        return True
    return False
