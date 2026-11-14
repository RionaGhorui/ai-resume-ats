"""Phase 1: ask Groq to turn raw text into structured data (plain Groq SDK, JSON mode)."""
import json
from datetime import date

from groq import Groq
from pydantic import BaseModel, ValidationError

from backend import config
from backend.schemas import ParsedJD, ParsedResume


class LLMError(RuntimeError):
    """The AI service failed. The message is safe to show to the user."""


RESUME_RULES = (
    "You extract structured data from a resume. Use ONLY facts present in the text; never invent anything. "
    "Use an empty string or empty list when something is missing. "
    "'description' = all bullet points of that job/project joined with newlines. "
    "'technologies' = tools named for that project. "
    "'skills' = individual skills from the skills section (not category names). "
    "'action_verbs' = the first word of each experience/project bullet when it is a verb. "
    "'keywords' = up to 25 important technical keywords. "
    "'duration_months' = integer months computed from the dates (today is {today})."
)
JD_RULES = (
    "You extract structured data from a job description. Use ONLY facts present in the text. "
    "'required_skills' and 'preferred_skills' = individual skills/tools (not sentences). "
    "'keywords' = other important terms. 'responsibilities' = up to 8 short duty statements."
)


def _extract(rules: str, text: str, schema: type[BaseModel]) -> BaseModel:
    if not config.GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    client = Groq(api_key=config.GROQ_API_KEY)
    system = (
        rules.format(today=date.today().isoformat())
        + "\nReturn ONLY one JSON object that follows this JSON Schema:\n"
        + json.dumps(schema.model_json_schema())
    )
    extra = {"reasoning_effort": "low"} if config.GROQ_MODEL.startswith("openai/gpt-oss") else {}
    last_error: Exception | None = None
    for _ in range(2):                       # one retry, and only when the first attempt failed
        try:
            resp = client.chat.completions.create(
                model=config.GROQ_MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": text[: config.MAX_LLM_CHARS]},
                ],
                **extra,
            )
            return schema.model_validate_json(resp.choices[0].message.content or "")
        except (ValidationError, ValueError) as exc:     # bad / empty JSON: try once more
            last_error = exc
        except Exception as exc:                          # network, auth, rate limit, retired model...
            raise LLMError(f"Groq request failed ({type(exc).__name__}): {str(exc)[:200]}") from exc
    raise LLMError(f"The AI returned an unreadable answer twice: {str(last_error)[:150]}")


def parse_resume(text: str) -> ParsedResume:
    return _extract(RESUME_RULES, text, ParsedResume)


def parse_job_description(text: str) -> ParsedJD:
    return _extract(JD_RULES, text, ParsedJD)
