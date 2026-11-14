"""Phase 2: LangChain does the prompting, structured output and validation.

Compare with Phase 1: no hand-written JSON schema in the prompt, no json.loads,
no retry loop. `with_structured_output(PydanticModel)` returns a validated object.
"""
from datetime import date

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, ValidationError

from backend import config
from backend.schemas import ParsedJD, ParsedResume


class LLMError(RuntimeError):
    """The AI service failed. The message is safe to show to the user."""


RESUME_RULES = (
    "You extract structured JSON data from a resume. Use ONLY facts present in the text; never invent anything. "
    "Use an empty string or empty list when something is missing. "
    "'description' = all bullet points of that job/project joined with newlines. "
    "'technologies' = tools named for that project. "
    "'skills' = individual skills from the skills section (not category names). "
    "'action_verbs' = the first word of each experience/project bullet when it is a verb. "
    "'keywords' = up to 25 important technical keywords. "
    "'duration_months' = integer months computed from the dates (today is {today})."
)
JD_RULES = (
    "You extract structured JSON data from a job description. Use ONLY facts present in the text. "
    "'required_skills' and 'preferred_skills' = individual skills/tools (not sentences). "
    "'keywords' = other important terms. 'responsibilities' = up to 8 short duty statements."
)


def _llm() -> ChatGroq:
    if not config.GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    extra = {"reasoning_effort": "low"} if config.GROQ_MODEL.startswith("openai/gpt-oss") else {}
    return ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY,
                    temperature=0, max_retries=2, **extra)


def _structured_llm(schema: type[BaseModel]):
    """A model that always answers with `schema`. Separate function so tests can replace it."""
    return _llm().with_structured_output(schema, method=config.GROQ_STRUCTURED_METHOD)


def run_structured(system: str, human: str, schema: type[BaseModel], **variables) -> BaseModel:
    """prompt | structured LLM, retried once if the model returns unusable output."""
    chain = (ChatPromptTemplate.from_messages([("system", system), ("human", human)])
             | _structured_llm(schema)).with_retry(
        retry_if_exception_type=(OutputParserException, ValidationError), stop_after_attempt=2)
    try:
        return chain.invoke(variables)
    except LLMError:
        raise
    except Exception as exc:       # network, auth, rate limit, retired model...
        raise LLMError(f"Groq request failed ({type(exc).__name__}): {str(exc)[:200]}") from exc


def parse_resume(text: str) -> ParsedResume:
    return run_structured(RESUME_RULES, "{text}", ParsedResume,
                          text=text[: config.MAX_LLM_CHARS], today=date.today().isoformat())


def parse_job_description(text: str) -> ParsedJD:
    return run_structured(JD_RULES, "{text}", ParsedJD, text=text[: config.MAX_LLM_CHARS])
