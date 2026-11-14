"""Term matching: aliases (JS = JavaScript), whole-word search, fuzzy fallback."""
import re

from rapidfuzz import fuzz

ALIASES = {
    "js": "javascript", "es6": "javascript", "ts": "typescript",
    "reactjs": "react", "react.js": "react", "nodejs": "node.js", "node": "node.js",
    "vuejs": "vue", "vue.js": "vue", "angularjs": "angular", "nextjs": "next.js",
    "expressjs": "express", "express.js": "express",
    "py": "python", "python3": "python",
    "postgres": "postgresql", "psql": "postgresql", "mongo": "mongodb",
    "k8s": "kubernetes", "gcp": "google cloud", "amazon web services": "aws",
    "ml": "machine learning", "dl": "deep learning", "ai": "artificial intelligence",
    "nlp": "natural language processing", "cv": "computer vision",
    "sklearn": "scikit-learn", "scikit learn": "scikit-learn", "tf": "tensorflow",
    "cicd": "ci/cd", "ci cd": "ci/cd",
    "rest api": "rest", "rest apis": "rest", "restful": "rest", "restful api": "rest",
    "restful apis": "rest", "oop": "object-oriented programming",
    "oops": "object-oriented programming", "dsa": "data structures and algorithms",
    "html5": "html", "css3": "css", "golang": "go", "c sharp": "c#", "dotnet": ".net",
    "genai": "generative ai", "gen ai": "generative ai", "llms": "llm",
    "large language model": "llm", "large language models": "llm",
    "rag": "retrieval-augmented generation",
}
_REVERSE: dict[str, set[str]] = {}
for _alias, _canon in ALIASES.items():
    _REVERSE.setdefault(_canon, set()).add(_alias)


def normalize(term: str) -> str:
    t = re.sub(r"\s+", " ", term.strip().lower()).strip(" ,;:()[]").rstrip(".")
    return ALIASES.get(t, t)


def variants(term: str) -> set[str]:
    n = normalize(term)
    return {n, term.strip().lower()} | _REVERSE.get(n, set())


def contains_term(text_lower: str, term: str) -> bool:
    """Whole-word search, so 'go' does not match 'good' and 'java' not 'javascript'."""
    for v in variants(term):
        if v and re.search(rf"(?<![a-z0-9]){re.escape(v)}(?![a-z0-9])", text_lower):
            return True
    return False


def dedupe(terms: list[str]) -> list[str]:
    seen, out = set(), []
    for t in terms:
        key = normalize(t)
        if key and key not in seen:
            seen.add(key)
            out.append(t.strip())
    return out


def match_terms(terms: list[str], resume_terms: list[str], resume_text: str):
    """Split `terms` into (matched, missing) by looking at the resume's skill list and raw text."""
    text = resume_text.lower()
    resume_norm = {normalize(t) for t in resume_terms}
    matched, missing = [], []
    for term in dedupe(terms):
        n = normalize(term)
        hit = (
            n in resume_norm
            or contains_term(text, term)
            or any(fuzz.ratio(n, r) >= 88 for r in resume_norm)
        )
        (matched if hit else missing).append(term)
    return matched, missing
