def final_ans(text: str) -> str | None:
    """
    Grab the substring after the last occurrence of patterns like
    '####', 'Final Answer:', 'Answer:'. Returns None if not found.
    """
    import re
    pat = re.compile(r"(?:####|[Ff]inal [Aa]nswer[:\-]?|[Aa]nswer[:\-]?)\s*(.*)")
    matches = pat.findall(text)
    return matches[-1].strip() if matches else None 