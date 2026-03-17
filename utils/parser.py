import re

def parse_api_call(text):

    pattern = r"API\s*\(\(\s*([^)]+)\),\s*\(([^)]+)\)\)"
    m = re.search(pattern, text)

    if not m:
        return None

    cam = [float(x) for x in m.group(1).split(",")]
    tgt = [float(x) for x in m.group(2).split(",")]

    return cam, tgt


def parse_answer(text):

    pattern = r"Answer\((.*)\)"
    m = re.search(pattern, text)

    if m:
        return m.group(1)

    return None