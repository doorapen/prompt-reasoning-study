import json, glob

def load_prompts(path="prompts/"):
    templates = {}
    for fn in glob.glob(path + "*.json"):
        j = json.load(open(fn, "r"))
        templates[j["family"]] = j["template"]
    return templates
