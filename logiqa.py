def process_logiqa(path):
    chunks = open(path).read().split("\n\n")
    chunks = [x.strip() for x in chunks if x.strip()]
    qa = [x.split("\n", 1) for x in chunks]
    return [{"question": q, "answer": a} for a, q in qa]


def create_mc_prompt(question, choices):
    result = question + "\n"
    for i, c in enumerate(choices):
        result += f"({chr(ord('A') + i)}) {c}\n"

    return result.strip()
