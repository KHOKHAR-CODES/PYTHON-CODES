import json
from pathlib import Path


json_path = Path(__file__).with_name("sentence.json")

with json_path.open(encoding="utf-8") as file:
    sentences = json.load(file)

for sentence in sentences:
    print(sentence)

print("total length:", len(sentences))