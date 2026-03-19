import json

from api.renderer import Renderer3DGS
from models.base_models import QwenAgent
from benchmark.runner import BenchmarkRunner

api_key = {
    "Qwen": "",
    "Zhipu": "",
}

def main():

    with open("prompts/system_prompt.txt","r", encoding='utf-8') as f:
        prompt = f.read()

    with open("dataset/questions.json","r", encoding='utf-8') as f:
        dataset = json.load(f)

    results = []

    for scene_data in dataset:

        ply = scene_data["scene"]

        print("\n\n====== Loading Scene ======")
        print(ply)

        renderer = Renderer3DGS(ply)

        for q in scene_data["questions"]:

            question = q["question"]

            print("\n\nQuestion\n\n:", question)

            agent = QwenAgent(api_key["Qwen"],prompt)
            runner = BenchmarkRunner(renderer, agent)

            ans = runner.run(question)

            print("\n\nAnswer:", ans)

            results.append({
                "scene": ply,
                "question": question,
                "pred": ans,
                "gt": q.get("answer", None)
            })

    with open("results.json","w") as f:
        json.dump(results,f,indent=2)


if __name__ == "__main__":
    main()