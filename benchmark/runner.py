from utils.parser import parse_api_call, parse_answer

class BenchmarkRunner:

    def __init__(self, renderer, agent, max_steps=10):

        self.renderer = renderer
        self.agent = agent
        self.max_steps = max_steps


    def run(self, question):

        msg = question
        img = None

        for step in range(self.max_steps):

            reply = self.agent.ask(msg,image=img)

            print("第 {} 轮回答 ||{}||".format(step,reply))

            api_call = parse_api_call(reply)

            if api_call:

                cam,target = api_call

                direction = (target[0] - cam[0], target[1] - cam[1], target[2] - cam[2])
                is_parallel_to_up = abs(direction[0]) < 1e-5 and abs(direction[2]) < 1e-5

                if is_parallel_to_up:
                    cam = (cam[0] + 0.01, cam[1], cam[2])

                img = self.renderer.render(cam,target)

                print("\n\n第 {} 次渲染已返回".format(step+1))

                msg = "Observation"

                continue


            else:

                ans = parse_answer(reply)

                if not ans:
                    ans = "没有API调用也没有回答"

                return ans

        return "FAILED"