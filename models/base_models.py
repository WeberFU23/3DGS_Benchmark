from openai import OpenAI
from utils.image_utils import encode_image


class BaseLLMAgent:
    """
    统一的 AI Agent 基类，支持多轮对话和图片输入。
    兼容所有提供 OpenAI 格式接口的平台（OpenAI, Qwen, Zhipu, DeepSeek等）。
    """

    def __init__(self, api_key, base_url, model, system_prompt):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.messages = [{"role": "system", "content": system_prompt}]

    def ask(self, text=None, image=None):
        # 1. 处理输入内容
        if image:
            base64_img = encode_image(image)
            user_content = [
                {"type": "text", "text": text if text else "描述这张图片"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                }
            ]
        else:
            user_content = text

        # 2. 更新记忆
        self.messages.append({"role": "user", "content": user_content})

        # 3. 请求模型
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages
            )
            reply = resp.choices[0].message.content

            # 4. 记录助手回答
            self.messages.append({"role": "assistant", "content": reply})
            return reply

        except Exception as e:
            return f"Error calling {self.model}: {str(e)}"


# --- 具体子类实现 ---

class GPTAgent(BaseLLMAgent):
    def __init__(self, api_key, system_prompt, model="gpt-4o"):
        super().__init__(api_key, None, model, system_prompt)


class QwenAgent(BaseLLMAgent):
    def __init__(self, api_key, system_prompt, model="qwen-vl-plus"):
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        super().__init__(api_key, base_url, model, system_prompt)


class ZhipuAgent(BaseLLMAgent):
    def __init__(self, api_key, system_prompt, model="glm-4v-plus"):
        base_url = "https://open.bigmodel.cn/api/paas/v4/"
        super().__init__(api_key, base_url, model, system_prompt)