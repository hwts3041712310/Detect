import torch
from detect_2d.detect import Detect
from http import HTTPStatus
import os
import cv2
import dashscope
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 设置API密钥
# dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key ="sk-9c35db928dce4d8891c7f54a6af082c6"


def generate_messages(rec_info, image_path):
    # 创建一个空的文本消息列表
    messages = []

    # 构建一个包含所有问题的文本消息
    questions_text = ""
    for idx, (box_id, color_name) in enumerate(rec_info):
        # 生成当前框的问题文本
        question_text = f"框的左上方的标签为{color_name}的{box_id}， 颜色为{color_name}的框是否为小广告\n"
        # 将当前框的问题文本添加到总问题文本中
        questions_text += question_text

    # 创建包含所有问题的文本消息
    content = [
        {"image": image_path},
        {"text": "根据总体场景判断，红色的框中是不是小广告"}
    ]
    # 将文本消息添加到消息列表中
    messages.append({"role": "user", "content": content})

    return messages

def simple_multimodal_conversation_call(rec_info, image_path):
    """Simple single round multimodal conversation call."""
    messages = generate_messages(rec_info, image_path)
    response = dashscope.MultiModalConversation.call(model='qwen-vl-max', messages=messages)

    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print(response.code)  # The error code.
        print(response.message)  # The error message.

class ModelRunner:
    def __init__(self):
        # 初始化检测模型
        self.det_model = Detect('detect_2d/weight/advertise.torchscript', device_idx=torch.device('cpu'))

    def run_model(self, img):
        rec_info = self.det_model(img)
        if not rec_info:
            return
        simple_multimodal_conversation_call(rec_info, 'detect.png')

if __name__ == '__main__':
    # 创建类的实例
    runner = ModelRunner()
    # 加载输入数据
    source_dir = 'images'
    for idx, img_name in enumerate(os.listdir(source_dir)):

        img_path = os.path.join(source_dir, img_name)
        img_array = cv2.imread(img_path)
        runner.run_model(img_array)