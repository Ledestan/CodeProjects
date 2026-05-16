"""
团队名称: 数溯工忆
项目名称: 工业遗产数字活化：AI+AR技术赋能实践
启动时间: 2026-01-12

依赖库:
flask>=3.1.2
numpy>=2.2.6
opencv-python>=4.12.0.88
pillow>=12.1.0
requests>=2.32.5
"""


import traceback

from flask import Flask, jsonify, render_template, request
from image_recognizer import ImageRecognizer
from qa_engine import QASystem

app = Flask(__name__)
qa_system = QASystem()
recognizer = ImageRecognizer()

@app.route('/')
def index():
    """网站首页"""
    return render_template('index.html')

@app.route('/chat')
def chat_page():
    """智能问答页面"""
    return render_template('chat.html')

@app.route('/recognize')
def recognize_page():
    """图片识别页面"""
    return render_template('recognize.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """问答API接口"""
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({"error": "问题不能为空"})
    
    result = qa_system.get_answer(question)
    return jsonify(result)

@app.route('/api/recognize', methods=['POST'])
def recognize_image():
    """图片识别API接口"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "请上传图片文件"})
        
        file = request.files['image']
        
        # 检查文件类型
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({"error": "仅支持PNG、JPG格式图片"})
        
        # 检查文件大小（限制2MB）
        file.seek(0, 2)  # 移动到文件末尾
        file_size = file.tell()
        file.seek(0)  # 移回文件开头
        
        if file_size > 2 * 1024 * 1024:  # 2MB
            return jsonify({"error": "图片大小不能超过2MB"})
        
        # 读取图片数据
        image_data = file.read()
        
        # 识别图片
        result = recognizer.recognize(image_data)
        
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"识别失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"识别失败: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)