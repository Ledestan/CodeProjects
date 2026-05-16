import json

import requests


class QASystem:
    def __init__(self):
        self.load_knowledge_base()
        self.api_key = "your_deepseek_api_key"  # 需替换
        
    def load_knowledge_base(self):
        """加载本地知识库"""
        with open('data/qa.json', 'r', encoding='utf-8') as f:
            self.qa_data = json.load(f)
    
    def keyword_match(self, question):
        """关键词匹配查找答案"""
        question_lower = question.lower()
        
        for item in self.qa_data:
            # 检查问题是否包含关键词
            for keyword in item['keywords']:
                if keyword in question_lower:
                    return item['answer']
            
            # 直接问题匹配
            if item['question'] in question:
                return item['answer']
        
        return None
    
    def call_deepseek_api(self, question):
        """调用DeepSeek API"""
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建系统提示词
        system_prompt = "你是一位辽宁工业历史专家，请用准确、易懂的语言回答问题。如果问题涉及工业遗产，请结合历史背景详细说明。"
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "max_tokens": 500
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            result = response.json()
            return result['choices'][0]['message']['content']
        except:
            return "抱歉，我现在无法回答这个问题。"
    
    def get_answer(self, question):
        """获取答案的主函数"""
        # 1. 尝试本地匹配
        local_answer = self.keyword_match(question)
        if local_answer:
            return {"answer": local_answer, "source": "local"}
        
        # 2. 调用API
        api_answer = self.call_deepseek_api(question)
        return {"answer": api_answer, "source": "api"}