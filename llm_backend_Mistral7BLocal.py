from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from time import time
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from transformers import AutoConfig


load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

config = BitsAndBytesConfig(load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    bnb_4bit_compute_dtype = torch.float32) 

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=config,
    )

@app.route('/api/analyze_hate_speech', methods=['POST'])
def analyze_hate_speech():
    try:
        data = request.json
        user_message = data.get('text', '')
        message = user_message
        messages = [
            {"role": "user", "content":"You are a model designed to detect Hate Speech. Answer with counter speech"},
                        {"role": "assistant", "content": f"{message}"}
        ]
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt").to('cpu')
        model_inputs = encodeds


        generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)

        # Use regular expression to remove text between <s> and </s>
        clean_text = re.sub(r'<s>.*?</s>', '', decoded[0][0:])

        print(clean_text.strip()[0:200])
        analysis_result = clean_text.strip()[0:200]
        return jsonify({"analysis_result": analysis_result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6002)

