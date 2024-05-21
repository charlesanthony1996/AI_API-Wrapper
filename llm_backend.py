from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

import json5
import re

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.chains import RetrievalQA

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# load config
with open('config.json') as config_file:
    config = json.load(config_file)

# env variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPEN_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

examples = [
    {"question": "Is this Hate Speech: 'Cultures have never been fair and this is a fact. Muslims for example are just a bunch of misogynist monkeys!' Answer a. Yes or b. No, Answer:", "explanation":"This statement would generally be considered hate speech. It's derogatory and targets a specific group of people based on their religion and ethnicity, using offensive language to degrade and dehumanize them.", "output":" The answer is a. Yes", "hs_type": "islamophobia"},
    {"question": "Is this Hate Speech: 'You deserve way more subscribers. The s*** you say is f***ing gold.' Answer a. Yes or b. No, Answer:", "explanation":"This statement is not hate speech. It's a compliment expressing admiration for someone's content and suggesting they deserve more recognition or subscribers.", "output":" The answer is b. No", "hs_type": "None"},
    {"question": "Is this Hate Speech: 'I went to a shit load of therapy. I still don't feel safe in my own house with all these black people around.' Answer a. Yes or b. No, Answer:", "explanation":"That statement would be considered hate speech. It contains racial prejudice and derogatory language towards a specific racial group, implying that their presence is inherently threatening or unsafe.", "output":" The answer is a. Yes", "hs_type": "anti-black racism"},
    {"question": "Is this Hate Speech: 'Because people becomes evil easily, it is in our genes' . Answer a. Yes or b. No, Answer:", "explanation":" This statement does not constitute hate speech on its own. It seems to express a belief about human behavior and genetics. ", "output":" The answer is b. No", "hs_type": "None"},
    {"question": "Is this Hate Speech: 'Asians are basically just a bunch of slobs and I don't like how they are accepted here.' Answer a. Yes or b. No, Answer:", "explanation":"This statement is considered hate speech. It makes a derogatory generalization about a specific racial group and expresses dislike or resentment towards their acceptance in a certain community.", "output":" The answer is a. Yes", "hs_type": "anti-asian racism"},
    {"question": "Is this Hate Speech: 'Freedom of speech is under attack. It's being taken away by Zionist Occupation Government.' Answer a. Yes or b. No, Answer:", "explanation":"That statement is considered hate speech. It perpetuates a conspiracy theory about a supposed 'Zionist Occupation Government' controlling or suppressing freedom of speech, which can contribute to anti-Semitic sentiments. ", "output":" The answer is a. Yes", "hs_type": "antisemitism"},
    {"question": "Is this Hate Speech: 'Gypsies are literally destroying Europe. They are literally invading and destroying our beautiful continent. They are a cancer on our race and they must be stopped' Answer a. Yes or b. No, Answer:", "explanation":"This statement is considered hate speech. It dehumanizes and vilifies an entire ethnic group, using inflammatory language like 'destroying,' 'invading,' and comparing them to a 'cancer.'", "output":" The answer is a. Yes", "hs_type": "anti-gypsyism"},
    {"question": "Is this Hate Speech: 'I think the whole equality thing is getting out of hand. We are different, that’s how were made!' Answer a. Yes or b. No, Answer:", "explanation":"This statement is considered hate speech. Yes, this statement can be considered sexist. The idea that men and women are fundamentally different and should therefore have different rights or opportunities is a core belief of sexism.", "output":" The answer is a. Yes", "hs_type": "sexism"},
    {"question": "Is this Hate Speech: 'We should isolate gays because they are spreading AIDS all around the world.' Answer a. Yes or b. No, Answer:", "explanation":"This statement is considered hate speech. It suggests that trans actors should not be allowed to play certain roles in movies, implying that they are not capable or qualified to do so because of their gender identity. ", "output":" The answer is a. Yes", "hs_type": "trans-hostility"},
    {"question": "Is this Hate Speech: 'The real problem is that fat people do not want to change' Answer a. Yes or b. No, Answer:", "explanation":"This statement is considered hate speech. It suggests that fat people are lazy or unwilling to change, which is a harmful stereotype that can perpetuate discrimination and stigmatization of individuals who are overweight or obese.", "output":" The answer is a. Yes", "hs_type": "fatphobia"},
    {"question": "Is this Hate Speech: 'Downy children should be isolated, they cannot cope with education in normal schools and are detrimental to their peer's education.' Answer a. Yes or b. No, Answer:", "explanation":"This statement is considered hate speech. It implies that children with Down syndrome are detrimental to their peers' education, which is a baseless and harmful assumption. This contributes to a culture of exclusion and marginalization of individuals with disabilities.", "output":" The answer is a. Yes", "hs_type": "ableism"},
    {"question": "Is this Hate Speech: 'Foreigners on UK benefits leaps 41% in 5 years with over 400K now handed payouts. This nonsense has got to stop' Answer a. Yes or b. No, Answer:", "explanation":"This statement is considered hate speech. It suggests that foreigners or immigrants in the UK are a burden on the welfare system and that they are taking advantage of benefits that should be reserved for UK citizens", "output":" The answer is a. Yes", "hs_type": "hostility towards migrants"},
    {"question": "Is this Hate Speech: 'You sad fuck. I'm going to restore my edit, and you are going to leave it, or else I will knee you in the groin yourself, prick.' Answer a. Yes or b. No, Answer:", "explanation":"This statement is considered hate speech. It is using offensive and threatening language to attack and intimidate a specific person based on their perceived identity or behavior.", "output":" The answer is a. Yes", "hs_type": "cyberbullying"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{question}"),
        ("ai", "{output}"),
        ("ai", "{explanation}"),
        ("ai", "{hs_type}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

final_prompt = ChatPromptTemplate.from_messages(
[
    #("system", "You are a model designed to detect Hate Speech using only the context and examples provided. Please answer a. Yes or b. No in the following JSON-format: {{'Answer': ''}}",),
    ("system", "You are a model designed to detect Hate Speech using only the context and examples provided."),
    few_shot_prompt,
    ("human", "{context}"),
    ("human", "{question}"),
]
)

loader = TextLoader("./Definitions_of_Different_Forms_of_Hate_Speech.md")
data = loader.load()
definitions_markdown = data[0].page_content

headers_to_split_on = [
("#", "Header 1"),
("##", "Header 2"),
]

text_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

data = text_splitter.split_text(definitions_markdown)

embeddings = OpenAIEmbeddings()

store = Chroma.from_documents(
data,
embeddings,
collection_name='hate_speech_definitions',
persist_directory = 'db',
)

store.persist()

llm_groq = ChatGroq(
#temperature=0.7,
model_name = "mixtral-8x7b-32768",
model_kwargs={"top_p":0.35}
)

qa_with_source_few_shot = RetrievalQA.from_chain_type(
llm=llm_groq,
chain_type='stuff',
retriever = store.as_retriever(),
chain_type_kwargs={'prompt': final_prompt},
return_source_documents = True,
)

def llm_prompting_calls(message):
    #result = qa_with_source_few_shot.invoke(f"Do you think this comment is hate speech? Comments: {message}. Please answer a. Yes or b. No for each comment. Format the response in the following JSON-Format: {{\"Comment\": \"\", \"Answer\": \"\",  \"HS_type\": \"\"}}")        
    #result = re.sub('\\\\', '', result["result"])
    #result = re.sub('\\n','', result)
    #result = re.sub('\\n','', result)
    #result = re.sub('""', '"', result) # to check for issue of double quotes

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Detect hate speech and respond 'Yes' or if no hate speech respond 'No hate speech detected.'"},
            {"role": "user", "content": message}
        ],
    )

# base class init here
class BaseLLMClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def create_completion(self, prompt):
        pass
    def create_completionCS(self, prompt):
        pass

class OpenAIClient(BaseLLMClient):
    def create_completion(self, prompt):
        client = OpenAI(api_key=self.api_key)
        return client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Detect hate speech and respond 'Yes' or if no hate speech respond 'No hate speech detected.'"},
                {"role": "user", "content": prompt}
            ],
        )
    def create_completionCS(self, prompt):
        client = OpenAI(api_key=self.api_key)
        return client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate counter speech"},
                {"role": "user", "content": prompt}
            ],
        )


class GroqClient(BaseLLMClient):
    def create_completion(self, prompt):
        client = Groq(api_key=self.api_key)
        return client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Detect hate speech and respond 'Yes' or if no hate speech respond 'No hate speech detected.'"},
                {"role": "user", "content": prompt}
            ],
            model="mixtral-8x7b-32768",
            temperature=0.5,
            max_tokens=150,
            top_p=1,
            stop=None,
            stream=False,
        )
    
    def create_completionCS(self, prompt):
        client = Groq(api_key=self.api_key)
        return client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Generate counter speech"},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=150,
            top_p=1,
            stop=None,
            stream=False,
        )
    
    def explain_HS(self, prompt):
        client = Groq(api_key=self.api_key)
        return client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a model designed to explain why a given text is hate speech. Please provide an explanation."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=150,
            top_p=1,
            stop=None,
            stream=False,
        )


if config['llm_type'] == 'openAI':
    client = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY'))
elif config['llm_type'] == 'groq':
    client = GroqClient(api_key=os.getenv('GROQ_API_KEY'))
else:
    raise ValueError("Unsupported LLM type specified in config.")
    
@app.route('/api/analyze_hate_speech', methods=['POST'])
def analyze_hate_speech():
    try:
        data = request.json
        user_message = data.get('text', '')
        llm_response = llm_prompting_calls(user_message)
        
        llm_result = llm_response.choices[0].message.content.strip()
        #llm_response = json5.loads(llm_response) 
        

        return jsonify({"filtered_text": llm_result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/generate_counter_speech', methods=['POST'])
def generate_counter_speech():
    try:
        data = request.json
        user_message = data.get('text', '')
        response = client.create_completionCS(user_message)
        counter_speech_result = response.choices[0].message.content.strip()
        return jsonify({"counter_speech_result": counter_speech_result}), 200

    except Exception as e:  
        return jsonify({"error": str(e)}), 700

@app.route('/api/explain_hate_speech', methods=['POST'])
def explain_hate_speech():
    try:
        data = request.json
        user_message = data.get('text', '')
        response = client.explain_HS(user_message)
        explanation_result = response.choices[0].message.content.strip()
        return jsonify({"explanation_result": explanation_result}), 200

    except Exception as e:  
        return jsonify({"error": str(e)}), 700

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6001)


# @app.route('/api/analyze_hate_speech', methods=['POST'])
# def analyze_hate_speech():
#     try:
#         data = request.json
#         user_message = data.get('text', '')
#         response_pre = client.create_completion(user_message)
#         analysis_result_pre = response_pre.choices[0].message.content.strip()
#         print("analysis result: ", analysis_result_pre)
#         if(analysis_result_pre == 'Yes'):
#             response = client.create_completionCS(analysis_result_pre)
#             analysis_result = response.choices[0].message.content.strip()
#         else:
#             analysis_result = analysis_result_pre
#         return jsonify({"analysis_result": analysis_result}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500