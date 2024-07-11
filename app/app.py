from flask import Flask, render_template, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.memory import ConversationBufferMemory

app = Flask(__name__)

# Global variables
api_key = None
llm = None
chain = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_api_key', methods=['POST'])
def set_api_key():
    global api_key, llm, chain
    api_key = request.json['api_key']
    
    # Initialize LLM and chain
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    
    system_template = """You are an AI assistant specialized in analyzing messages for hidden content or meanings. Your task is to:
    1. Carefully examine the given message.
    2. Identify any potential hidden messages, double meanings, or subtle implications.
    3. Be prepared to explain your analysis when asked.

    Message to analyze: {message}

    {chat_history}
    Human: {human_input}
    AI: """

    human_template = "{human_input}"

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])

    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history", input_key="human_input")

    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
        memory=memory,
        verbose=True
    )
    
    return jsonify({"success": True})

@app.route('/analyze', methods=['POST'])
def analyze():
    message = request.json['message']
    query = "Analyze this message for any hidden meanings or content."
    response = chain.run(message=message, human_input=query)
    # Store the original message in the memory
    chain.memory.chat_memory.add_user_message(f"Original message to analyze: {message}")
    return jsonify({"response": response})



@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    # Retrieve the original message to analyze from the memory
    original_message = chain.memory.chat_memory.messages[0].content if chain.memory.chat_memory.messages else ""
    response = chain.run(message=original_message, human_input=message)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)