from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import ServiceContext
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core import Settings
from flask import Flask,jsonify
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import huggingface_hub
import posixpath
import torch
from flask import render_template, request,jsonify
import requests
from llama_index.core import PromptTemplate
import base64
import json
import os
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.core.llms import ChatMessage
import markdown
import re
from werkzeug.utils import secure_filename


# Log in using your access token
huggingface_hub.login(token="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")


app = Flask(__name__)


UPLOAD_FOLDER = 'uploads_new'
# INDEX_FOLDER = 'index'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# System prompt for LLMS
system_prompt = """
    Consider yourself as the representative of Advoice Law Consultance "your creator".Given a question input, your task is to identify relevant keywords,sentences,phrases in the question and retrieve corresponding answers from the context.
    The model should analyze the input question, extract key terms, and search for similar or related questions in the context.The output should provide the answers associated with the identified keywords or closely related topics.
    The model should understand the context of the question, identify relevant keywords,phrases and sentences, and retrieve information from the provided context based on these keywords.
    It should be able to handle variations in question phrasing and retrieve accurate answers accordingly with smart generative answers like a law consultance bot answers to users query.Do not show "relevant keyword fetched" or "from the context provided" or "In the context provided" in the answer simply answer the questions in an intelligent manner.If the passage is out of the context from the documents and also out of the law domain, do not answer and say the provided question is out of law context so i cannot answer that.
    Answer every questions that are asked in max 3 lines.If user greets you then greet them back and if they say goodbye then also say "goodbye".
    Try not to include phrases like "Based on the context provided" or "In the context provided" instead use "According to my knowledge" or "As per Advoice Consultancy " or "as far as I know" give answer in a more generative and smart manner like a law consultance AI bot agent does.
    If the passage is out of the context from the documents say that sorry but i am not allowed to answer outside law domain in a respectful manner.
Context:\n {context}?\n
Question: \n{question}\n
"""
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

llm2 = HuggingFaceLLM(
    context_window=8192,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=SimpleInputPrompt("{query_str}"),
    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    device_map="auto",
    tokenizer_kwargs={"max_length": 4096},
    model_kwargs={"torch_dtype": torch.bfloat16}
)

# Embedding model
embed_model = HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v1", trust_remote_code=True)


# Set settings
Settings.llm = llm2
Settings.embed_model = embed_model


# def translate_hinglish_to_english(text):
#     translator = Translator()
#     detected_lang = translator.detect(text).lang
#     translated_text = translator.translate(text, src='hi', dest='en').text
#     return translated_text

# Format document
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


# Detect language
def detect_language(text):
    if any(ord(char) > 127 for char in text):
        return "hinglish"
    else:
        return "english"

# Process input
# def process_input(input_text):
#     lang = detect_language(input_text)
#     if lang == "english":
#         return "english", input_text
#     elif lang == "hinglish":
#         return "hinglish", translate_hinglish_to_english(input_text)
#     else:
#         print("Unsupported language detected.")
#         return None


def format_to_markdown(text):
  lines = text.strip().split('\n')
  formatted_text = ""
  for line in lines:
    formatted_text += f"- {line.replace('*', '')}\n"
  return formatted_text

documents1 = SimpleDirectoryReader('uploads').load_data()
index = VectorStoreIndex.from_documents(documents1)

memory = ChatMemoryBuffer.from_defaults(token_limit=20000)
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=system_prompt,
    llm=llm2,
    verbose=False
)

def gen_response(question):
    if question=="hi" or question=="Hi" or question =="hi!" or question =="Hi!" or question =="Hello" or question =="hello" or question =="hey" or question =="Hey":
        return "Hello! How can I assist you today?"
    
    # history = chat_engine.chat_history
        
    # chat_history = [ChatMessage(role="user",content=history)]  # Replace with your actual chat history
    response = chat_engine.chat(question)
   
    return str(response)

def clean_response(response):
    pattern = r'^\s*assistant\s*'

    # Use re.sub to replace the pattern with an empty string
    cleaned_response = re.sub(pattern, '', response)
    return cleaned_response



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/service')
def service():
    return render_template('service.html')

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/pdf')
def files():
    return render_template('file.html')

@app.route('/try consulting')
def consulting():
    return render_template('chat.html')


@app.route('/ask_question', methods=['POST'])
def ask_question():
    if request.method == 'POST':
        input_text = request.form['question']
        
        # Assuming these functions are defined elsewhere in your code
        # Modify as necessary to fit your implementation
        response1 = gen_response(input_text)
        cleaned_response = clean_response(response1)
        
        # Process response to HTML
        html_response = markdown.markdown(cleaned_response)
        # soup = BeautifulSoup(html_response, "html.parser")
        
        # # Unwrap all tags to remove any surrounding <p> tags
        # for tag in soup.find_all():
        #     tag.unwrap()
        
        # # Clean HTML using bleach
        # clean_html_response = bleach.clean(str(soup), tags=bleach.sanitizer.ALLOWED_TAGS, strip=True)
        
        return jsonify({'response': html_response})
    else:
        return jsonify({'response': 'Unsupported language detected.'})

@app.route('/convert', methods=['POST'])
def convert_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part"})
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"})

    if file:

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext in {'.txt', '.pdf'}:
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            return jsonify({"success": True, "message": "File uploaded successfully"})
        else:
            return jsonify({"success": False, "message": "Unsupported file format"})



@app.route('/ask pdf', methods=['POST'])
def ask_pdf():
    if request.method == 'POST':
        input_text = request.form['question']
        documents = SimpleDirectoryReader('uploads_new').load_data()
        Settings.llm = llm2
        Settings.embed_model = embed_model
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        
        print("generating response----------------------")
        response = query_engine.query(input_text) 
        print(response)
        return str(response)
 
    else:
        return {'result': 'Unsupported language detected.'}

    
if __name__ == '__main__':
    app.run(debug=False,port=5001)



