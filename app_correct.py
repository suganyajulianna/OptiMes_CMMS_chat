from flask import Flask, request, jsonify, render_template
import logging
import os
from pymongo import MongoClient
import boto3
from dotenv import load_dotenv
from flask_cors import CORS
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline, T5ForConditionalGeneration, T5Tokenizer
import fitz  # PyMuPDF
import io
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from collections import Counter
from sentence_transformers import util
import re

# Download nltk punkt tokenizer once
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__, template_folder="templates")
CORS(app)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("MONGO_DB")
collection_name = os.getenv("MONGO_COLLECTION")
client_mongo = MongoClient(mongo_uri)
db = client_mongo[db_name]
collection = db[collection_name]

# AWS S3 setup
aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_key = os.getenv("AWS_SECRET_KEY")
region_name = os.getenv("AWS_REGION")
s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=region_name)

# Load models
tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model_bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

qg_tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-base-qg-hl")
qg_model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-qg-hl")

qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
qg_tokenizer_large = T5Tokenizer.from_pretrained("lmqg/t5-base-squad-qg")
qg_model_large = T5ForConditionalGeneration.from_pretrained("lmqg/t5-base-squad-qg")

current_machine_id = None
machine_summary = ""
auto_generated_qa = []
custom_query_answers = []

def reset_machine_state():
    global machine_summary, auto_generated_qa, custom_query_answers
    machine_summary = ""
    auto_generated_qa = []
    custom_query_answers = []
    logger.info("Machine state reset: Cleared summary and QA lists.")


def generate_contextual_questions(text, question_model, tokenizer, keywords, top_n=5):
    sentences = sent_tokenize(text)
    filtered_sentences = filter_sentences_by_keywords(sentences, keywords)
    highlight_sentences = filtered_sentences[:top_n * 2]  # Take more to filter later

    questions = []
    for sent in highlight_sentences:
        prompt = f"generate question: {sent}"
        input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True)
        outputs = question_model.generate(input_ids, max_length=64, num_beams=5, early_stopping=True)
        question = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Semantic check
        if len(question) > 10 and "?" in question and is_semantically_relevant(question, keywords):
            questions.append(question)

    # Clean and deduplicate
    return clean_questions(questions)

def generate_template_questions(text):
    install_steps = extract_installation_steps(text)
    questions = []
    for step in install_steps[:5]:
        step_num = step.split('.')[0]
        step_action = ' '.join(step.split('.')[1:]).strip()
        questions.append(f"What is done in {step_num}? (e.g., {step_action})")
    return questions

def get_manual_link(machine_id):
    doc = collection.find_one({"_id": machine_id})
    if doc:
        return doc.get("manual_link")
    return None


def clean_repeated_lines(text):
    lines = text.split('\n')
    seen = set()
    cleaned = []
    for line in lines:
        line_clean = line.strip()
        if line_clean and line_clean not in seen:
            seen.add(line_clean)
            cleaned.append(line_clean)
    return '\n'.join(cleaned)

def clean_pdf_text(text):
    # Remove page numbers like "Page 45 of 84"
    text = re.sub(r'Page\s*\d+\s*of\s*\d+', '', text, flags=re.IGNORECASE)

    # Remove image/figure placeholders
    text = re.sub(r'\b(Figure|Fig\.|Image|Illustration|Diagram|Photo)\s*\d*[:.]?', '', text, flags=re.IGNORECASE)

    # Remove weird characters like bullets, formatting artifacts, or mathematical symbols
    text = re.sub(r'[•]+', '', text)
    text = re.sub(r'[-–—]{2,}', ' ', text)  # multiple dashes

    # Remove symbols like '800XM/', standalone short words or sequences
    text = re.sub(r'\b[0-9A-Z]{2,}[\/\\]?\b', '', text)

    # Remove extra whitespace or line breaks
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


def extract_keywords(text, top_n=20):
    stop_words = set(stopwords.words('english'))
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered_words = [w for w in words if w not in stop_words]
    word_counts = Counter(filtered_words)
    most_common = word_counts.most_common(top_n)
    keywords = [word for word, count in most_common]
    return keywords

def clean_questions(questions):
    seen = set()
    cleaned = []
    for q in questions:
        q = q.strip()
        if q and q not in seen and len(q) > 10 and "?" in q:
            seen.add(q)
            cleaned.append(q)
    return cleaned[:6]


def extract_text_from_s3_pdf(s3_link):
    try:
        bucket_name, key = s3_link.replace("s3://", "").split("/", 1)
        pdf_obj = s3.get_object(Bucket=bucket_name, Key=key)
        pdf_stream = io.BytesIO(pdf_obj['Body'].read())
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        cleaned_text = clean_pdf_text(text)
        return cleaned_text
    except Exception as e:
        logger.error(f"Failed to extract PDF text: {e}")
        return None

def summarize_text(text, max_length=512):
    inputs = tokenizer_bart.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model_bart.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
    summary = tokenizer_bart.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def filter_sentences_by_keywords(sentences, keywords):
    filtered = []
    for sent in sentences:
        if any(kw.lower() in sent.lower() for kw in keywords):
            if 8 < len(sent.split()) < 40:
                filtered.append(sent)
    return filtered


def summarize_large_text(text, chunk_size=1000, max_length=150):
    sentences = sent_tokenize(text)
    chunks, chunk = [], []
    count = 0
    for sent in sentences:
        count += len(sent.split())
        chunk.append(sent)
        if count >= chunk_size:
            chunks.append(" ".join(chunk))
            chunk, count = [], 0
    if chunk:
        chunks.append(" ".join(chunk))
    summary_all = []
    for chunk_text in chunks:
        inputs = tokenizer_bart.encode(chunk_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model_bart.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
        summary = tokenizer_bart.decode(summary_ids[0], skip_special_tokens=True)
        summary_all.append(summary)
    return " ".join(summary_all)

def is_large_manual(text, threshold=5000):
    return len(text.split()) > threshold

def filter_relevant_sentences(sentences, keywords):
    filtered = []
    for sent in sentences:
        # Remove page number lines
        if re.search(r'page\s*\d+\s*of\s*\d+', sent, re.IGNORECASE):
            continue
        # Remove lines that contain image placeholders or references
        if re.search(r'(image|fig\.|figure|photo|diagram|illustration)', sent, re.IGNORECASE):
            continue
        # Sentence must contain any keyword and length check (avoid very short/long)
        if any(kw.lower() in sent.lower() for kw in keywords) and 8 < len(sent.split()) < 40:
            # Strip leading/trailing spaces and add
            filtered.append(sent.strip())
    return filtered



def normalize_question(question):
    question = question.lower().strip()
    if question.startswith("explain") or question.startswith("define"):
        return f"what is {question.split(' ', 1)[1]}"
    return question

def process_answers_to_bullets(answers):
    bullets = []
    seen = set()
    for ans in answers:
        ans = clean_pdf_text(ans.strip())
        if ans and ans not in seen:
            seen.add(ans)
            if not ans[0].isupper():
                ans = ans[0].upper() + ans[1:]
            if ans[-1] not in ['.', '?', '!']:
                ans += '.'
            bullets.append(f"• {ans}")
    return '\n'.join(bullets)

# New enhanced question generation logic with section segmentation
def highlight_keywords_for_qg(text, keywords):
    for kw in keywords:
        pattern = re.compile(rf'\b({re.escape(kw)})\b', re.IGNORECASE)
        text = pattern.sub(r'<hl> \1 <hl>', text)
    return text

def is_valid_step(text):
    text = text.lower()
    return (
        "install" in text
        or "connect" in text
        or "turn on" in text
        or "insert" in text
        or "place" in text
        or "remove" in text
        or "attach" in text
        or "detach" in text
        or "power on" in text
        or "power off" in text
        or "plug" in text
        or "unplug" in text
    )

def is_semantically_relevant(question, keywords):
    q_words = set(re.findall(r'\b[a-z]{3,}\b', question.lower()))
    k_words = set(keywords)
    overlap = q_words & k_words
    return len(overlap) > 0


def extract_installation_steps(text):
    pattern = r"(Step\s+\d+\.)(.*?)(?=Step\s+\d+\.|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)
    steps = []
    for step_num, action in matches:
        clean_action = action.strip().replace('\n', ' ')
        full_step = f"{step_num.strip()} {clean_action}"
        if len(full_step.split()) > 5 and is_valid_step(full_step):
            steps.append(full_step)
    return steps

def search_manual_content(text, query, max_results=5):
    query = query.lower().strip()

    # Define keyword groups for query intent
    installation_keywords = ["install", "installation", "steps", "how to", "setup", "configure", "connect", "insert", "attach", "remove", "fix", "repair", "replace"]
    troubleshooting_keywords = ["error", "issue", "problem", "fail", "troubleshoot", "fault", "stop", "not working"]
    explanation_keywords = ["what is", "explain", "define", "meaning of", "describe"]

    sentences = sent_tokenize(text)

    # Prioritize installation steps extraction if installation-related
    if any(kw in query for kw in installation_keywords):
        steps = extract_installation_steps(text)
        if steps:
            # Filter steps that contain any query keyword for better precision
            filtered_steps = [step for step in steps if any(kw in step.lower() for kw in query.split())]
            if filtered_steps:
                return "\n".join(filtered_steps[:max_results])
            else:
                # fallback to top steps anyway
                return "\n".join(steps[:max_results])

    # For troubleshooting-related queries, try semantic search on sentences that mention troubleshooting keywords
    if any(kw in query for kw in troubleshooting_keywords):
        filtered_sents = [s for s in sentences if any(kw in s.lower() for kw in troubleshooting_keywords)]
        if filtered_sents:
            # Semantic similarity ranking on filtered sentences
            query_embed = semantic_model.encode(query, convert_to_tensor=True)
            sent_embeds = semantic_model.encode(filtered_sents, convert_to_tensor=True)
            scores = util.cos_sim(query_embed, sent_embeds)[0]
            top_results = scores.topk(min(max_results, len(filtered_sents)))
            results = [filtered_sents[idx] for idx in top_results.indices]
            return "\n".join(results)

    # For explanation queries or generic queries: semantic search on entire manual text
    query_embed = semantic_model.encode(query, convert_to_tensor=True)
    sent_embeds = semantic_model.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(query_embed, sent_embeds)[0]
    top_results = scores.topk(max_results)
    results = [sentences[idx] for idx in top_results.indices]

    return "\n".join(results)

def extract_installation_steps(text):
    pattern = r"(Step\s+\d+\.)(.*?)(?=Step\s+\d+\.|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)
    steps = []
    for step_num, action in matches:
        cleaned_action = action.strip().replace('\n', ' ')
        full_step = f"{step_num.strip()} {cleaned_action}"
        if is_valid_step(full_step):
            steps.append(full_step)
    return steps

def extract_manual_sections(text):
    section_pattern = re.compile(r"\n?(\d+\.\s+[A-Z \-]+)\n")
    splits = section_pattern.split(text)
    sections = {}
    if splits:
        for i in range(1, len(splits), 2):
            title = splits[i].strip()
            content = splits[i + 1].strip() if i + 1 < len(splits) else ""
            sections[title] = content
    return sections

def generate_questions_from_summary(summary_text, use_large_model=False, num_questions=5):
    tokenizer = qg_tokenizer_large if use_large_model else qg_tokenizer
    model = qg_model_large if use_large_model else qg_model

    summary_sentences = sent_tokenize(summary_text)
    selected = [s for s in summary_sentences if len(s.split()) > 6][:num_questions]

    all_questions = set()
    for sentence in selected:
        input_text = f"generate question: {sentence.strip()}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

        outputs = model.generate(
            input_ids,
            max_length=64,
            num_beams=5,
            early_stopping=True
        )
        question = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        all_questions.add(question)

    return list(all_questions)



current_machine_id = None
machine_summary = ""
auto_generated_qa = []
custom_query_answers = []

def reset_machine_state():
    global machine_summary, auto_generated_qa, custom_query_answers
    machine_summary = ""
    auto_generated_qa = []
    custom_query_answers = []

@app.route("/", methods=["GET", "POST"])
def home():
    global current_machine_id
    if request.method == "POST":
        data = request.get_json()
        machine_id = data.get("machine_id")
        logger.info(f"Received machine_id: {machine_id} (current: {current_machine_id})")

        # Reset the state if a different machine is selected
        if str(machine_id) != str(current_machine_id):
            logger.info(f"Machine ID changed from {current_machine_id} to {machine_id}. Resetting state.")
            current_machine_id = machine_id
            reset_machine_state()

        # Now generate summary, QA etc. fresh for the new machine
        manual_link = get_manual_link(machine_id)
        if not manual_link:
            logger.warning(f"Manual link not found for machine_id: {machine_id}")
            return jsonify({"error": "Manual not found for the given machine ID."}), 404

        manual_text = extract_text_from_s3_pdf(manual_link)
        if not manual_text:
            return jsonify({"error": "Failed to extract manual text."}), 500

        cleaned_text = clean_repeated_lines(manual_text)
        use_large_model = is_large_manual(cleaned_text)
        keywords = extract_keywords(cleaned_text)
        summary = summarize_large_text(cleaned_text) if use_large_model else summarize_text(cleaned_text)
        summary_lines = summary.split('. ')
        questions = generate_contextual_questions(
            cleaned_text,
            question_model=qg_model_large if use_large_model else qg_model,
            tokenizer=qg_tokenizer_large if use_large_model else qg_tokenizer,
            keywords=keywords,
            top_n=5
        )

        # Append template-based QA questions for installation/configuration steps if any
        questions += generate_template_questions(cleaned_text)

        response = {
            "manual_text": cleaned_text,
            "summary": summary_lines,
            "questions": questions,
            "keywords": keywords,
            "manual_type": "large" if use_large_model else "small"
        }
        return jsonify(response)

    return render_template("index.html")

@app.route("/api/manual_qa", methods=["POST"])
def manual_qa():
    data = request.get_json()
    question = data.get("question")
    manual_text = data.get("manual_text")

    if not question or not manual_text:
        return jsonify({"error": "Missing question or manual_text"}), 400

    sentences = sent_tokenize(manual_text)

    # Get embeddings
    question_embedding = semantic_model.encode(question, convert_to_tensor=True)
    sentences_embeddings = semantic_model.encode(sentences, convert_to_tensor=True)

    # Perform semantic search to get top-k sentences
    hits = util.semantic_search(question_embedding, sentences_embeddings, top_k=5)[0]

    # Filter by a minimum similarity threshold (optional, e.g., 0.3)
    threshold = 0.3
    filtered_hits = [hit for hit in hits if hit['score'] >= threshold]

    # Extract top sentences based on filtered hits
    top_sentences = [sentences[hit['corpus_id']] for hit in filtered_hits]

    # Return as joined text or list
    answer_text = top_sentences if top_sentences else ["No relevant content found."]
    processed_answers = process_answers_to_bullets(answer_text)
    return jsonify({"answer": processed_answers})

@app.route("/query_manual", methods=["POST"])
def query_manual():
    data = request.get_json()
    machine_id = data.get("machine_id")
    user_query = data.get("query")

    if not machine_id or not user_query:
        return jsonify({"error": "machine_id and query are required"}), 400

    manual_link = get_manual_link(machine_id)
    if not manual_link:
        return jsonify({"error": "Manual link not found for machine"}), 404

    manual_text = extract_text_from_s3_pdf(manual_link)
    if not manual_text:
        return jsonify({"error": "Failed to extract manual text"}), 500

    cleaned_text = clean_repeated_lines(manual_text)
    relevant_content = search_manual_content(cleaned_text, user_query)

    processed_answers = process_answers_to_bullets(relevant_content)
    return jsonify({"answer": processed_answers})


@app.route('/reset-machine', methods=['POST'])
def reset_machine():
    reset_machine_state()
    return jsonify({"status": "success", "message": "Machine state reset."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
