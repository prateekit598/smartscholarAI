import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from groq import Groq
from openai import OpenAI
import requests
import time
import json
import streamlit.components.v1 as components
import random
import re

Total_quiz_questions = 15
def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return None


groq_api_key = get_secret("GROQ_API_KEY")
gemini_api_key = get_secret("GEMINI_API_KEY")
openai_api_key = get_secret("OPENAI_API_KEY")
openrouter_api_key = get_secret("OPENROUTER_API_KEY")


groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
#-----------------------------------------------------------------
#PAGE CONFIG
#-----------------------------------------------------------------
st.set_page_config(
    page_title="SmartScholar AI",
    layout="wide"
)
#-----------------------------------------------------------------
#SESSION STATE
#-----------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "chunk_pages" not in st.session_state:
    st.session_state.chunk_pages = None

if "index" not in st.session_state:
    st.session_state.index = None

if "full_text" not in st.session_state:
    st.session_state.full_text = ""


#------------------------------
if "ask_count" not in st.session_state:
    st.session_state.ask_count = 0

if "viva_count" not in st.session_state:
    st.session_state.viva_count = 0

if "quiz_count" not in st.session_state:
    st.session_state.quiz_count = 0

if "flashcard_count" not in st.session_state:
    st.session_state.flashcard_count = 0

if "summary_count" not in st.session_state:
    st.session_state.summary_count = 0

#---------------------------------
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []

if "current_quiz_question" not in st.session_state:
    st.session_state.current_quiz_question = 0

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

if "quiz_start_time" not in st.session_state:
    st.session_state.quiz_start_time = None

if "hints_used" not in st.session_state:
    st.session_state.hints_used = 0

if "quiz_feedback" not in st.session_state:
    st.session_state.quiz_feedback = ""

if "quiz_answer_submitted" not in st.session_state:
    st.session_state.quiz_answer_submitted = False

if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = []

if "quiz_completed" not in st.session_state:
    st.session_state.quiz_completed = False

if "hint_shown_for_question" not in st.session_state:
    st.session_state.hint_shown_for_question = False

#-----------------------------------------------------------------
#CUSTOM CSS
#-----------------------------------------------------------------
st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background: 
    radial-gradient(circle at top left, rgba(56,189,248,0.18), transparent 35%),
    radial-gradient(circle at bottom right, rgba(129,140,248,0.22), transparent 35%),
    linear-gradient(135deg,#020617,#0f172a,#1e1b4b);
}

.block-container{
            padding-top:3rem;
            padding-bottom:3rem;
            max-width:1200px;
}
            
.big-title{
    font-size:64px;
    font-weight:900;
    text-align:center;
    margin-bottom:8px;
    background:linear-gradient(90deg,#38bdf8,#a78bfa,#f0abfc);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.big-subtitle{
    font-size:20px;
    font-weight:600;
    color:#cbd5e1;
    text-align:center;
    margin-bottom:35px;
}
            
[data-testid="stSidebar"]{
    background:rgba(15,23,42,0.92);
    border-right:1px solid rgba(255,255,255,0.12);
}

.glass-card{
    background:rgba(255,255,255,0.08);
    border:1px solid rgba(255,255,255,0.16);
    border-radius:24px;
    padding:24px;
    box-shadow:0 20px 60px rgba(0,0,0,0.25);
    backdrop-filter:blur(18px);
    margin-bottom:20px;
}
.feature-card{
    background:rgba(255,255,255,0.09);
    border:1px solid rgba(255,255,255,0.14);
    border-radius:20px;
    padding:22px;
    text-align:center;
    color:white;
    min-height:130px;
}     

.stTabs [data-baseweb="tab-list"]{
    gap:12px;
    justify-content:center;
}

.stTabs [data-baseweb="tab"]{
    background:rgba(255,255,255,0.08);
    border-radius:14px;
    color:#e5e7eb;
    padding:12px 22px;
    border:1px solid rgba(255,255,255,0.12);
}

.stTabs [aria-selected="true"]{
    background:linear-gradient(90deg,#2563eb,#7c3aed);
    color:white;
} 
            
.stButton button{
    background:linear-gradient(90deg,#2563eb,#7c3aed);
    color:white;
    border:none;
    border-radius:14px;
    padding:0.65rem 1.2rem;
    font-weight:700;
    box-shadow:0 10px 25px rgba(37,99,235,0.28);
}

.stButton button:hover{
    transform:translateY(-2px);
    box-shadow:0 14px 35px rgba(124,58,237,0.35);
}

input, textarea{
    border-radius:14px !important;
}
            
[data-testid="stFileUploader"]{
    background:rgba(255,255,255,0.08);
    border:1px dashed rgba(255,255,255,0.25);
    border-radius:18px;
    padding:16px;
}

[data-testid="stMetric"]{
    background:rgba(255,255,255,0.08);
    border:1px solid rgba(255,255,255,0.12);
    padding:16px;
    border-radius:18px;
}

h1,h2,h3,p,label,span{
    color:#f8fafc;
}
            
.upload-box{
    text-align:center;
    padding:30px;
    margin-bottom:25px;
    border-radius:25px;
    background:rgba(255,255,255,0.08);
    border:2px dashed rgba(255,255,255,0.25);
    backdrop-filter:blur(12px);
}

.upload-title{
    font-size:24px;
    font-weight:700;
    color:white;
    margin-bottom:10px;
}

.upload-subtitle{
    color:#cbd5e1;
    font-size:15px;
}
            
.upload-box h3{
    color:white;
    font-size:24px;
    font-weight:700;
    margin-bottom:10px;
}

.upload-box p{
    color:#cbd5e1;
    font-size:15px;
}
            
</style>
""", unsafe_allow_html=True)

#-----------------------------------------------------------------
#CHAT MEMORY
#-----------------------------------------------------------------
with st.sidebar:

    st.markdown("## 🎓 SmartScholar AI")
    st.caption("Academic Knowledge Assistant")

    st.divider()

    if st.button("🔄 Reset Workspace"):
        keys_to_keep = []
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        st.rerun()

    st.subheader("Chat History")
    if len(st.session_state.chat_history) == 0:
        st.info("No chats yet")

    for chat in reversed(st.session_state.chat_history):
        with st.sidebar.expander(chat["title"]):
            st.write(chat["content"])


#-----------------------------------------------------------------
#TITLE
#-----------------------------------------------------------------
st.markdown(
    """
    <div class="big-title">
        SmartScholar AI
    </div>

    <div class="big-subtitle">
        Domain Specific AI-Powered Knowledge Assistant
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("""
<div class="upload-box">
    <h3>📄 Upload Your Academic Material</h3>
    <p>Textbooks • Notes • Research Papers • Lab Manuals</p>
</div>
""", unsafe_allow_html=True)
with st.expander("📘 How to Use SmartScholar AI", expanded=False):
    st.markdown("""
    **1. Upload PDF**  
    Upload your textbook, notes, research paper, or study material.

    **2. Ask Questions**  
    Use the Ask Questions tab to ask doubts from the uploaded PDF.

    **3. Check Reference Pages**  
    After every answer, SmartScholar AI shows the PDF pages used for generating the answer.

    **4. Viva Preparation**  
    Use Viva mode to generate viva questions or practice answers.

    **5. Quiz Practice**  
    Generate 15 MCQ-based questions with timer, hints, score, and result table.

    **6. Flashcards and Summary**  
    Use Flashcards for quick revision and Summary for exam-ready notes.
    """)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    label_visibility="collapsed"
)

if not st.session_state.pdf_processed:
    
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 👋 Welcome to SmartScholar AI")
    st.write("Upload an academic PDF and use AI to ask questions, " \
    "prepare viva answers, generate quizzes, make flashcards,"
    " and create chapter summaries.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="feature-card">📚<br><br><b>Upload Books</b><br>Use your own study material</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="feature-card">🧠<br><br><b>Ask Questions</b><br>Get context-based answers</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="feature-card">🎯<br><br><b>Quiz & Viva</b><br>Practice for exams</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("ℹ️ About SmartScholar AI"):
        st.markdown("""
    **SmartScholar AI** is a Retrieval-Augmented Generation based academic assistant
    designed to help students learn directly from their own uploaded study material.

    **Main Features**
    - Context-based Question Answering
    - Viva Question Generation and Practice
    - Quiz Practice with timer, hints, and result table
    - Flashcard Generation
    - Chapter Summary Generation
    - Reference Page Tracking

    **Technologies Used**
    - Streamlit for web interface
    - PyPDF for text extraction
    - Sentence Transformers for embeddings
    - FAISS for semantic search
    - Groq Llama 3.3 for answer generation
    """)
    with st.expander("📊 Workspace Details"):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("📚 Chunks", 0 if st.session_state.chunks is None else len(st.session_state.chunks))

        with c2:
            st.metric("💬 Questions", st.session_state.ask_count)

        with c3:
            st.metric("🎤 Viva Sets", st.session_state.viva_count)

        with c4:
            st.metric("📝 Summaries", st.session_state.summary_count)

    with st.expander("⚙️ How SmartScholar AI Works"):
        st.markdown("""
        1. **PDF Upload** — The user uploads academic material.  
        2. **Text Extraction** — Text is extracted page-by-page using `PdfReader`.  
        3. **Chunking** — Text is divided into smaller chunks.  
        4. **Embeddings** — Chunks are converted into numerical vectors.  
        5. **FAISS Search** — Relevant chunks are retrieved using semantic search.  
        6. **LLM Generation** — Groq Llama 3.3 generates the answer.  
        7. **Reference Pages** — The system shows source page numbers.
        """)

    

#-----------------------------------------------------------------
#SIDEBAR
#-----------------------------------------------------------------

#mode = st.sidebar.selectbox(
    #"Choose Mode",
    #[
        #"Ask Questions",
        #"Viva Practice",
        #"Quiz Generator",
        #"Flashcards",
        #"Chapter Summary"
    #]
#)



#st.write("File uploaded:", uploaded_file is not None)
#st.write("PDF processed:", st.session_state.pdf_processed)

#-----------------------------------------------------------------
#CHUNK FUNCTION
#-----------------------------------------------------------------
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

#-----------------------------------------------------------------
#LOADING EMBEDDING MOEDL
#-----------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

#----------------------
#PDF READER
#----------------------              
if uploaded_file and not st.session_state.pdf_processed:

    with st.spinner("Reading PDF..."):
        reader = PdfReader(uploaded_file)
        text = ""
        page_wise_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                cleaned = page_text.replace("\n"," ")
                text += cleaned + " "
                page_wise_text.append(
                    {
                        "page": page_num,
                        "text": cleaned
                    }
                )
        #----------------------
        #CHUNK CREATER
        #----------------------
        chunks = []
        chunk_pages = []

        for item in page_wise_text:
            page_chunks = split_text(item["text"])
            for chunk in page_chunks:
                chunks.append(chunk)
                chunk_pages.append(item["page"])
        #----------------------
        #EMBEDDING CREATOR
        #----------------------
        embeddings = embedding_model.encode(chunks,show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)

        #----------------------
        #BUILD FAISS INDEX
        #----------------------
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)


        
        st.session_state.chunks = chunks
        st.session_state.chunk_pages = chunk_pages
        st.session_state.index = index
        st.session_state.full_text = text
        st.session_state.pdf_processed = True
        
        st.success("PDF Processed Successfully")
        st.rerun()
#----------------------
#QUESTION ANSWER
#----------------------
if st.session_state.pdf_processed:
    st.success("✅ PDF Processed Successfully")
    st.success("✅ PDF is ready. You can now use all study tools.")



    tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "💬 Ask Questions",
        "🎤 Viva",
        "🧠 Quiz",
        "📝 Flashcards",
        "📖 Summary"
    ]
)

    st.divider()
    def get_context(query, k=5):
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_embedding)

        D, I = st.session_state.index.search(query_embedding, k=k)

        contexts = []
        source_pages = set()

        for i in I[0]:
            contexts.append(
                st.session_state.chunks[i]
            )

            source_pages.add(
                st.session_state.chunk_pages[i]
            )

        return{
            "context": "\n\n".join(contexts),
            "pages": sorted(source_pages)
        }
        #st.write("Context created")
        #st.write(context[:300])
        #with st.expander("Retrieved Context"):
        #st.write(context)
    
    #----------------------
    #AI GENERATED ANSWER 
    #----------------------
    def call_groq(prompt, model):
        if groq_client is None:
            raise Exception("Groq API key missing")

        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1200
        )

        return response.choices[0].message.content


    def call_gemini(prompt, model):
        if gemini_api_key is None:
            raise Exception("Gemini API key missing")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": gemini_api_key
            },
            json={
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 1200
                }
            },
            timeout=60
        )

        response.raise_for_status()
        data = response.json()

        return data["candidates"][0]["content"]["parts"][0]["text"]


    def call_openai(prompt, model):
        if openai_client is None:
            raise Exception("OpenAI API key missing")

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
            temperature=0.3,
            max_tokens=1200
        )

        return response.choices[0].message.content


    def call_openrouter(prompt, model):
        if openrouter_api_key is None:
            raise Exception("OpenRouter API key missing")

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1200
            },
            timeout=60
        )

        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"]


    def generate_answer(prompt):
        fallback_models = [
            {
                "provider": "Groq",
                "model": "llama-3.3-70b-versatile",
                "function": call_groq
            },
            {
                "provider": "Groq",
                "model": "llama-3.1-8b-instant",
                "function": call_groq
            },
            {
                "provider": "Groq",
                "model": "gemma2-9b-it",
                "function": call_groq
            },
            {
                "provider": "Gemini",
                "model": "gemini-2.5-flash",
                "function": call_gemini
            },
            {
                "provider": "Gemini",
                "model": "gemini-2.0-flash",
                "function": call_gemini
            },
            {
                "provider": "OpenAI",
                "model": "gpt-4.1-mini",
                "function": call_openai
            },
            {
                "provider": "OpenAI",
                "model": "gpt-4o-mini",
                "function": call_openai
            },
            {
                "provider": "OpenRouter",
                "model": "meta-llama/llama-3.1-8b-instruct",
                "function": call_openrouter
            },
            {
                "provider": "OpenRouter",
                "model": "google/gemini-flash-1.5",
                "function": call_openrouter
            }
        ]

        errors = []

        for item in fallback_models:
            provider = item["provider"]
            model = item["model"]
            function = item["function"]

            try:
                answer = function(prompt, model)

                if answer and len(answer.strip()) > 0:
                    return answer

            except Exception as e:
                errors.append(f"{provider} - {model}: {str(e)}")
                continue

        st.error("All AI providers failed. Please try again after some time.")
    
        with st.expander("Technical error details"):
            for error in errors:
                st.write(error)

        st.stop()
    with tab1:
            
        st.subheader("Ask Questions")
        query = st.text_input("ask a question from your uploaded academic material")
        if query:
            retrieval = get_context(query)
            context = retrieval["context"]
            source_pages = retrieval["pages"]
            is_numerical = any(char.isdigit() for char in query)

            if is_numerical:
                prompt = f"""
                You are an expert tutor

                solve this problem step-by-step

                show:
                1. Given Data
                2. Formula Used
                3. Substitution
                4. Calculation
                5. Final Answer

                Context:
                {context}

                Question:
                {query}
                """
            else:
                prompt = f"""
                You are an academic assistant.

                Answer the question ONLY from the provided textbook context.

                Rules:
                - Give concise academic answers.
                - Do not copy the full context.
                - Do not repeat the question.
                - If answer is unavailable, say:
                'Answer not found in uploaded material.'

                Textbook Context:
                {context}

                Question:
                {query}

                Academic Answer:
                """
            with st.spinner("Generating AI response"):
                answer = generate_answer(prompt)
                
                st.session_state.ask_count += 1
                st.session_state.chat_history.append(
                    {
                        "title": f"Question{st.session_state.ask_count}",
                        "content": answer         
                    }
                )
                
                with st.chat_message("user"):
                    st.write(query)

                with st.chat_message("assistant"):
                    st.write(answer)
                st.caption(
                    f"📖 Reference Pages:{','.join(map(str, source_pages))}"
                )


    with tab2:
        st.subheader("Viva")
        viva_mode = st.radio(
            "Choose Viva Mode",
            [
            "Viva Question Generator",
            "Viva Practice"
            ]
        )

        if viva_mode == "Viva Question Generator":

            if st.button("Generate Viva Questions"):

                context = st.session_state.full_text[:8000]

                prompt = f"""
                You are a university viva examiner.

                Generate 10 viva questions from the academic context.

                Also provide short model answers.

                Format:

                Q1:
                Answer:

                Q2:
                Answer:

                Context:
                {context}
                """

                with st.spinner("Generating Viva Questions..."):
                    answer = generate_answer(prompt)

                st.session_state.viva_count += 1
                st.session_state.chat_history.append(
                    {
                        "title": f"Viva{st.session_state.viva_count}",
                        "content": answer
                    }
                )

                st.write(answer)

        elif viva_mode == "Viva Practice":
                if "viva_questions" not in st.session_state:
                    st.session_state.viva_questions = []

                if "current_viva_question" not in st.session_state:
                    st.session_state.current_viva_question = 0

                if st.button("Start Viva Practice"):
                    context = st.session_state.full_text[:12000]
                    prompt = f"""
                    Generate 5 viva questions from this academic context.
                    Only give questions.
                    Do not give answers.
                    Format:
                    Q1: ...
                    Q2: ...
                    Q3: ...
                    Q4: ...
                    Q5: ...
                    Context:
                    {context}
                    """

                    with st.spinner("Preparing viva questions..."):
                        questions_text = generate_answer(prompt)

                    st.session_state.viva_questions = [q.strip()
                                                       for q in questions_text.split("\n")
                                                       if q.strip()]
                    st.session_state.current_viva_question = 0

                if len(st.session_state.viva_questions) > 0:

                    question = st.session_state.viva_questions[
                        st.session_state.current_viva_question
                    ]

                    st.subheader(question)

                    student_answer = st.text_area(
                        "Type your answer here",
                        key="viva_answer"
                    )

                    if st.button("Check Viva Answer"):

                        context = st.session_state.full_text[:12000]

                        check_prompt = f"""
                        You are a university viva examiner.

                        Check the student's answer using only the academic context.

                        Give:
                        1. Correct or Incorrect
                        2. Marks out of 10
                        3. What was correct
                        4. What was missing
                        5. Ideal answer

                        Context:
                        {context}

                        Question:
                        {question}

                        Student Answer:
                        {student_answer}
                        """

                        with st.spinner("Checking your answer..."):
                            feedback = generate_answer(check_prompt)

                        st.write(feedback)

                    if st.button("Next Question"):

                        if st.session_state.current_viva_question < len(st.session_state.viva_questions) - 1:
                            st.session_state.current_viva_question += 1
                            st.rerun()
                        else:
                            st.success("Viva practice completed!")

    with tab3:
        st.subheader("Quiz")
        st.write("15 questions, timer, answer checking, and maximum 3 hints, are you ready?")
        
        if st.button("Start Quiz"):
            st.session_state.hints_used = 0
            st.session_state.hint_shown_for_question = False
            st.session_state.quiz_questions = []

            st.session_state.quiz_completed = False
            st.session_state.quiz_results = []
            st.session_state.quiz_answer_submitted = False
            context = st.session_state.full_text[:12000]

            random_seed = random.randint(1000, 9999)
            prompt = f"""
            Create exactly 15 quiz questions from the academic context.
            Return ONLY valid JSON.
            Random variation seed: {random_seed}

            JSON format:
            [
                {{
                    "question": "question text",
                    "options": ["A option", "B option", "C option", "D option"],
                    "correct_answer": "exact correct option text",
                    "explanation": "short explanation",
                    "hint": "helpful hint",
                    "time_limit": 30
                }}
            ]

            Rules:
            - Easy questions: 30 seconds
            - Medium questions: 60 seconds
            - Difficult questions: 100 seconds
            - Give mixed difficulty questions.
            - correct_answer must exactly match one option.
            - Do not write anything outside JSON.
            - Create a fresh new set of questions every time.
            - Do not repeat the same question pattern from previous quiz attempts.
            - Vary the order of topics, difficulty, and options.

            Context:
            {context}
            """

            with st.spinner("Preparing quiz..."):
                quiz_text = generate_answer(prompt)

            try:
                quiz_data = json.loads(quiz_text)

                if not isinstance(quiz_data, list):
                    st.error("Quiz format error. Please click Start Quiz again.")
                    st.stop()

                if len(quiz_data) < Total_quiz_questions:
                    st.error("AI generated fewer than 15 questions. Please click Start Quiz again.")
                    st.stop()

                st.session_state.quiz_questions = quiz_data[:Total_quiz_questions]

                st.session_state.current_quiz_question = 0
                st.session_state.quiz_score = 0
                st.session_state.hints_used = 0
                st.session_state.quiz_feedback = ""
                st.session_state.quiz_start_time = time.time()
                st.rerun()

            except Exception:
                st.error("Quiz format error. Please click Start Quiz again.")

        if len(st.session_state.quiz_questions )>0 and not st.session_state.quiz_completed:

            q_index = st.session_state.current_quiz_question
            question_data = st.session_state.quiz_questions[q_index]

            time_limit = question_data["time_limit"]

            if st.session_state.quiz_start_time is None:
                st.session_state.quiz_start_time = time.time()

            elapsed_time = int(time.time() - st.session_state.quiz_start_time)
            remaining_time = time_limit - elapsed_time


            st.markdown(f"### Question {q_index + 1} of {Total_quiz_questions}")
            if st.session_state.quiz_answer_submitted:
                st.success("Answer submitted")
            else:
                components.html(
                f"""
                <div style="font-size:26px; font-weight:bold; color:white;">
                Time left: <span id="timer">--</span> seconds
                </div>

                <div style="width:100%; background-color:#334155; border-radius:10px; margin-top:10px;">
                <div id="bar" style="height:18px; width:100%; background-color:#22c55e; border-radius:10px;"></div>
                </div>

                <script>
                    var timeLimit = {time_limit};
                    

                    function updateTimer() {{
                        var startTime = {int(st.session_state.quiz_start_time * 1000)};
                        var elapsed = Math.floor((Date.now() - startTime) / 1000);
                        var remaining = timeLimit - elapsed;

                        if (remaining < 0) {{
                            remaining = 0;
                        }}

                        document.getElementById("timer").innerText = remaining;

                        var percent = (remaining / timeLimit) * 100;
                        document.getElementById("bar").style.width = percent + "%";

                        if (remaining <= 10) {{
                            document.getElementById("bar").style.backgroundColor = "#ef4444";
                        }}
                        else if (remaining <= 20) {{
                            document.getElementById("bar").style.backgroundColor = "#f59e0b";
                        }}
                        else {{
                            document.getElementById("bar").style.backgroundColor = "#22c55e";
                        }}
                    }}

                    updateTimer();
                    setInterval(updateTimer, 1000);
                </script>
                """,
                height=90
            )


            st.write(question_data["question"])
            
            hints_left = 3 - st.session_state.hints_used
            st.info(f"💡 Hints Left: {hints_left}/3")
            selected_answer = st.radio(
                "Choose your answer:",
                question_data["options"],
                key=f"quiz_answer_{q_index}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Submit Answer"):
                    if not st.session_state.quiz_answer_submitted:
                        st.session_state.quiz_answer_submitted = True
                        if remaining_time <= 0:
                            result_status = "Incorrect"
                            st.session_state.quiz_feedback = f"""
                            Time Over

                            Correct Answer:
                            {question_data["correct_answer"]}

                            Explanation:    
                            {question_data["explanation"]}
                            """
                        elif selected_answer == question_data["correct_answer"]:
                            result_status = "Correct"
                            st.session_state.quiz_score += 1
                            st.session_state.quiz_feedback = f"""
                            Correct answer.

                            Explanation:
                            {question_data["explanation"]}
                            """
                        else:
                            result_status = "Incorrect"
                            st.session_state.quiz_feedback = f"""
                            Wrong answer.

                            Correct Answer:
                            {question_data["correct_answer"]}

                            Explanation:
                            {question_data["explanation"]}
                            """

                        st.session_state.quiz_results.append(
                            {
                                "S.No.": q_index + 1,
                                "Question": question_data["question"],
                                "My Answer": selected_answer,
                                "Result": result_status,
                                "Hints used": st.session_state.hints_used,
                                "Time Taken (sec)": elapsed_time,
                                "Correct Answer If Incorrect": "" if result_status == "Correct" else question_data["correct_answer"]
                            }
                        )

                        st.rerun()

                    else:
                        st.warning("You have already submitted this answer.")


            with col2:
                if st.button("Hint"):
                    if st.session_state.quiz_answer_submitted:
                        st.warning("You cannot use a hint after submitting the answer")
                    elif st.session_state.hint_shown_for_question:
                        st.warning("You have already used a hint for this question")
                    elif st.session_state.hints_used >=3:
                        st.warning("You have already used all 3 hints")
                    else:
                        st.session_state.hints_used +=1
                        st.session_state.hint_shown_for_question = True
                        st.info(question_data["hint"])

            with col3:
                if st.button("Next Question"):
                    st.session_state.quiz_answer_submitted = False
                    st.session_state.hint_shown_for_question = False
                    if st.session_state.current_quiz_question < Total_quiz_questions - 1:
                        st.session_state.current_quiz_question += 1
                        st.session_state.quiz_start_time = time.time()
                        st.session_state.quiz_feedback = ""
                        st.rerun()
                    else:
                        st.session_state.quiz_completed = True
                        st.session_state.quiz_feedback = ""
                        st.rerun()

        if st.session_state.quiz_completed:
            st.success(f"Quiz completed. Your score is {st.session_state.quiz_score}/{Total_quiz_questions}")
            st.subheader("Quiz Result Table")
            st.table(st.session_state.quiz_results[:Total_quiz_questions])    

        if st.session_state.quiz_feedback and not st.session_state.quiz_completed:
            st.write(st.session_state.quiz_feedback)

        if (
            len(st.session_state.quiz_questions)>0
            and not st.session_state.quiz_completed 
            and remaining_time <= 0):
            st.warning("Time is over. Submit or move to the next question.")   
        
        def format_flashcards_text(text):
            text = re.sub(r"\s+(A\d*:|A:|Answer:)", r"\n\1", text)
            text = re.sub(r"(Q\d*:|Q:|Question:)", r"\n\n\1", text)
            return text.strip()
    with tab4:
        st.subheader("Flashcards")

        if st.button("Generate Flashcards"):
            context = st.session_state.full_text[:12000]
            prompt = f"""
            Generate exactly 20 exam-focused flashcards from this academic context

            Return ONLY valid JSON.
            Do not write markdown.
            Do not write ```json.
            Do not write anything outside the JSON.

            JSON format:
            [
                {{
                    "question": "Write a clear question here",
                    "answer": "Write a short answer here"
                }}
            ]
            Rules:
            - Question and answer must be on separate lines.
            - Add one blank line after every answer.
            - Do not write question and answer in the same line.
            - Keep answers short and exam-focused.
            - Use only the academic context.

            Context:
            {context}
            """
            with st.spinner("Generating flashcards..."):
                flashcard_text = generate_answer(prompt)
            try:
                flashcard_text = flashcard_text.strip()
                if flashcard_text.startswith("```json"):
                    flashcard_text = flashcard_text.replace("```json", "").replace("```", "").strip()
                elif flashcard_text.startswith("```"):
                    flashcard_text = flashcard_text.replace("```", "").strip()

                flashcards = json.loads(flashcard_text)

                if not isinstance(flashcards, list):
                    st.error("Flashcard format error. Please generate again.")
                    st.stop()

                flashcards = flashcards[:10]

                valid_flashcards = []

                for card in flashcards:
                    question = card.get("question", "").strip()
                    answer = card.get("answer", "").strip()

                    if question and answer:
                        valid_flashcards.append(
                            {
                                "question": question,
                                "answer": answer
                            }
                        )

                if len(valid_flashcards) == 0:
                    st.error("Flashcard format error. Please generate again.")
                    st.stop()

                st.session_state.flashcard_count += 1

                history_text = ""

                for i, card in enumerate(valid_flashcards, start=1):
                    st.markdown(f"### Q{i}: {card['question']}")
                    st.markdown(f"**A{i}:** {card['answer']}")
                    st.divider()

                    history_text += f"Q{i}: {card['question']}\nA{i}: {card['answer']}\n\n"

                st.session_state.chat_history.append(
                    {
                        "title": f"Flashcards {st.session_state.flashcard_count}",
                        "content": history_text
                    }
                )

            except Exception:
                st.error("Flashcard format error. Please click Generate Flashcards again.")    
    with tab5:
        st.subheader("Chapter Summary")
        if st.button("Generate Summary"):
            context = st.session_state.full_text[:10000]
            prompt = f"""
            Create detailed academic notes from the chapter
            include:
            - Definations
            - Important concepts
            - Formulas
            - Key points
            - Exam tips

            Chapter:
            {context}
            summary:
            """

            with st.spinner("Generating summary..."):
                answer = generate_answer(prompt)
            st.session_state.summary_count += 1

            st.session_state.chat_history.append(
                {
                    "title": f"Summary {st.session_state.summary_count}",
                    "content": answer
                }
            )
            st.write(answer)
        st.divider()

    with st.expander("ℹ️ About SmartScholar AI"):
        st.markdown("""
    **SmartScholar AI** is a Retrieval-Augmented Generation based academic assistant
    designed to help students learn directly from their own uploaded study material.

    **Main Features**
    - Context-based Question Answering
    - Viva Question Generation and Practice
    - Quiz Practice with timer, hints, and result table
    - Flashcard Generation
    - Chapter Summary Generation
    - Reference Page Tracking

    **Technologies Used**
    - Streamlit for web interface
    - PyPDF for text extraction
    - Sentence Transformers for embeddings
    - FAISS for semantic search
    - Groq Llama 3.3 for answer generation
    """)

    with st.expander("📊 Workspace Details"):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("📚 Chunks",0 if st.session_state.chunks is None else len(st.session_state.chunks))

        with c2:
            st.metric("💬 Questions", st.session_state.ask_count)

        with c3:
            st.metric("🎤 Viva Sets", st.session_state.viva_count)

        with c4:
            st.metric("📝 Summaries", st.session_state.summary_count)

    with st.expander("⚙️ How SmartScholar AI Works"):
        st.markdown("""
        1. **PDF Upload** — The user uploads academic material.  
        2. **Text Extraction** — Text is extracted page-by-page using `PdfReader`.  
        3. **Chunking** — Text is divided into smaller chunks.  
        4. **Embeddings** — Chunks are converted into numerical vectors.  
        5. **FAISS Search** — Relevant chunks are retrieved using semantic search.  
        6. **LLM Generation** — Groq Llama 3.3 generates the answer.  
        7. **Reference Pages** — The system shows source page numbers.
        """)

#----------------------
#DEBUG
#----------------------

    #st.write("Total characters:", len(text))
    #st.write("Total chunks:", len(chunks))

#----------------------
#SAMPLE RUN
#----------------------
    #with st.expander("Sample chunk"):
      #  st.write(chunks[0])
