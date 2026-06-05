import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from groq import Groq
import time
import json
import streamlit.components.v1 as components


client = Groq(
     api_key= st.secrets["GROQ_API_KEY"]
     )
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
    background: linear-gradient(135deg,#0f172a,#1e293b,#312e81);
}

.big-title{
    font-size:60px;
    font-weight:bold;
    color:white;
    text-align:center;
    margin-bottom:10px;
}

.big-subtitle{
    font-size:20px;
    font-weight:800;
    color:white;
    text-align:center;
    margin-bottom:50px;
}
.block-container{
    padding-top:10rem;
}

</style>
""", unsafe_allow_html=True)

#-----------------------------------------------------------------
#CHAT MEMORY
#-----------------------------------------------------------------
with st.sidebar:
    st.subheader ("Chat History")
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Ask Questions",
        "Viva",
        "Quiz",
        "Flashcards",
        "Summary"
    ]
)

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
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

        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                cleaned = page_text.replace("\n"," ")
                text += cleaned + " "
        #----------------------
        #CHUNK CREATER
        #----------------------
        chunks = split_text(text)

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
        st.session_state.index = index
        st.session_state.full_text = text
        st.session_state.pdf_processed = True
        
        st.success("PDF Processed Successfully")
#----------------------
#QUESTION ANSWER
#----------------------
if st.session_state.pdf_processed:

    def get_context(query, k=5):
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_embedding)

        D, I = st.session_state.index.search(query_embedding, k=k)

        contexts = [
            st.session_state.chunks[i]
            for i in I[0]
        ]

        return "\n\n".join(contexts)
        #st.write("Context created")
        #st.write(context[:300])
        #with st.expander("Retrieved Context"):
        #st.write(context)
    
    #----------------------
    #AI GENERATED ANSWER 
    #----------------------
    def generate_answer(prompt):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.choices[0].message.content
    with tab1:
            
        st.subheader("Ask Questions")
        query = st.text_input("ask a question from your uploaded academic material")
        if query:
            context = get_context(query)
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

                context = st.session_state.full_text[:12000]

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

                    st.session_state.viva_questions = questions_text.split("\n")
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

            prompt = f"""
            Create exactly 15 quiz questions from the academic context.
            Return ONLY valid JSON.
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

            Context:
            {context}
            """

            with st.spinner("Preparing quiz..."):
                quiz_text = generate_answer(prompt)

            try:
                st.session_state.quiz_questions = json.loads(quiz_text)
                st.session_state.current_quiz_question = 0
                st.session_state.quiz_score = 0
                st.session_state.hints_used = 0
                st.session_state.quiz_feedback = ""
                st.session_state.quiz_start_time = time.time()
                st.rerun()

            except Exception:
                st.error("Quiz format error. Please click Start Quiz again.")

        if len(st.session_state.quiz_questions) > 0:

            q_index = st.session_state.current_quiz_question
            question_data = st.session_state.quiz_questions[q_index]

            time_limit = question_data["time_limit"]

            if st.session_state.quiz_start_time is None:
                st.session_state.quiz_start_time = time.time()

            elapsed_time = int(time.time() - st.session_state.quiz_start_time)
            remaining_time = time_limit - elapsed_time


            st.markdown(f"### Question {q_index + 1} of 15")
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
                    if st.session_state.current_quiz_question < len(st.session_state.quiz_questions) - 1:
                        st.session_state.current_quiz_question += 1
                        st.session_state.quiz_start_time = time.time()
                        st.session_state.quiz_feedback = ""
                        st.rerun()
                    else:
                        st.session_state.quiz_completed = True
                        st.rerun()

            if st.session_state.quiz_completed:
                st.success(f"Quiz completed. Your score is {st.session_state.quiz_score}/15")
                st.subheader("Quiz Result Table")
                st.table(st.session_state.quiz_results)

            if st.session_state.quiz_feedback:
                st.write(st.session_state.quiz_feedback)

            if remaining_time <= 0:
                st.warning("Time is over. Submit or move to the next question.")
        
    with tab4:
        st.subheader("Flashcards")

        if st.button("Generate Flashcards"):
            context = st.session_state.full_text[:12000]
            prompt = f"""
            Generate exam-focused flashcards from this academic context

            Format:
            Q:
            A:

            Context:
            {context}
            """
            with st.spinner("Generating flashcards..."):
                answer = generate_answer(prompt)
            st.session_state.flashcard_count += 1

            st.session_state.chat_history.append(
                {
                "title": f"Flashcards {st.session_state.flashcard_count}",
                "content": answer
                }
            )
            st.write(answer)
    with tab5:
        st.subheader("Chapter Summary")
        if st.button("Generate Summary"):
            context = st.session_state.full_text[:20000]
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
