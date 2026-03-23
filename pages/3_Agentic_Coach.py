import streamlit as st
import os
import sys

# Add parent directory so we can import ui_style
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ui_style

# -----------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------
st.set_page_config(
    page_title="AI Study Coach — ILAAS",
    page_icon="🤖",
    layout="wide"
)
ui_style.apply_german_ui()

# Page header
st.markdown("<h1>AI Study Coach</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='lead'>Powered by Groq & LangGraph. Enter a student's profile and goal, "
    "and the AI agent will diagnose learning gaps, build a personalized study plan, "
    "recommend resources, and generate a practice quiz — all automatically.</p>",
    unsafe_allow_html=True
)

# -----------------------------------------------------------------
# Try to import the LangGraph/Groq libraries.
# If they are not installed, show a friendly error message.
# -----------------------------------------------------------------
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage
    from langgraph.graph import StateGraph, END
    from typing import TypedDict
    libs_ok = True
except ImportError:
    libs_ok = False

if not libs_ok:
    st.error(
        "⚠️ Required libraries are not installed. "
        "Please run: `pip install langchain langchain-groq langgraph`"
    )
    st.stop()

# -----------------------------------------------------------------
# SECTION 1: API Key Input
# The user must provide their Groq API Key to use this page.
# -----------------------------------------------------------------
st.markdown("<h3>🔑 Step 1: Enter Your Groq API Key</h3>", unsafe_allow_html=True)
st.markdown(
    "<p>Get a free API key from "
    "<a href='https://console.groq.com' target='_blank' style='color:#a78bfa;'>console.groq.com</a>. "
    "Your key is never stored — it only lives in this session.</p>",
    unsafe_allow_html=True
)

# Password input so the key is hidden on screen
groq_api_key = st.text_input(
    "Groq API Key",
    type="password",
    placeholder="gsk_...",
    value=st.session_state.get('groq_api_key', '')
)

if groq_api_key:
    # Save key in session so it sticks across reruns
    st.session_state['groq_api_key'] = groq_api_key
    st.success("✅ API Key received.")

st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------------------------------------------
# SECTION 2: Student Profile Input Form
# -----------------------------------------------------------------
st.markdown("<h3>📋 Step 2: Student Profile</h3>", unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    # Student's name
    student_name = st.text_input(
        "Student Name",
        value=st.session_state.get('coach_name', ''),
        placeholder="e.g. Arjun Sharma"
    )

    # What the student wants to achieve
    student_goals = st.text_area(
        "Learning Goal",
        value=st.session_state.get('coach_goals', ''),
        placeholder="e.g. I want to score at least 75% in my final Math exam next month.",
        height=80
    )

    # Subject they are struggling with
    subject = st.selectbox(
        "Main Subject",
        ["Mathematics", "Physics", "Chemistry", "Biology", "Computer Science",
         "History", "Economics", "English", "Other"],
        index=["Mathematics", "Physics", "Chemistry", "Biology", "Computer Science",
               "History", "Economics", "English", "Other"].index(
            st.session_state.get('coach_subject', 'Mathematics'))
    )

with col_right:
    # Recent quiz scores (we'll ask for 3)
    st.markdown("<p style='font-weight:600; color:#e8e8f0;'>Recent Quiz Scores (out of 100)</p>",
                unsafe_allow_html=True)

    score1 = st.slider("Quiz 1 Score", 0, 100, st.session_state.get('score1', 50))
    score2 = st.slider("Quiz 2 Score", 0, 100, st.session_state.get('score2', 55))
    score3 = st.slider("Quiz 3 Score", 0, 100, st.session_state.get('score3', 60))

    # Study hours per week
    study_hours = st.slider("Study Hours per Week", 0, 40, st.session_state.get('study_hrs', 5))

# Weak topics text input
weak_topics_raw = st.text_input(
    "Weak Topics (comma-separated)",
    value=st.session_state.get('weak_topics_raw', ''),
    placeholder="e.g. Algebra, Integration, Probability"
)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------
# SECTION 3: If we already have an ML prediction from page 1,
# offer to auto-fill the risk level. This connects both milestones.
# -----------------------------------------------------------------
if 'last_prediction' in st.session_state:
    st.info(
        f"💡 The Evaluation Matrix already classified this student as "
        f"**{st.session_state['last_prediction']}**. "
        f"The AI Coach will factor this into the diagnosis automatically."
    )
    ml_risk = st.session_state['last_prediction']
else:
    # If there's no prior prediction, let the user pick manually
    ml_risk_opt = st.selectbox(
        "Estimated Risk Level (from ML model or manual assessment)",
        ["At Risk", "Average", "High Performer"],
        help="Run the Evaluation Matrix first to auto-fill this."
    )
    ml_risk = ml_risk_opt

st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------------------------------------------
# SECTION 4: Run the LangGraph Agent
# -----------------------------------------------------------------
st.markdown("<h3>🚀 Step 3: Generate Study Plan</h3>", unsafe_allow_html=True)

run_button = st.button("✨ Generate My Personalized Study Plan", type="primary")

if run_button:
    # ----- Basic validation -----
    if not groq_api_key:
        st.error("❌ Please enter your Groq API Key above.")
        st.stop()
    if not student_name.strip():
        st.error("❌ Please enter the student's name.")
        st.stop()
    if not student_goals.strip():
        st.error("❌ Please enter the student's learning goal.")
        st.stop()

    # Save to session state
    st.session_state['coach_name'] = student_name
    st.session_state['coach_goals'] = student_goals
    st.session_state['coach_subject'] = subject
    st.session_state['score1'] = score1
    st.session_state['score2'] = score2
    st.session_state['score3'] = score3
    st.session_state['study_hrs'] = study_hours
    st.session_state['weak_topics_raw'] = weak_topics_raw

    # Parse weak topics into a list
    weak_topics = [t.strip() for t in weak_topics_raw.split(',') if t.strip()]
    if not weak_topics:
        weak_topics = ["General concepts"]

    # Build the performance dictionary to pass to the agent
    performance_data = {
        "subject": subject,
        "recent_scores": [score1, score2, score3],
        "average_score": round((score1 + score2 + score3) / 3, 1),
        "weak_topics": weak_topics,
        "study_hours_per_week": study_hours,
        "ml_risk_classification": ml_risk
    }

    # ----------------------------------------------------------------
    # Define the LangGraph Agent State
    # This is what the agent carries from step to step
    # ----------------------------------------------------------------
    class AgentState(TypedDict):
        student_name: str
        student_goals: str
        performance_data: dict
        learning_diagnosis: str
        study_plan: str
        resources: str
        practice_quiz: str

    # ----------------------------------------------------------------
    # Initialize the Groq LLM using the user's API key
    # ----------------------------------------------------------------
    try:
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.6,
            max_tokens=1500,
            api_key=groq_api_key
        )
    except Exception as e:
        st.error(f"❌ Failed to connect to Groq API: {e}")
        st.stop()

    # ----------------------------------------------------------------
    # Define Agent Node 1: Diagnose the student
    # This node reads the performance data and finds learning gaps
    # ----------------------------------------------------------------
    def diagnose_student(state: AgentState):
        prompt = f"""You are a learning analytics expert and academic counselor.

Student Name: {state['student_name']}
Subject: {state['performance_data']['subject']}
Recent Quiz Scores: {state['performance_data']['recent_scores']}
Average Score: {state['performance_data']['average_score']}%
Known Weak Topics: {state['performance_data']['weak_topics']}
Weekly Study Hours: {state['performance_data']['study_hours_per_week']}
ML Risk Classification: {state['performance_data']['ml_risk_classification']}
Student Goal: {state['student_goals']}

Write a clear, structured learning diagnosis that:
1. Identifies the student's STRENGTHS (2-3 sentences)
2. Identifies the student's WEAKNESSES and specific learning gaps (3-4 sentences)
3. Gives a SHORT verdict sentence on their current academic risk

Keep it honest, encouraging, and easy to understand."""

        response = llm.invoke([HumanMessage(content=prompt)])
        return {"learning_diagnosis": response.content}

    # ----------------------------------------------------------------
    # Define Agent Node 2: Create a 4-week study plan
    # This node reads the diagnosis and builds a weekly plan
    # ----------------------------------------------------------------
    def make_plan(state: AgentState):
        prompt = f"""You are a professional academic coach creating a study plan.

Student: {state['student_name']}
Diagnosis Summary:
{state['learning_diagnosis']}

Goal: {state['student_goals']}

Create a detailed 4-week study plan:
- Week 1: Foundation review (focus on the biggest gaps, easiest fixes)
- Week 2: Core skill building (go deeper on weak topics)
- Week 3: Practice and application (problem sets, past papers)
- Week 4: Final review and mock exams

For each week:
• List 3-4 specific daily tasks or topics
• Include one review session at the end of the week
• Keep tasks realistic for someone studying {state['performance_data']['study_hours_per_week']} hours/week

Format with clear Week headings and bullet points."""

        response = llm.invoke([
            SystemMessage(content="You are an expert study planner. Be specific, motivating, and realistic."),
            HumanMessage(content=prompt)
        ])
        return {"study_plan": response.content}

    # ----------------------------------------------------------------
    # Define Agent Node 3: Recommend learning resources
    # This node recommends free online resources based on the plan
    # ----------------------------------------------------------------
    def fetch_resources(state: AgentState):
        prompt = f"""You are a helpful academic librarian.

Based on this study plan for the subject "{state['performance_data']['subject']}":
{state['study_plan']}

Recommend exactly 5 FREE, high-quality online resources.

For each resource, provide:
1. Resource Name
2. URL (must be a real, well-known URL like Khan Academy, YouTube, Coursera free tier, etc.)
3. What it covers (1 sentence)
4. Why it is perfect for this student's weak areas

Focus specifically on: {state['performance_data']['weak_topics']}"""

        response = llm.invoke([HumanMessage(content=prompt)])
        return {"resources": response.content}

    # ----------------------------------------------------------------
    # Define Agent Node 4: Generate a practice quiz (Extension Feature)
    # This node writes 5 multiple-choice questions based on the diagnosis
    # ----------------------------------------------------------------
    def make_quiz(state: AgentState):
        prompt = f"""You are an experienced teacher writing a quick diagnostic quiz.

Subject: {state['performance_data']['subject']}
Student's weak areas: {state['performance_data']['weak_topics']}

Write exactly 5 multiple-choice questions to test the student's understanding of their weak areas.

For EACH question:
Q[N]: [Clear question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
✅ Correct Answer: [Letter] — [Brief one-sentence explanation]

Make questions progressively harder from Q1 (easy) to Q5 (challenging)."""

        response = llm.invoke([HumanMessage(content=prompt)])
        return {"practice_quiz": response.content}

    # ----------------------------------------------------------------
    # Build the LangGraph workflow
    # Connect all the nodes in order like a flowchart
    # ----------------------------------------------------------------
    workflow = StateGraph(AgentState)

    workflow.add_node("Diagnose", diagnose_student)
    workflow.add_node("Plan", make_plan)
    workflow.add_node("Resources", fetch_resources)
    workflow.add_node("Quiz", make_quiz)

    workflow.set_entry_point("Diagnose")
    workflow.add_edge("Diagnose", "Plan")
    workflow.add_edge("Plan", "Resources")
    workflow.add_edge("Resources", "Quiz")
    workflow.add_edge("Quiz", END)

    study_coach = workflow.compile()

    # ----------------------------------------------------------------
    # Run the agent — this calls each node one by one
    # We use placeholders to show the output as each step completes
    # ----------------------------------------------------------------
    initial_state = {
        "student_name": student_name,
        "student_goals": student_goals,
        "performance_data": performance_data,
        "learning_diagnosis": "",
        "study_plan": "",
        "resources": "",
        "practice_quiz": ""
    }

    # Display a loading message while each step runs
    step_icons = {
        "Diagnose": ("🔍", "Diagnosing learning gaps..."),
        "Plan":     ("📅", "Building your 4-week study plan..."),
        "Resources":("📚", "Finding the best free resources..."),
        "Quiz":     ("❓", "Writing your practice quiz..."),
    }

    result = {}
    with st.status("AI Study Coach is working...", expanded=True) as status:
        try:
            # Stream the graph execution step by step
            for step_output in study_coach.stream(initial_state, stream_mode="updates"):
                for node_name, node_result in step_output.items():
                    icon, msg = step_icons.get(node_name, ("⚙️", f"Running {node_name}..."))
                    st.write(f"{icon} {msg}")
                    # Accumulate results
                    result.update(node_result)

            status.update(label="✅ Study Plan Ready!", state="complete")

        except Exception as e:
            status.update(label="❌ Agent encountered an error.", state="error")
            st.error(f"Error: {e}")
            st.stop()

    # Save result to session state so it persists when the page rerenders
    st.session_state['agent_result'] = result


# -----------------------------------------------------------------
# SECTION 5: Display the Agent's Output
# -----------------------------------------------------------------
if 'agent_result' in st.session_state and st.session_state['agent_result']:
    result = st.session_state['agent_result']

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f"<h3>📄 Study Coach Report for {st.session_state.get('coach_name', 'Student')}</h3>",
        unsafe_allow_html=True
    )

    # ---- Diagnosis ----
    if result.get("learning_diagnosis"):
        st.markdown("""
        <div class="agent-section">
            <div class="agent-section-title">🔍 Learning Diagnosis</div>
        """, unsafe_allow_html=True)
        st.markdown(
            f"<div class='agent-section-body'>{result['learning_diagnosis']}</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Study Plan ----
    if result.get("study_plan"):
        st.markdown("""
        <div class="agent-section">
            <div class="agent-section-title">📅 Personalized 4-Week Study Plan</div>
        """, unsafe_allow_html=True)
        st.markdown(
            f"<div class='agent-section-body'>{result['study_plan']}</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Resources ----
    if result.get("resources"):
        st.markdown("""
        <div class="agent-section">
            <div class="agent-section-title">📚 Recommended Free Resources</div>
        """, unsafe_allow_html=True)
        st.markdown(
            f"<div class='agent-section-body'>{result['resources']}</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Practice Quiz (Extension Feature) ----
    if result.get("practice_quiz"):
        st.markdown("""
        <div class="agent-section">
            <div class="agent-section-title">❓ Practice Quiz — Test Your Knowledge</div>
        """, unsafe_allow_html=True)
        st.markdown(
            f"<div class='agent-section-body'>{result['practice_quiz']}</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Reset Button ----
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Generate New Plan", type="primary"):
        # Clear the result and let the user start over
        del st.session_state['agent_result']
        st.rerun()
