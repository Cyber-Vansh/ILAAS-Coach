import os
import sys
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_model_path(relative_path: str):
    return os.path.join(os.path.dirname(__file__), relative_path)


class PredictRequest(BaseModel):
    school: str
    sex: str
    age: int
    address: str
    studytime: int
    failures: int
    schoolsup: str
    famsup: str
    paid: str
    higher: str
    internet: str
    traveltime: int
    absences: int
    G1: int
    G2: int


class AgentRequest(BaseModel):
    student_name: str
    student_goals: str
    subject: str
    recent_scores: list[int]
    weak_topics: list[str]
    study_hours: int
    ml_risk: str
    predicted_g3: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        model = joblib.load(get_model_path("model/saved_model.pkl"))
        expected_columns = model.feature_names_in_

        g3_model = joblib.load(get_model_path("model/g3_regressor.pkl"))
        g3_features = joblib.load(get_model_path("model/g3_features.pkl"))
    except Exception as e:
        return {"error": f"Model load failed: {e}"}

    is_high_risk = req.failures > 0 or req.absences > 15 or (req.G1 + req.G2) < 20

    input_data = {
        "school": req.school,
        "sex": req.sex,
        "age": req.age,
        "address": req.address,
        "famsize": "GT3", "Pstatus": "T", "Medu": 2, "Fedu": 2,
        "Mjob": "other", "Fjob": "other", "reason": "other", "guardian": "mother",
        "traveltime": req.traveltime,
        "studytime": req.studytime,
        "failures": req.failures,
        "schoolsup": req.schoolsup,
        "famsup": req.famsup,
        "paid": req.paid,
        "activities": "no" if is_high_risk else "yes",
        "nursery": "yes",
        "higher": req.higher,
        "internet": req.internet,
        "romantic": "yes" if is_high_risk else "no",
        "famrel": 2 if is_high_risk else 4,
        "freetime": 5 if is_high_risk else 3,
        "goout": 5 if is_high_risk else 3,
        "health": 4,
        "Dalc": 3 if is_high_risk else 1,
        "Walc": 4 if is_high_risk else 1,
        "absences": req.absences,
        "G1": req.G1,
        "G2": req.G2,
    }

    df_raw = pd.DataFrame([input_data])
    df_encoded = pd.get_dummies(df_raw)

    final_features = pd.DataFrame(0, index=[0], columns=expected_columns)
    for col in expected_columns:
        if col in df_encoded.columns:
            final_features[col] = df_encoded[col]

    numeric_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2"]
    for col in final_features.columns:
        if col in numeric_cols:
            final_features[col] = pd.to_numeric(final_features[col], downcast="integer")
        else:
            final_features[col] = final_features[col].astype("bool")

    classes = model.classes_
    probs = model.predict_proba(final_features)[0]
    prob_map = dict(zip(classes, probs))

    if prob_map.get("At Risk", 0) >= 0.15:
        prediction = "At Risk"
    elif prob_map.get("High Performer", 0) > prob_map.get("Average", 0):
        prediction = "High Performer"
    else:
        prediction = "Average"

    g3_final_features = pd.DataFrame(0, index=[0], columns=g3_features)
    for col in g3_features:
        if col in df_encoded.columns:
            g3_final_features[col] = df_encoded[col]
            
    predicted_g3 = g3_model.predict(g3_final_features)[0]

    return {"prediction": prediction, "probabilities": prob_map, "predicted_g3": round(predicted_g3, 2)}


class AgentState(TypedDict):
    student_name: str
    student_goals: str
    performance_data: dict
    learning_diagnosis: str
    study_plan: str
    resources: str
    practice_quiz: str


@app.post("/agent")
def run_agent(req: AgentRequest):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return {"error": "backend configuration missing. GROQ_API_KEY not found in .env"}

    try:
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.6,
            max_tokens=1500,
            api_key=groq_api_key,
        )
    except Exception as e:
        return {"error": f"LLM init failed: {e}"}

    performance_data = {
        "subject": req.subject,
        "recent_scores": req.recent_scores,
        "average_score": round(sum(req.recent_scores) / len(req.recent_scores), 1),
        "weak_topics": req.weak_topics,
        "study_hours_per_week": req.study_hours,
        "ml_risk_classification": req.ml_risk,
        "predicted_g3": req.predicted_g3
    }

    def diagnose(state: AgentState):
        p = f"""You are a learning analytics expert.
Student: {state['student_name']}
Subject: {state['performance_data']['subject']}
Scores: {state['performance_data']['recent_scores']} (avg {state['performance_data']['average_score']}%)
Predicted Final Grade (G3): {state['performance_data']['predicted_g3']}/20
Weak topics: {state['performance_data']['weak_topics']}
Study hours/week: {state['performance_data']['study_hours_per_week']}
Risk level: {state['performance_data']['ml_risk_classification']}
Goal: {state['student_goals']}

Write a concise, plain-language learning diagnosis. Cover: strengths, specific gaps, interpretation of their predicted final grade vs goal, and risk verdict."""
        r = llm.invoke([HumanMessage(content=p)])
        return {"learning_diagnosis": r.content}

    def plan(state: AgentState):
        p = f"""You are an academic study coach.
Diagnosis: {state['learning_diagnosis']}
Goal: {state['student_goals']}
Study hours/week: {state['performance_data']['study_hours_per_week']}

Create a 4-week study plan. For each week: a theme label, and 3-4 specific daily tasks.
Format each week as: Week N — Theme, then bullet tasks."""
        r = llm.invoke([SystemMessage(content="You are an expert study planner. Be specific and concise."),
                        HumanMessage(content=p)])
        return {"study_plan": r.content}

    def resources(state: AgentState):
        p = f"""Recommend exactly 5 free learning resources for a student studying {state['performance_data']['subject']}.
Weak areas: {state['performance_data']['weak_topics']}
For each: resource name, URL (real and working), one sentence on what it covers."""
        r = llm.invoke([HumanMessage(content=p)])
        return {"resources": r.content}

    def quiz(state: AgentState):
        p = f"""Write a 5-question multiple choice quiz on:
Subject: {state['performance_data']['subject']}
Topics: {state['performance_data']['weak_topics']}
Format each as:
Q[N]: question
A) option  B) option  C) option  D) option
Answer: X — one-sentence explanation"""
        r = llm.invoke([HumanMessage(content=p)])
        return {"practice_quiz": r.content}

    workflow = StateGraph(AgentState)
    workflow.add_node("Diagnose", diagnose)
    workflow.add_node("Plan", plan)
    workflow.add_node("Resources", resources)
    workflow.add_node("Quiz", quiz)
    workflow.set_entry_point("Diagnose")
    workflow.add_edge("Diagnose", "Plan")
    workflow.add_edge("Plan", "Resources")
    workflow.add_edge("Resources", "Quiz")
    workflow.add_edge("Quiz", END)
    agent = workflow.compile()

    initial = {
        "student_name": req.student_name,
        "student_goals": req.student_goals,
        "performance_data": performance_data,
        "learning_diagnosis": "",
        "study_plan": "",
        "resources": "",
        "practice_quiz": "",
    }

    try:
        result = agent.invoke(initial)
        return {
            "diagnosis": result["learning_diagnosis"],
            "study_plan": result["study_plan"],
            "resources": result["resources"],
            "quiz": result["practice_quiz"],
        }
    except Exception as e:
        return {"error": str(e)}
