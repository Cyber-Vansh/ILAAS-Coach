"use client";
import { useState, useEffect } from "react";
import styles from "./page.module.css";

type Assessment = {
  name: string;
  subject: string;
  goal: string;
  weakTopics: string[];
  studyHours: number;
  mlRisk: string;
  predictedG3: number;
  scores: number[];
};

type AgentResult = {
  diagnosis: string;
  study_plan: string;
  resources: string;
  quiz: string;
  error?: string;
};

type QuizQuestion = {
  question: string;
  options: string[];
  answer: string;
  explanation: string;
  revealed: boolean;
};

function parseQuiz(raw: string): QuizQuestion[] {
  const questions: QuizQuestion[] = [];
  const blocks = raw.split(/Q\d+:/g).filter(Boolean);
  for (const block of blocks) {
    const lines = block.trim().split("\n").filter(Boolean);
    if (lines.length < 5) continue;
    const question = lines[0].trim();
    const options = lines.filter((l) => /^[A-D]\)/.test(l.trim())).map((l) => l.trim());
    const answerLine = lines.find((l) => /^Answer:/i.test(l.trim())) || "";
    const answerMatch = answerLine.match(/Answer:\s*([A-D])\s*[—–-]?\s*(.*)/i);
    questions.push({
      question,
      options,
      answer: answerMatch ? answerMatch[1] : "A",
      explanation: answerMatch ? answerMatch[2] : "",
      revealed: false,
    });
  }
  return questions;
}

function parsePlan(raw: string) {
  const weeks: { title: string; tasks: string[] }[] = [];
  const lines = raw.split("\n");
  let current: { title: string; tasks: string[] } | null = null;
  for (const line of lines) {
    const weekMatch = line.match(/^(Week\s+\d+[^:—–]*)[—–:]/i);
    if (weekMatch) {
      if (current) weeks.push(current);
      current = { title: weekMatch[0].replace(/[—–:]$/, "").trim(), tasks: [] };
    } else if (current && /^[\-•*]/.test(line.trim())) {
      current.tasks.push(line.replace(/^[\-•*\s]+/, "").trim());
    }
  }
  if (current) weeks.push(current);
  return weeks;
}

const RISK_MAP: Record<string, { label: string; color: string }> = {
  "At Risk":       { label: "At Risk",       color: "var(--danger)" },
  "Average":       { label: "Average",        color: "var(--warning)" },
  "High Performer":{ label: "High Performer", color: "var(--success)" },
};

export default function CoachPage() {
  const [assessment, setAssessment] = useState<Assessment | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<AgentResult | null>(null);
  const [quizItems, setQuizItems] = useState<QuizQuestion[]>([]);
  const [activeTab, setActiveTab] = useState<"diagnosis" | "plan" | "resources" | "quiz">("diagnosis");
  const [checkedTasks, setCheckedTasks] = useState<Set<string>>(new Set());

  useEffect(() => {
    const stored = localStorage.getItem("ilaas_assessment");
    if (stored) setAssessment(JSON.parse(stored));
    const storedResult = localStorage.getItem("ilaas_result");
    if (storedResult) {
      const r = JSON.parse(storedResult);
      setResult(r);
      setQuizItems(parseQuiz(r.quiz || ""));
    }
  }, []);

  const run = async () => {
    if (!assessment) { setError("Complete the assessment first."); return; }
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch("http://localhost:8000/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          student_name: assessment.name,
          student_goals: assessment.goal,
          subject: assessment.subject,
          recent_scores: assessment.scores,
          weak_topics: assessment.weakTopics,
          study_hours: assessment.studyHours,
          ml_risk: assessment.mlRisk,
          predicted_g3: assessment.predictedG3,
        }),
      });
      const data: AgentResult = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
      setQuizItems(parseQuiz(data.quiz || ""));
      localStorage.setItem("ilaas_result", JSON.stringify(data));
      setActiveTab("diagnosis");
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Request failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const toggleTask = (key: string) => {
    setCheckedTasks((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  };

  const revealAnswer = (i: number) => {
    setQuizItems((prev) => prev.map((q, idx) => idx === i ? { ...q, revealed: true } : q));
  };

  const riskInfo = assessment ? RISK_MAP[assessment.mlRisk] ?? { label: assessment.mlRisk, color: "var(--text-secondary)" } : null;

  return (
    <div className={styles.page}>
      <div className={styles.sidebar}>
        <div className={styles.sideSection}>
          <h4>Student Profile</h4>
          {assessment ? (
            <>
              <p className={styles.studentName}>{assessment.name || "—"}</p>
              <p className={styles.studentSub}>{assessment.subject}</p>
              
              <div className={styles.metricsBox}>
                <div className={styles.metricItem}>
                  <span className={styles.metricLabel}>Risk Level</span>
                  {riskInfo && (
                    <div className={styles.riskBadge} style={{ color: riskInfo.color, borderColor: riskInfo.color }}>
                      {riskInfo.label}
                    </div>
                  )}
                </div>
                <div className={styles.metricItem}>
                  <span className={styles.metricLabel}>Predicted Final Grade</span>
                  <div className={styles.g3Score}>
                    {assessment.predictedG3.toFixed(1)} <span className={styles.g3Max}>/ 20</span>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <p className={styles.noData}>No assessment found. <a href="/assess" className={styles.link}>Take assessment</a></p>
          )}
        </div>

        {assessment && (
          <div className={styles.sideSection}>
            <h4>Stated Goal</h4>
            <p className={styles.goalText}>"{assessment.goal}"</p>
          </div>
        )}

        {assessment && assessment.weakTopics.length > 0 && (
          <div className={styles.sideSection}>
            <h4>Weak Topics identified</h4>
            <div className={styles.topicList}>
              {assessment.weakTopics.map((t) => (
                <span key={t} className="tag">{t}</span>
              ))}
            </div>
          </div>
        )}

        <button className={`btn-primary ${styles.runBtn}`} onClick={run} disabled={loading || !assessment}>
          {loading ? "Agent working..." : result ? "Recalculate Plan" : "Calculate My Plan"}
        </button>

        {error && <p className={styles.error}>{error}</p>}
      </div>

      <div className={styles.main}>
        {loading && (
          <div className={styles.loadingState}>
            <div className={styles.spinner} />
            <p>The AI Agent is processing your learning data globally. (Takes about ~15 seconds)</p>
            <div className={styles.loadingSteps}>
              <span>Diagnosing</span>
              <span className={styles.arrow}>→</span>
              <span>Planning</span>
              <span className={styles.arrow}>→</span>
              <span>Curating Resources</span>
              <span className={styles.arrow}>→</span>
              <span>Drafting Quiz</span>
            </div>
          </div>
        )}

        {!loading && !result && (
          <div className={styles.emptyState}>
            <h2>Awaiting Data</h2>
            <p>Please complete your self assessment to unlock your personalized Study Coach.</p>
            {!assessment && <a href="/assess" className="btn-primary" style={{ marginTop: "1rem", display: "inline-block" }}>Take Assessment</a>}
          </div>
        )}

        {!loading && result && (
          <>
            <div className={styles.tabs}>
              {([
                ["diagnosis", "Learning Diagnosis"],
                ["plan", "Interactive Study Plan"],
                ["resources", "Curated Resources"],
                ["quiz", "Practice Quiz"],
              ] as const).map(([key, label]) => (
                <button
                  key={key}
                  className={`${styles.tab} ${activeTab === key ? styles.tabActive : ""}`}
                  onClick={() => setActiveTab(key)}
                >
                  {label}
                </button>
              ))}
            </div>

            {activeTab === "diagnosis" && (
              <div className={styles.section}>
                <h3>Your Learning Diagnosis</h3>
                <div className={styles.prose}>{result.diagnosis}</div>
              </div>
            )}

            {activeTab === "plan" && (
              <div className={styles.section}>
                <h3>Your 4-Week Action Plan</h3>
                {parsePlan(result.study_plan).length > 0 ? (
                  <div className={styles.weeks}>
                    {parsePlan(result.study_plan).map((week, wi) => (
                      <div key={wi} className={styles.week}>
                        <div className={styles.weekHeader}>{week.title}</div>
                        <ul className={styles.taskList}>
                          {week.tasks.map((task, ti) => {
                            const key = `${wi}-${ti}`;
                            return (
                              <li key={key} className={`${styles.task} ${checkedTasks.has(key) ? styles.taskDone : ""}`}>
                                <input type="checkbox" checked={checkedTasks.has(key)}
                                  onChange={() => toggleTask(key)} />
                                <span>{task}</span>
                              </li>
                            );
                          })}
                        </ul>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className={styles.prose}>{result.study_plan}</div>
                )}
              </div>
            )}

            {activeTab === "resources" && (
              <div className={styles.section}>
                <h3>Curated High-Value Resources</h3>
                <div className={styles.prose}>{result.resources}</div>
              </div>
            )}

            {activeTab === "quiz" && (
              <div className={styles.section}>
                <h3>Interactive Practice Quiz</h3>
                <p style={{ marginBottom: "1.5rem", fontSize: "0.88rem", color: "var(--text-secondary)" }}>
                  Contains {quizItems.length} multiple-choice questions targeting your identified weak points.
                </p>
                {quizItems.length > 0 ? (
                  <div className={styles.quizList}>
                    {quizItems.map((q, i) => (
                      <div key={i} className={styles.quizCard}>
                        <p className={styles.qNum}>Question {i + 1}</p>
                        <p className={styles.qText}>{q.question}</p>
                        <ul className={styles.options}>
                          {q.options.map((o, oi) => (
                            <li key={oi} className={`${styles.option} ${q.revealed && o.startsWith(q.answer) ? styles.correct : ""}`}>
                              {o}
                            </li>
                          ))}
                        </ul>
                        {q.revealed ? (
                          <div className={styles.explanation}>
                            <strong>Answer: {q.answer}</strong> — {q.explanation}
                          </div>
                        ) : (
                          <button className="btn-secondary" style={{ marginTop: "0.75rem", width: "auto", padding: "0.4rem 1rem", fontSize: "0.83rem" }}
                            onClick={() => revealAnswer(i)}>
                            Reveal Answer & Explanation
                          </button>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className={styles.prose}>{result.quiz}</div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
