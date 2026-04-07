"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import styles from "./page.module.css";

const SUBJECTS = ["Mathematics", "Physics", "Chemistry", "Biology", "Computer Science", "Economics", "History", "English"];
const STUDYTIME_OPTIONS = [
  { label: "Less than 2 hrs/week", value: 1 },
  { label: "2–5 hrs/week", value: 2 },
  { label: "5–10 hrs/week", value: 3 },
  { label: "More than 10 hrs/week", value: 4 },
];

type FormData = {
  name: string;
  subject: string;
  goal: string;
  g1: string;
  g2: string;
  failures: string;
  absences: string;
  studytime: number;
  weakTopicsInput: string;
  weakTopics: string[];
  schoolsup: string;
  famsup: string;
  internet: string;
  higher: string;
};

const initial: FormData = {
  name: "", subject: "Mathematics", goal: "",
  g1: "", g2: "", failures: "0", absences: "0",
  studytime: 2, weakTopicsInput: "", weakTopics: [],
  schoolsup: "no", famsup: "yes", internet: "yes", higher: "yes",
};

export default function AssessPage() {
  const router = useRouter();
  const [step, setStep] = useState(1);
  const [form, setForm] = useState<FormData>(initial);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const set = (key: keyof FormData, val: string | number | string[]) =>
    setForm((f) => ({ ...f, [key]: val }));

  const addTopic = () => {
    const t = form.weakTopicsInput.trim();
    if (t && !form.weakTopics.includes(t)) {
      set("weakTopics", [...form.weakTopics, t]);
    }
    set("weakTopicsInput", "");
  };

  const removeTopic = (t: string) =>
    set("weakTopics", form.weakTopics.filter((x) => x !== t));

  const submit = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          school: "GP", sex: "M",
          age: 17, address: "U",
          studytime: form.studytime,
          failures: parseInt(form.failures) || 0,
          schoolsup: form.schoolsup,
          famsup: form.famsup,
          paid: "no",
          higher: form.higher,
          internet: form.internet,
          traveltime: 1,
          absences: parseInt(form.absences) || 0,
          G1: parseInt(form.g1) || 10,
          G2: parseInt(form.g2) || 10,
        }),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      localStorage.setItem("ilaas_assessment", JSON.stringify({
        name: form.name,
        subject: form.subject,
        goal: form.goal,
        weakTopics: form.weakTopics.length ? form.weakTopics : [form.subject + " fundamentals"],
        studyHours: form.studytime * 2,
        mlRisk: data.prediction,
        predictedG3: data.predicted_g3,
        scores: [parseInt(form.g1) * 5, parseInt(form.g2) * 5],
      }));

      router.push("/coach");
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Request failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const stepLabels = ["Your Profile", "Academic Grades", "Study Habits"];

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <h1>Self Assessment</h1>
        <p>Answer honestly — the AI coach uses this to build your study plan.</p>
      </div>

      <div className={styles.steps}>
        {stepLabels.map((label, i) => (
          <div key={i} className={`${styles.stepItem} ${step > i + 1 ? styles.done : ""} ${step === i + 1 ? styles.active : ""}`}>
            <div className={styles.stepNum}>{step > i + 1 ? "✓" : i + 1}</div>
            <span>{label}</span>
          </div>
        ))}
      </div>

      <div className={styles.card}>
        {step === 1 && (
          <div className={styles.fields}>
            <h2>Tell us about yourself</h2>
            <div className={styles.field}>
              <label className="label">Your name</label>
              <input className="input-field" placeholder="e.g. Arjun" value={form.name}
                onChange={(e) => set("name", e.target.value)} />
            </div>
            <div className={styles.field}>
              <label className="label">Subject you need help with</label>
              <select className="input-field" value={form.subject}
                onChange={(e) => set("subject", e.target.value)}>
                {SUBJECTS.map((s) => <option key={s}>{s}</option>)}
              </select>
            </div>
            <div className={styles.field}>
              <label className="label">What is your goal?</label>
              <textarea className="input-field" rows={3}
                placeholder="e.g. Score at least 75% in my final Math exam this semester."
                value={form.goal} onChange={(e) => set("goal", e.target.value)} />
            </div>
            <div className={styles.field}>
              <label className="label">Weak topics (add one at a time)</label>
              <div className={styles.tagRow}>
                <input className="input-field" placeholder="e.g. Algebra"
                  value={form.weakTopicsInput}
                  onChange={(e) => set("weakTopicsInput", e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && addTopic()} />
                <button className="btn-secondary" onClick={addTopic}>Add</button>
              </div>
              {form.weakTopics.length > 0 && (
                <div className={styles.tags}>
                  {form.weakTopics.map((t) => (
                    <span key={t} className="tag">
                      {t}
                      <button onClick={() => removeTopic(t)} style={{ background: "none", color: "inherit", fontSize: "0.9rem", lineHeight: 1 }}>×</button>
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {step === 2 && (
          <div className={styles.fields}>
            <h2>Academic performance</h2>
            <p style={{ marginBottom: "1.5rem", fontSize: "0.9rem" }}>
              Enter grades out of 20 (Portuguese grading scale). Use your most recent results.
            </p>
            <div className={styles.row2}>
              <div className={styles.field}>
                <label className="label">Semester 1 grade (0–20)</label>
                <input className="input-field" type="number" min={0} max={20}
                  value={form.g1} onChange={(e) => set("g1", e.target.value)} placeholder="e.g. 12" />
              </div>
              <div className={styles.field}>
                <label className="label">Semester 2 grade (0–20)</label>
                <input className="input-field" type="number" min={0} max={20}
                  value={form.g2} onChange={(e) => set("g2", e.target.value)} placeholder="e.g. 14" />
              </div>
            </div>
            <div className={styles.row2}>
              <div className={styles.field}>
                <label className="label">Previous class failures</label>
                <input className="input-field" type="number" min={0} max={4}
                  value={form.failures} onChange={(e) => set("failures", e.target.value)} />
              </div>
              <div className={styles.field}>
                <label className="label">Days absent this year</label>
                <input className="input-field" type="number" min={0}
                  value={form.absences} onChange={(e) => set("absences", e.target.value)} />
              </div>
            </div>
          </div>
        )}

        {step === 3 && (
          <div className={styles.fields}>
            <h2>Study habits</h2>
            <div className={styles.field}>
              <label className="label">Weekly study hours</label>
              <div className={styles.optionGrid}>
                {STUDYTIME_OPTIONS.map((o) => (
                  <button key={o.value}
                    className={`${styles.optionBtn} ${form.studytime === o.value ? styles.optionActive : ""}`}
                    onClick={() => set("studytime", o.value)}>
                    {o.label}
                  </button>
                ))}
              </div>
            </div>
            <div className={styles.row2}>
              <div className={styles.field}>
                <label className="label">Do you have internet at home?</label>
                <select className="input-field" value={form.internet}
                  onChange={(e) => set("internet", e.target.value)}>
                  <option value="yes">Yes</option>
                  <option value="no">No</option>
                </select>
              </div>
              <div className={styles.field}>
                <label className="label">Planning to go to university?</label>
                <select className="input-field" value={form.higher}
                  onChange={(e) => set("higher", e.target.value)}>
                  <option value="yes">Yes</option>
                  <option value="no">No</option>
                </select>
              </div>
            </div>
            <div className={styles.row2}>
              <div className={styles.field}>
                <label className="label">Family helps with studying?</label>
                <select className="input-field" value={form.famsup}
                  onChange={(e) => set("famsup", e.target.value)}>
                  <option value="yes">Yes</option>
                  <option value="no">No</option>
                </select>
              </div>
              <div className={styles.field}>
                <label className="label">Extra school support?</label>
                <select className="input-field" value={form.schoolsup}
                  onChange={(e) => set("schoolsup", e.target.value)}>
                  <option value="no">No</option>
                  <option value="yes">Yes</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {error && <p className={styles.error}>{error}</p>}

        <div className={styles.footer}>
          {step > 1 && (
            <button className="btn-secondary" onClick={() => setStep(step - 1)}>Back</button>
          )}
          {step < 3 ? (
            <button className="btn-primary" onClick={() => setStep(step + 1)}>
              Continue
            </button>
          ) : (
            <button className="btn-primary" onClick={submit} disabled={loading}>
              {loading ? "Analysing..." : "Generate Study Plan"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
