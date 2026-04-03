"use client";
import Link from "next/link";
import styles from "./page.module.css";

export default function Home() {
  return (
    <div className={styles.page}>
      <div className={styles.hero}>
        <div className={styles.badge}>Milestone 2 — Agentic AI</div>
        <h1 className={styles.title}>
          Your intelligent<br />study coach.
        </h1>
        <p className={styles.sub}>
          ILAAS analyses your academic performance, identifies where you&apos;re falling behind,
          and builds a personalised study plan — week by week — using an AI agent.
        </p>
        <div className={styles.actions}>
          <Link href="/assess" className="btn-primary">Start Assessment</Link>
          <Link href="/coach" className="btn-secondary">Go to Study Coach</Link>
        </div>
      </div>

      <div className={styles.features}>
        <div className={styles.feature}>
          <div className={styles.featureLabel}>Performance Analysis</div>
          <p>Enter your recent scores and weak topics. Our ML model classifies your current risk level based on historical student data.</p>
        </div>
        <div className={styles.feature}>
          <div className={styles.featureLabel}>Learning Diagnosis</div>
          <p>The AI agent reads your performance profile and produces a clear, plain-language diagnosis of where you need to focus.</p>
        </div>
        <div className={styles.feature}>
          <div className={styles.featureLabel}>4-Week Study Plan</div>
          <p>A structured week-by-week plan with specific daily tasks, calibrated to your available study hours and timeline.</p>
        </div>
        <div className={styles.feature}>
          <div className={styles.featureLabel}>Practice Quiz</div>
          <p>5 multiple-choice questions targeting your exact weak topics, with answers and explanations you can reveal on demand.</p>
        </div>
      </div>

      <div className={styles.strip}>
        <span>Powered by LangGraph</span>
        <span className={styles.dot} />
        <span>Groq API</span>
        <span className={styles.dot} />
        <span>scikit-learn Random Forest</span>
        <span className={styles.dot} />
        <span>UCI Student Performance Dataset</span>
      </div>
    </div>
  );
}
