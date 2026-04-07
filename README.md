---
title: Email Triage OpenEnv
emoji: "📧"
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Email Triage OpenEnv Environment

## Overview

This project implements a real-world **Email Triage System** as an OpenEnv-compatible environment, where an AI agent learns to classify, prioritize, and route emails based on contextual understanding.

The goal is to simulate how humans manage emails in professional settings such as customer support, HR, and internal communication.

---

## Real-World Motivation

Email overload is a major challenge in organizations. Efficient triaging helps:

* Reduce response time
* Improve productivity
* Automate repetitive workflows

This environment provides a structured setup for training and evaluating AI agents on email decision-making tasks.

---

## Environment Design

### Observation Space

The agent receives:

* `email_text`: Body of the email
* `sender`: Email sender
* `subject`: Email subject line

---

### Action Space

The agent must decide:

* `category`: spam / urgent / normal
* `priority`: 1 (high) → 3 (low)
* `route`: support / hr / sales / none

---

### Reward Function

* Correct classification → **1.0**
* Incorrect classification → **0.0**

This ensures a clear and deterministic reward signal.

---

## Tasks

The environment includes **3 tasks**:

| Task   | Description                |
| ------ | -------------------------- |
| Easy   | Basic email classification |
| Medium | Moderate complexity emails |
| Hard   | Ambiguous or tricky emails |

Each task evaluates the agent’s ability to correctly triage emails.

---

## Baseline Agent

A simple rule-based agent is implemented in `inference.py`:

* Detects spam via sender
* Detects urgency via subject
* Assigns default routing

### Sample Output

```
[START]
[STEP] Task: easy
[STEP] Score: 1.0
[STEP] Task: medium
[STEP] Score: 1.0
[STEP] Task: hard
[STEP] Score: 1.0
[END]
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd email-triage-env
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run inference

```bash
python inference.py
```

---

## Docker Setup

```bash
docker build -t email-triage-env .
docker run email-triage-env
```

---

## Project Structure

```
email-triage-env/
│
├── app/
│   ├── env.py
│   ├── models.py
│   ├── tasks.py
│   ├── grader.py
│
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Future Improvements

* Integrate LLM-based understanding
* Add multi-step decision-making
* Introduce partial rewards
* Expand dataset with real emails

---

## Conclusion

This project demonstrates a clean, scalable, and real-world OpenEnv environment for training AI agents in email triaging tasks.

It is lightweight, reproducible, and aligned with real-world applications.

---
