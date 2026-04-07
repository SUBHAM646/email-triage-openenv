import os
from openai import OpenAI
from app.env import EmailEnv
from app.tasks import TASKS
from app.models import Action

print("[START]")

# ✅ IMPORTANT: use their proxy
client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["API_BASE_URL"]
)

env = EmailEnv()

for task in TASKS:
    print(f"[STEP] Task: {task['name']}")

    obs = env.reset()

    # ✅ LLM CALL (IMPORTANT)
    prompt = f"""
    Classify this email:
    Subject: {obs.subject}
    Sender: {obs.sender}
    Body: {obs.email_text}

    Return JSON:
    {{
      "category": "spam/urgent/normal",
      "priority": 1-3,
      "route": "support/hr/sales/none"
    }}
    """

    response = client.chat.completions.create(
        model=os.environ["MODEL_NAME"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    result = response.choices[0].message.content

    # simple fallback parsing
    if "spam" in result:
        action = Action(category="spam", priority=3, route="none")
    elif "urgent" in result:
        action = Action(category="urgent", priority=1, route="support")
    else:
        action = Action(category="normal", priority=2, route="hr")

    obs, reward, done, _ = env.step(action)

    print(f"[STEP] Score: {reward}")

print("[END]")