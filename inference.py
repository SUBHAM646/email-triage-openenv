import os
from openai import OpenAI
from app.env import EmailEnv
from app.tasks import TASKS
from app.models import Action

print("[START]")

client = OpenAI(
    api_key=os.environ.get("API_KEY"),
    base_url=os.environ.get("API_BASE_URL")
)

env = EmailEnv()

for task in TASKS:
    print(f"[STEP] Task: {task['name']}")

    obs = env.reset()

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",   # ✅ SAFE MODEL
            messages=[
                {
                    "role": "user",
                    "content": f"Classify email: {obs.subject}"
                }
            ],
            max_tokens=50
        )

        result = response.choices[0].message.content
        print("[DEBUG] LLM Response:", result)

    except Exception as e:
        print("[ERROR]", e)
        result = "normal"

    # fallback logic
    if "spam" in result.lower():
        action = Action(category="spam", priority=3, route="none")
    elif "urgent" in result.lower():
        action = Action(category="urgent", priority=1, route="support")
    else:
        action = Action(category="normal", priority=2, route="hr")

    obs, reward, done, _ = env.step(action)

    print(f"[STEP] Score: {reward}")

print("[END]")