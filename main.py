import os
from openai import OpenAI
from fastapi import FastAPI
from app.env import EmailEnv
from app.models import Action

client = OpenAI(
    api_key=os.environ.get("API_KEY"),
    base_url=os.environ.get("API_BASE_URL")
)

app = FastAPI()

env = EmailEnv()

# ✅ ROOT (ADD THIS)
@app.get("/")
def home():
    return {"message": "Email Triage API Running 🚀"}

# ✅ RESET
@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()

# ✅ STEP
@app.post("/step")
def step(action: Action):
    obs = env.state()

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
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

    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }

# ✅ STATE
@app.get("/state")
def state():
    obs = env.state()
    return obs.dict()