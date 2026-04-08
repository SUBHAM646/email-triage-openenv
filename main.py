import os
from fastapi import FastAPI
from openai import OpenAI
from app.env import EmailEnv
from app.models import Action
from typing import Dict

app = FastAPI()
env = EmailEnv()

# ✅ ROOT
@app.get("/")
def home():
    return {"message": "Email Triage API Running 🚀"}

# ✅ RESET
@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()

# ✅ STEP (FIXED)
@app.post("/step")
def step(action: Dict):
    try:
        action = Action(**action)

        result = "normal"

        try:
            obs = env.state()
        except:
            obs = env.reset()

        try:
            api_key = os.environ.get("API_KEY")
            base_url = os.environ.get("API_BASE_URL")

            if api_key and base_url:
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": f"Classify email: {obs.subject}"}
                    ],
                    max_tokens=50
                )

                result = response.choices[0].message.content

        except Exception as e:
            return {"error": f"LLM Error: {str(e)}"}

        obs, reward, done, info = env.step(action)

        return {
            "observation": obs.dict(),
            "reward": reward,
            "done": done,
            "info": info
        }

    except Exception as e:
        return {"error": f"Main Error: {str(e)}"}

# ✅ STATE
@app.get("/state")
def state():
    return env.state().dict()