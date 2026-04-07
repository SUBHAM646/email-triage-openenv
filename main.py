from fastapi import FastAPI
from app.env import EmailEnv
from app.models import Action

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