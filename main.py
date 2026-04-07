from fastapi import FastAPI
from app.env import EmailEnv
from app.tasks import TASKS
from app.models import Action

app = FastAPI()

@app.get("/")
def run_env():
    results = []

    env = EmailEnv()

    for task in TASKS:
        obs = env.reset()

        if "spam" in obs.sender:
            action = Action(category="spam", priority=3, route="none")
        elif "urgent" in obs.subject.lower():
            action = Action(category="urgent", priority=1, route="support")
        else:
            action = Action(category="normal", priority=2, route="hr")

        obs, reward, done, _ = env.step(action)

        results.append({
            "task": task["name"],
            "score": reward,
            "email": obs.subject
        })

    return {
        "status": "success",
        "results": results
    }