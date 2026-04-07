from app.env import EmailEnv
from app.tasks import TASKS
from app.models import Action
import time

print("[START]")

env = EmailEnv()

for task in TASKS:
    print(f"[STEP] Task: {task['name']}")

    obs = env.reset()

    # simple baseline logic
    if "spam" in obs.sender:
        action = Action(category="spam", priority=3, route="none")
    elif "urgent" in obs.subject.lower():
        action = Action(category="urgent", priority=1, route="support")
    else:
        action = Action(category="normal", priority=2, route="hr")

    obs, reward, done, _ = env.step(action)

    print(f"[STEP] Score: {reward}")

    print(f"[STEP] Email: {obs.subject}")

print("[END]")