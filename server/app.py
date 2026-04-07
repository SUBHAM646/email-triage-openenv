from app.env import EmailEnv
from app.models import Action

env = EmailEnv()

def reset():
    obs = env.reset()
    return obs.dict()

def step(action_dict):
    action = Action(**action_dict)
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }

def state():
    obs = env.state()
    return obs.dict()

# ✅ IMPORTANT: REQUIRED MAIN FUNCTION
def main():
    print("OpenEnv Email Triage Environment Running")

# ✅ REQUIRED ENTRY POINT
if __name__ == "__main__":
    main()