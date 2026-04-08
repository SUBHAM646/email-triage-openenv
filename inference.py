import os
from openai import OpenAI
from app.env import EmailEnv
from app.models import Action

print("[START]")

# Initialize environment and LLM
env = EmailEnv()
client = OpenAI(
    api_key=os.environ.get("API_KEY"),
    base_url=os.environ.get("API_BASE_URL")
)

# 3 Tasks with task IDs that match openenv.yaml
TASKS = [
    {"id": "easy", "email_index": 0},
    {"id": "medium", "email_index": 1},
    {"id": "hard", "email_index": 2}
]

# Track results
results = []

for task in TASKS:
    task_id = task["id"]
    email_index = task["email_index"]
    
    print(f"\n=== Task: {task_id} ===")
    
    # Reset environment to specific email
    obs = env.reset(index=email_index)
    print(f"Email: {obs.subject} from {obs.sender}")
    
    # Get LLM classification
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"Classify this email as 'spam', 'urgent', or 'normal'. Email subject: {obs.subject}. Email text: {obs.email_text}"
                }
            ],
            max_tokens=50,
            temperature=0
        )
        result = response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"[WARNING] LLM Error: {e}. Using fallback classification.")
        result = "normal"
    
    # Parse LLM response to action
    if "spam" in result:
        action = Action(category="spam", priority=3, route="none")
    elif "urgent" in result:
        action = Action(category="urgent", priority=1, route="support")
    else:
        action = Action(category="normal", priority=2, route="hr")
    
    print(f"Classification: {action.category}")
    
    # Step environment with task_id for proper grading
    obs, reward, done, info = env.step(action, task_id=task_id)
    
    # Ensure score is strictly within (0, 1)
    score = float(reward)
    score = max(0.01, min(0.99, score))
    
    # Store result
    results.append({
        "task_id": task_id,
        "score": score
    })
    
    # Print score in validator-compatible format
    print(f"Score: {score}")

print("\n=== Summary ===")
for result in results:
    print(f"{result['task_id']}: {result['score']}")

print("[END]")