import random
from app.models import Observation

EMAILS = [
    {"text": "Buy now offer!!!", "sender": "ads@spam.com", "subject": "SALE"},
    {"text": "Server down urgent", "sender": "boss@company.com", "subject": "URGENT"},
    {"text": "Meeting tomorrow", "sender": "hr@company.com", "subject": "Meeting"},
]

class EmailEnv:
    def __init__(self):
        self.current_email = None

    def reset(self, index=None):
        if index is not None:
            self.current_email = EMAILS[index]
        else:
            self.current_email = random.choice(EMAILS)
        return self.state()

    def state(self):
        return Observation(
            email_text=self.current_email["text"],
            sender=self.current_email["sender"],
            subject=self.current_email["subject"]
        )

    def step(self, action, task_id=None):
        done = True
        correct = False

        # Deterministic grading logic
        if "spam" in self.current_email["sender"]:
            correct = action.category == "spam"
        elif "urgent" in self.current_email["subject"].lower():
            correct = action.category == "urgent"
        else:
            correct = action.category == "normal"

        # Task-specific reward map (all strictly between 0 and 1)
        task_rewards = {
            "easy": {"correct": 0.75, "incorrect": 0.25},
            "medium": {"correct": 0.70, "incorrect": 0.30},
            "hard": {"correct": 0.65, "incorrect": 0.35}
        }

        # Get reward from task config
        if task_id in task_rewards:
            reward_config = task_rewards[task_id]
            reward = reward_config["correct"] if correct else reward_config["incorrect"]
        else:
            reward = 0.70 if correct else 0.30

        # CRITICAL: Ensure strictly between 0 and 1 (not inclusive)
        # Use explicit boundaries to prevent edge cases
        reward = float(reward)
        if reward <= 0.0:
            reward = 0.01
        if reward >= 1.0:
            reward = 0.99
        if reward < 0.01:
            reward = 0.01
        if reward > 0.99:
            reward = 0.99

        return self.state(), reward, done, {}