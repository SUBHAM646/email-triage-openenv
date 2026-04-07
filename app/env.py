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

    def reset(self):
        self.current_email = random.choice(EMAILS)
        return self.state()

    def state(self):
        return Observation(
            email_text=self.current_email["text"],
            sender=self.current_email["sender"],
            subject=self.current_email["subject"]
        )

    def step(self, action):
        done = True
        correct = False

        # simple logic
        if "spam" in self.current_email["sender"]:
            correct = action.category == "spam"
        elif "urgent" in self.current_email["subject"].lower():
            correct = action.category == "urgent"
        else:
            correct = action.category == "normal"

        reward = 1.0 if correct else 0.0

        return self.state(), reward, done, {}