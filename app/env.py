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

        # ✅ FIXED REWARD (IMPORTANT)
        if correct:
            reward = 0.75
        else:
            reward = 0.35

        return self.state(), reward, done, {}