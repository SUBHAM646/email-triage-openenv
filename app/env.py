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
        # Deterministic grader scores - strictly between 0 and 1
        self.grader_config = {
            "easy": {"correct": 0.75, "incorrect": 0.25},
            "medium": {"correct": 0.70, "incorrect": 0.30},
            "hard": {"correct": 0.65, "incorrect": 0.35},
        }

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

        # Get score from grader config
        if task_id and task_id in self.grader_config:
            config = self.grader_config[task_id]
            reward = config["correct"] if correct else config["incorrect"]
        else:
            # Default fallback - but always use task_id in production
            reward = 0.70 if correct else 0.30

        # Ensure reward is strictly between 0 and 1
        reward = float(max(0.01, min(0.99, reward)))

        return self.state(), reward, done, {}