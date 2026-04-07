from pydantic import BaseModel

class Observation(BaseModel):
    email_text: str
    sender: str
    subject: str

class Action(BaseModel):
    category: str   # spam / urgent / normal
    priority: int   # 1–3
    route: str      # sales / support / hr

class Reward(BaseModel):
    score: float