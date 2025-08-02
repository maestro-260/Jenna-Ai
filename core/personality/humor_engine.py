import random

class HumorEngine:
    """Generates witty, funny, or sarcastic responses for the assistant."""
    def __init__(self):
        self.jokes = [
            "Why did the computer show up at work late? It had a hard drive!",
            "I'm reading a book on anti-gravity. It's impossible to put down!",
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "I'd tell you a UDP joke, but you might not get it.",
            "Why did the AI go to therapy? It had too many neural issues!"
        ]
        self.sarcasm_templates = [
            "Oh, absolutely, because that's never gone wrong before...",
            "Sure, let me just hack into the mainframe... again!",
            "Of course, because I'm totally not just a bunch of code...",
            "Right, because that's exactly what I was programmed for!"
        ]
        self.witty_templates = [
            "If I had a nickel for every time someone asked me that...",
            "Well, that's one way to look at it!",
            "I see what you did there!",
            "You humans are so creative!"
        ]

    def get_joke(self):
        return random.choice(self.jokes)

    def get_sarcasm(self):
        return random.choice(self.sarcasm_templates)

    def get_witty(self):
        return random.choice(self.witty_templates)

    def get_humorous_reply(self, style: str = "joke"):
        if style == "joke":
            return self.get_joke()
        elif style == "sarcasm":
            return self.get_sarcasm()
        elif style == "witty":
            return self.get_witty()
        else:
            return self.get_joke()
