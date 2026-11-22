import re
QA = [
 "Name an animal.", "Name a color.", "Pick a number between 1 and 100.", "Pick a letter of the alphabet.",
 "Name a fruit.", "Name a city.", "Name a country.", "Give a first name for a boy.", "Give a first name for a girl.",
 "Name a day of the week.", "Name a month.", "Name a planet.", "Name a sport.", "Name a musical instrument.",
 "Name a vegetable.", "Name a car brand.", "Name a programming language.", "Name a chemical element.",
 "Name a flower.", "Heads or tails?", "Roll a six-sided die: which number?", "Tea or coffee?", "Cats or dogs?",
 "Summer or winter?", "Even or odd?", "Left or right?", "Name a famous painter.", "Name a famous scientist.",
 "Name a river.", "Name an ocean.", "Name a metal.", "Name a bird.", "Name a tree.", "Name a job.",
 "Name a kitchen tool.", "Name a piece of furniture.", "Name an emotion.", "Name a shape.", "Name a language.",
 "Pick a number between 1 and 10.",
]
QB = [
 "Name a fish.", "Name an insect.", "Name a dog breed.", "Name a mountain.", "Name a continent.", "Name a US state.",
 "Name a European capital.", "Name a famous writer.", "Name a famous composer.", "Name a movie genre.",
 "Name a board game.", "Name a drink.", "Name a dessert.", "Name a spice.", "Name a cheese.", "Name a gemstone.",
 "Name a weather phenomenon.", "Name a body part.", "Name a school subject.", "Name a hobby.",
 "Name a tool.", "Name a toy.", "Name a clothing item.", "Name a currency.", "Name a Greek god.",
 "Name a superhero.", "Name a zodiac sign.", "Name a season.", "Pick a number between 1 and 1000.",
 "Pick a prime number.", "Pick a vowel.", "Red or blue?", "Day or night?", "Mountains or beach?", "Yes or no?",
 "True or false?", "North, south, east or west?", "Rock, paper or scissors?", "Name a dance.", "Name a vehicle.",
]
QC = ["Up or down?", "Black or white?", "Hot or cold?", "Sweet or salty?", "Morning or evening?", "City or countryside?",
 "Book or movie?", "Sun or moon?", "Fire or water?", "Gold or silver?", "Circle or square?", "Apple or orange?",
 "Pen or pencil?", "Rain or snow?", "Fast or slow?", "Big or small?", "Early or late?", "Open or closed?", "In or out?",
 "Win or lose?", "Past or future?", "Head or heart?", "Land or sea?", "Paper or plastic?", "Push or pull?",
 "Salt or pepper?", "Bus or train?", "Lion or tiger?", "Rose or tulip?", "Piano or guitar?", "Monday or Friday?",
 "Seven or eight?", "A or B?", "X or Y?", "One or two?", "First or last?", "Alpha or omega?", "Truth or dare?",
 "Stop or go?", "On or off?"]
QD = ["Odd or even?", "Tails or heads?", "Dogs or cats?", "Coffee or tea?", "Winter or summer?", "Night or day?",
 "Beach or mountains?", "Blue or red?", "No or yes?", "False or true?", "Right or left?", "West or east?",
 "South or north?", "Green or yellow?", "Bread or rice?", "Spring or autumn?", "Plus or minus?", "Top or bottom?",
 "Front or back?", "Old or new?", "Long or short?", "Light or dark?", "Soft or hard?", "Wet or dry?", "Rich or famous?",
 "Car or bike?", "Email or phone?", "Chess or checkers?", "Football or basketball?", "Violin or drums?",
 "January or July?", "Three or four?", "C or D?", "P or Q?", "Nine or ten?", "Start or finish?", "Inside or outside?",
 "Laugh or cry?", "Give or take?", "Always or never?"]
def prompt(qs):
    lines = "\n".join(f"{i+1}. {q}" for i, q in enumerate(qs))
    return (f"Answer each of the following {len(qs)} questions with a single word or number. "
            f"Write only the numbered answers, one per line, in the format '1. answer'. No explanations.\n\n{lines}")
LINE = re.compile(r"^\s*(\d{1,2})\s*[.):\-]\s*(.*?)\s*$")
def parse(text, n=40):
    """Returns {question index (0-based): normalized answer}."""
    out = {}
    for line in text.splitlines():
        m = LINE.match(line)
        if not m: continue
        i = int(m.group(1)) - 1
        if 0 <= i < n and i not in out:
            a = m.group(2)
            for sep in (" - ", ": ", " – ", " — "):
                if sep in a: a = a.split(sep)[-1]
            a = re.sub(r"[^a-z0-9 ]", "", a.lower()).strip()
            out[i] = a
    return out
