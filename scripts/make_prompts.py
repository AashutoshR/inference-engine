"""Build tests/data/prompts.jsonl: the 200-prompt Gate 1 test set.

    python scripts/make_prompts.py                       # write the file
    python scripts/make_prompts.py --model models/X      # also print token-length stats

Deterministic: the long documents come from a seeded generator, so rerunning gives the
same file. Categories follow the Phase 1 plan: chat, code, math, long documents,
non-English text, emoji, multi-turn, and system prompts.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "data" / "prompts.jsonl"

CHAT = [
    "What is the capital of France?",
    "Hi!",
    "Why is the sky blue?",
    "Give me three tips for sleeping better.",
    "What's the difference between weather and climate?",
    "How do vaccines train the immune system?",
    "Suggest a name for a grey cat.",
    "Explain inflation to a ten-year-old.",
    "What should I pack for a weekend hiking trip?",
    "Is a tomato a fruit or a vegetable?",
    "How does a refrigerator keep food cold?",
    "Write a haiku about autumn rain.",
    "What are the pros and cons of working from home?",
    "How can I politely decline a meeting invitation?",
    "Describe the water cycle in one paragraph.",
    "What is a black hole?",
    "Recommend a good habit to start this week.",
    "How do I make a cup of pour-over coffee?",
    "Why do leaves change color in the fall?",
    "Write a two-line poem about the ocean.",
    "What does a product manager do?",
    "Explain the rules of chess in short bullet points.",
    "How many continents are there, and what are they?",
    "Give me a 5-minute desk stretching routine.",
    "What is the difference between a virus and a bacterium?",
    "Write a short thank-you note to a teacher.",
    "Why do we have leap years?",
    "What is machine learning, in one sentence?",
    "List four renewable energy sources.",
    "How do I remove a coffee stain from a cotton shirt?",
    "Tell me a fun fact about octopuses.",
    "What is the tallest mountain on Earth?",
    "Summarize the plot of a typical heist movie.",
    "What is compound interest?",
    "Explain what an API is using a restaurant analogy.",
]

CODE = [
    "Write a Python function that checks whether a string is a palindrome.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain what this does: `[x * x for x in range(10) if x % 2 == 0]`",
    "Write a SQL query that returns the top 5 customers by total order amount.",
    "What is the difference between a list and a tuple in Python?",
    "Write a JavaScript function that debounces another function.",
    "Write a bash one-liner that counts lines in all .py files in a directory tree.",
    "Fix the bug: `def avg(xs): return sum(xs) / len(xs)` crashes on an empty list.",
    "Write a Rust function that reverses a vector in place.",
    "Explain Big-O notation with an example.",
    "Write a Python class for a stack with push, pop and peek.",
    "What does `git rebase` do, compared to `git merge`?",
    "Write a regular expression that matches a US ZIP code (5 digits or ZIP+4).",
    "Write a C function that computes the length of a null-terminated string.",
    "Explain what a race condition is and how a mutex prevents it.",
    "Write a Python function that merges two sorted lists into one sorted list.",
    "Convert this loop to a list comprehension:\n\nresult = []\nfor w in words:\n"
    "    if len(w) > 3:\n        result.append(w.upper())",
    "Write a TypeScript interface for a blog post with title, body, tags and a date.",
    "What is the difference between a process and a thread?",
    "Write a Go function that returns the maximum of a slice of ints.",
    "Explain what this SQL does: SELECT dept, AVG(salary) FROM emp GROUP BY dept "
    "HAVING AVG(salary) > 50000;",
    "Write a Python decorator that times how long a function takes.",
    "What is memoization? Show a short Python example.",
    "Write a binary search in Python and explain its complexity.",
    "Explain what `async` and `await` do in Python.",
]

MATH = [
    "A train travels 180 km in 2.5 hours. What is its average speed in km/h? Show your steps.",
    "What is 17 times 24?",
    "Solve for x: 3x + 7 = 22.",
    "What is the derivative of x^3 + 2x^2 - 5x + 1?",
    "A shirt costs $40 and is on sale for 25% off. What is the sale price?",
    "What is the sum of the integers from 1 to 100?",
    "Is 221 a prime number? Explain.",
    "A rectangle has a perimeter of 30 cm and a width of 5 cm. What is its area?",
    "Convert 72 degrees Fahrenheit to Celsius.",
    "What is the probability of rolling two sixes with two fair dice?",
    "Simplify (2x^2 * 3x^3) / (6x^4).",
    "If you invest $1,000 at 5% annual interest compounded yearly, how much after 3 years?",
    "What is the integral of 2x from 0 to 3?",
    "Find the greatest common divisor of 84 and 126.",
    "A recipe needs 3 eggs for 4 people. How many eggs for 10 people?",
    "What is 15% of 260?",
    "Solve the system: x + y = 10, x - y = 4.",
    "How many ways can 5 books be arranged on a shelf?",
    "What is the square root of 1764?",
    "A car's fuel use is 6.5 litres per 100 km. How much fuel for 340 km?",
]

NON_ENGLISH = [
    "Explique la photosynthèse en trois phrases.",
    "¿Cuál es la diferencia entre 'ser' y 'estar'?",
    "Erkläre in zwei Sätzen, wie ein Regenbogen entsteht.",
    "Qual è la capitale del Giappone?",
    "Escreva uma frase sobre o mar.",
    "Wat is de hoofdstad van Nederland?",
    "Объясни, что такое гравитация, простыми словами.",
    "日本の四季について簡単に説明してください。",
    "请用三句话介绍一下长城。",
    "한국의 전통 음식 세 가지를 알려 주세요.",
    "ما هي عاصمة مصر؟",
    "हिंदी में सूर्य के बारे में दो वाक्य लिखिए।",
    "Türkiye'nin en kalabalık şehri hangisidir?",
    "Napisz krótki wiersz o zimie.",
    "Hãy giải thích ngắn gọn về mưa.",
    "Ποια είναι η πρωτεύουσα της Ελλάδας;",
    "מה ההבדל בין חתול לכלב?",
    "Jelaskan apa itu fotosintesis dalam satu kalimat.",
    "Kuinka monta päivää on vuodessa?",
    "Translate to English: 'Le chat dort sur le canapé.'",
    "Translate to Spanish: 'Where is the train station?'",
    "Traduce al inglés: 'Me gusta aprender idiomas nuevos.'",
    "Svenska: vad är huvudstaden i Norge?",
    "Kiswahili: Habari za asubuhi? Jibu kwa Kiswahili.",
    "ภาษาไทย: กรุณาแนะนำอาหารไทยสองอย่าง",
]

EMOJI = [
    "Reply with five food emoji and their names, like 🍕 pizza.",
    "What does this emoji sequence mean: 🌧️☔🏠📚?",
    "Write a one-line birthday message with emoji 🎂🎉.",
    "Rank these by size: 🐘 🐁 🐕 🐋",
    "Turn this into emoji only: 'I love coffee in the morning.'",
    "What flags are these: 🇯🇵 🇧🇷 🇨🇦?",
    "Describe this weather forecast in words: ☀️☀️⛅🌧️⛈️",
    "Write a tweet about a cat 🐈 using three emoji.",
    "What is the Unicode name of ✓ and of ✗?",
    "Explain the family emoji 👨‍👩‍👧 and how it is built from smaller emoji.",
    "Use emoji to tell a three-step story about planting a seed.",
    "Which of these is a fruit: 🥕 🍓 🥔?",
    "Translate this math into words: 2 ➕ 3 🟰 5",
    "Write a café menu with prices: ☕ 🥐 🍰 (in euros, like €2,50).",
    "Pick an emoji for each season and explain why.",
]

MULTI_TURN = [
    [
        ("user", "My name is Sam."),
        ("assistant", "Nice to meet you, Sam!"),
        ("user", "What is my name?"),
    ],
    [
        ("user", "I'm planning a trip to Lisbon."),
        ("assistant", "Lisbon is lovely! How many days will you stay?"),
        ("user", "Three days. What should I see?"),
    ],
    [
        ("user", "Let's play a word game. I say a word, you say one that rhymes."),
        ("assistant", "Sounds fun! Go ahead."),
        ("user", "Light."),
    ],
    [
        ("user", "What's 12 squared?"),
        ("assistant", "12 squared is 144."),
        ("user", "And the square root of that plus 25?"),
    ],
    [
        ("user", "Write a function to add two numbers in Python."),
        ("assistant", "def add(a, b):\n    return a + b"),
        ("user", "Now add type hints and a docstring."),
    ],
    [
        ("user", "I have eggs, spinach and cheese."),
        ("assistant", "You could make an omelette or a frittata."),
        ("user", "How do I make the frittata?"),
    ],
    [
        ("user", "Remember this number: 4721."),
        ("assistant", "Got it: 4721."),
        ("user", "What number did I ask you to remember, plus one?"),
    ],
    [
        ("user", "Translate 'good night' to French."),
        ("assistant", "Bonne nuit."),
        ("user", "And to German?"),
    ],
    [
        ("user", "Recommend a sci-fi book."),
        ("assistant", "Try a classic space-exploration novel with a strong ensemble cast."),
        ("user", "Something shorter, please, under 200 pages."),
    ],
    [
        ("user", "I feel stressed about exams."),
        ("assistant", "That's understandable. When is your first exam?"),
        ("user", "Next Monday, and I haven't started revising."),
    ],
    [
        ("user", "Define entropy."),
        ("assistant", "Entropy measures disorder, or the number of microstates of a system."),
        ("user", "Give an everyday example."),
    ],
    [
        ("user", "Count from 1 to 5."),
        ("assistant", "1, 2, 3, 4, 5."),
        ("user", "Now backwards, in Spanish."),
    ],
    [
        ("user", "What's a good name for a bakery?"),
        ("assistant", "How about 'Rise & Shine'?"),
        ("user", "Give me three more with a pun."),
    ],
    [
        ("user", "Is Python compiled or interpreted?"),
        ("assistant", "CPython compiles source to bytecode, then interprets the bytecode."),
        ("user", "So why is it slower than C?"),
    ],
    [
        ("user", "Let's write a story together. Once upon a time, there was a robot."),
        ("assistant", "The robot lived alone in a lighthouse at the edge of the sea."),
        ("user", "Continue with two more sentences."),
    ],
]

SYSTEM = [
    ("You are a pirate. Always answer in pirate speak.", "How is the weather today?"),
    ("Answer with exactly one word.", "What color is grass?"),
    ("You are a concise assistant. Use at most two sentences.", "Explain how a CPU works."),
    ("You only reply in JSON with keys 'answer' and 'confidence'.", "What is 2 + 2?"),
    (
        "You are a helpful math tutor. Never give the final answer; give hints.",
        "How do I solve x^2 - 5x + 6 = 0?",
    ),
    ("Respond in French no matter what language the user writes in.", "What time is it?"),
    ("You are Shakespeare.", "Describe a smartphone."),
    ("You are a strict code reviewer.", "def f(l): return [i for i in l if i != None]"),
    ("Format every answer as a numbered list.", "How do I change a flat tire?"),
    ("You are a kind assistant for children. Use simple words.", "Why does the moon change shape?"),
    ("Always end your answer with the word 'Done.'", "Name three primary colors."),
    ("You are a SQL expert.", "How do I find duplicate emails in a users table?"),
    ("Reply in all lowercase.", "Tell me about the Eiffel Tower."),
    ("You are a travel agent.", "I have $500 and one week. Where should I go?"),
    (
        "Think step by step before answering.",
        "If 3 cats catch 3 mice in 3 minutes, how many cats catch 100 mice in 100 minutes?",
    ),
]

# --- Long documents (synthetic, seeded) -----------------------------------------------------

_NAMES = ["Ana", "Ben", "Chen", "Dara", "Eli", "Fatima", "Goran", "Hana", "Ivan", "Jae"]
_TEAMS = ["platform", "billing", "search", "mobile", "data", "security"]
_TOPICS = [
    "latency",
    "the Q3 roadmap",
    "on-call load",
    "the database migration",
    "hiring",
    "customer churn",
    "the new pricing page",
    "test flakiness",
]
_SERVICES = ["api-gateway", "auth", "payments", "search-indexer", "notifier", "scheduler"]
_LEVELS = ["INFO", "INFO", "INFO", "WARN", "ERROR", "DEBUG"]


def _meeting_notes(rng: random.Random, n: int) -> str:
    lines = [f"Meeting notes, {rng.choice(_TEAMS)} team sync"]
    for i in range(n):
        who, topic = rng.choice(_NAMES), rng.choice(_TOPICS)
        num = rng.randint(2, 40)
        lines.append(
            f"{i + 1}. {who} raised {topic}. The team agreed to revisit it in {num} days; "
            f"{rng.choice(_NAMES)} owns the follow-up and will share a written proposal."
        )
    return "\n".join(lines)


def _server_log(rng: random.Random, n: int) -> str:
    lines = []
    for i in range(n):
        svc, lvl = rng.choice(_SERVICES), rng.choice(_LEVELS)
        ms = rng.randint(3, 2500)
        lines.append(
            f"2026-10-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{i % 60:02d}:"
            f"{rng.randint(0, 59):02d}Z {lvl} {svc} request_id={rng.getrandbits(32):08x} "
            f"status={rng.choice([200, 200, 200, 201, 404, 500, 503])} duration_ms={ms}"
        )
    return "\n".join(lines)


def _csv_table(rng: random.Random, n: int) -> str:
    rows = ["region,month,units_sold,revenue_usd,returns"]
    for _ in range(n):
        units = rng.randint(50, 5000)
        rows.append(
            f"{rng.choice(['north', 'south', 'east', 'west'])},2026-{rng.randint(1, 9):02d},"
            f"{units},{units * rng.randint(8, 40)},{rng.randint(0, units // 20)}"
        )
    return "\n".join(rows)


def _story(rng: random.Random, n: int) -> str:
    places = ["a harbor town", "a mountain village", "an old library", "a night market"]
    acts = [
        "found a sealed letter",
        "repaired a broken clock",
        "followed a stray dog",
        "lost a map",
        "met a stranger with a lantern",
        "climbed the bell tower",
    ]
    sentences = []
    for _ in range(n):
        sentences.append(
            f"In {rng.choice(places)}, {rng.choice(_NAMES)} {rng.choice(acts)}, and by "
            f"evening {rng.choice(_NAMES)} knew that something had changed."
        )
    return " ".join(sentences)


def _code_file(rng: random.Random, n: int) -> str:
    funcs = []
    for i in range(n):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        funcs.append(
            f"def step_{i}(x):\n"
            f'    """Stage {i} of the pipeline."""\n'
            f"    if x % {a} == 0:\n        return x // {a} + {b}\n"
            f"    return x * {b} - {a}\n"
        )
    return "\n".join(funcs)


LONG_KINDS = [
    (_meeting_notes, "Summarize the decisions and list every owner mentioned.", (8, 60)),
    (
        _server_log,
        "Which service has the most errors (status 500 or 503)? Explain briefly.",
        (10, 60),
    ),
    (_csv_table, "Which region has the highest total revenue? Show the main numbers.", (20, 120)),
    (_story, "Summarize this story in three sentences.", (10, 70)),
    (
        _code_file,
        "What does step_3 return for x = 12? Then describe the file in one line.",
        (5, 40),
    ),
]


def build() -> list[dict]:
    rng = random.Random(1234)
    prompts: list[dict] = []

    def add(pid: str, messages: list[dict]) -> None:
        prompts.append({"id": pid, "messages": messages})

    def user(text: str) -> list[dict]:
        return [{"role": "user", "content": text}]

    # Keep the six original ids first, so earlier results stay comparable.
    add("chat-short", user(CHAT[0]))
    add("code", user(CODE[0]))
    add("math", user(MATH[0]))
    add("non-english", user(NON_ENGLISH[0]))
    add("emoji-utf8", user(EMOJI[0]))
    add("multi-turn", [{"role": r, "content": c} for r, c in MULTI_TURN[0]])

    for i, text in enumerate(CHAT[1:], 1):
        add(f"chat-{i:02d}", user(text))
    for i, text in enumerate(CODE[1:], 1):
        add(f"code-{i:02d}", user(text))
    for i, text in enumerate(MATH[1:], 1):
        add(f"math-{i:02d}", user(text))
    for i, text in enumerate(NON_ENGLISH[1:], 1):
        add(f"lang-{i:02d}", user(text))
    for i, text in enumerate(EMOJI[1:], 1):
        add(f"emoji-{i:02d}", user(text))
    for i, turns in enumerate(MULTI_TURN[1:], 1):
        add(f"multi-{i:02d}", [{"role": r, "content": c} for r, c in turns])
    for i, (system, text) in enumerate(SYSTEM, 1):
        add(f"system-{i:02d}", [{"role": "system", "content": system}, *user(text)])

    n_long = 200 - len(prompts)
    for i in range(n_long):
        make, question, (lo, hi) = LONG_KINDS[i % len(LONG_KINDS)]
        # Spread sizes evenly from short to the top of the range (about 2,000 tokens).
        size = lo + (hi - lo) * (i // len(LONG_KINDS)) // max(n_long // len(LONG_KINDS) - 1, 1)
        add(f"long-{i:02d}", user(f"{make(rng, size)}\n\n{question}"))
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="tokenizer dir; prints prompt-length stats")
    args = parser.parse_args()

    prompts = build()
    assert len(prompts) == 200, len(prompts)
    assert len({p["id"] for p in prompts}) == 200, "duplicate ids"
    OUT.write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts), encoding="utf-8"
    )
    print(f"wrote {len(prompts)} prompts to {OUT}")

    if args.model:
        from transformers import AutoTokenizer

        from engine.runtime.chat import chat_prompt_ids

        tok = AutoTokenizer.from_pretrained(args.model)
        lengths = sorted(len(chat_prompt_ids(tok, p["messages"])) for p in prompts)
        raw = sorted(
            len(tok.encode(m["content"], add_special_tokens=False))
            for p in prompts
            for m in p["messages"][-1:]
        )
        print(f"templated tokens: min {lengths[0]}, median {lengths[100]}, max {lengths[-1]}")
        print(f"last-message tokens: min {raw[0]}, median {raw[100]}, max {raw[-1]}")


if __name__ == "__main__":
    main()
