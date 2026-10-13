# SynaptiX overview

## The problem

People in rural India receive land notices, loan agreements, court summons and scheme rules written
in official language. They often don't know what a paper means, what they are entitled to, or whom
to approach: the Sarpanch, the MLA or the MP. Getting a lawyer costs money and a trip to town, and
internet access is unreliable.

## What SynaptiX does

| Ask | SynaptiX answers with |
|---|---|
| "What is MPLADS?" / "ग्राम सभा क्या है?" | A plain-language explanation from a verified glossary |
| "Who fixes the village road?" | Who to go to, in order, and who *can't* help, based on the Constitution |
| "My contractor didn't pay minimum wage" | What the law says, what to do, and where to get free help |
| "When is the next Gram Sabha?" | Upcoming civic events |

Questions can be typed, spoken, or signed in Indian Sign Language, and answers can be read or heard.

## Principles

- **Trust over cleverness:** answers come from verified, sourced information. The AI only rewrites it
  in simpler words and says "I don't know" rather than guess.
- **Works anywhere:** runs offline on an ordinary laptop's CPU, including speech.
- **Many languages:** Hindi, English, Kannada, Marwari, Bengali, Marathi, Telugu, Tamil, Gujarati,
  Malayalam, Odia and Punjabi.
- **Information, not advice:** serious cases are referred to free legal aid (DLSA) or the right office.
- **Neutral:** explains how government works, never takes political sides.

## How it works, briefly

1. Detect the language and understand what kind of question it is.
2. Look up the answer in the verified knowledge base (`data/`).
3. Optionally simplify the wording with a small offline language model.
4. Reply in the user's language as text, and as speech from a built-in rule-based voice.

Technical details: [ARCHITECTURE.md](ARCHITECTURE.md). Safeguards: [RESPONSIBLE_AI.md](RESPONSIBLE_AI.md).
Notes for AI coding agents: [../AGENTS.md](../AGENTS.md).

## Status

Early prototype built for the Mozilla–CHRIST Human Scale AI Buildathon 2026. Knowledge-base entries
are awaiting expert and native-speaker review.
