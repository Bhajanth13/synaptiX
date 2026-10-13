# Responsible AI

- **Information, not advice.** SynaptiX explains what a law or document says. Disputes and
  high-stakes cases are routed to a named human: DLSA (free legal aid, helpline 15100), the labour
  office, the Protection Officer, or 181 / 112 in emergencies.
- **Political neutrality.** It explains terms, institutions and processes. It never rates, compares
  or gives opinions on parties, politicians or policies. The SLM system prompt enforces this.
- **No invented facts.** Every answer cites a source from `data/`. Anything the model paraphrases
  without a verified source is labelled unverified.
- **Human verification.** Every KB entry carries `review: pending` until a law student (content) and a
  native speaker (translation) sign it off.
- **Privacy.** Queries are processed locally. Sign input stores keypoints, never video. There are no network
  calls by default; the optional Sarvam voice sends only the answer text, never the user's question.
- **Known limits.** Devanagari text defaults to Hindi; Marathi and Marwari need an explicit language.
  Which panchayat subjects are devolved differs by state.
