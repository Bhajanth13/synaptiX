import argparse
import sys

from synaptix.core.language import LANGUAGES


def main(argv=None):
    p = argparse.ArgumentParser(prog="synaptix", description="Offline civic AI for Indian languages.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="ask a question")
    a.add_argument("text")
    a.add_argument("--lang", choices=LANGUAGES, help="language code (auto-detected from script if omitted)")
    a.add_argument("--skill", choices=["jargon", "legal", "jurisdiction", "events"])
    a.add_argument("--speak", action="store_true", help="also write speech audio (Sarvam TTS, cached)")

    s = sub.add_parser("signs", help="answer from a sequence of ISL glosses")
    s.add_argument("glosses", nargs="+")

    sub.add_parser("languages", help="list supported languages")

    args = p.parse_args(argv)
    if args.cmd == "languages":
        for code, info in LANGUAGES.items():
            voice = info["sarvam"] or f"{info['tts_fallback']} voice (fallback)"
            print(f"{code:4} {info['name']:10} TTS: {voice}")
        return 0

    from synaptix.core.pipeline import Pipeline
    pipe = Pipeline()
    ans = pipe.ask(args.text, args.lang, args.skill) if args.cmd == "ask" else pipe.ask_signs(args.glosses)
    print(f"[{ans.skill}] " + ans.render())
    if getattr(args, "speak", False):
        from synaptix.outputs.tts import SarvamTTS
        try:
            print("Audio:", SarvamTTS().synthesize(ans.text, ans.lang))
        except RuntimeError as e:
            print(e, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
