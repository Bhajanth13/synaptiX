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
    a.add_argument("--speak", action="store_true", help="also write speech audio (offline rule-based TTS)")
    a.add_argument("--voice", choices=["rule", "sarvam"], default="rule",
                   help="rule = hard-coded offline synthesizer (default); sarvam = Sarvam AI cloud voices")

    t = sub.add_parser("say", help="speak any text with the offline rule-based TTS")
    t.add_argument("text")
    t.add_argument("--lang", choices=LANGUAGES)
    t.add_argument("--phonemes", action="store_true", help="print the phonemes too")

    s = sub.add_parser("signs", help="answer from a sequence of ISL glosses")
    s.add_argument("glosses", nargs="+")

    lv = sub.add_parser("sign-live", help="sign questions to the webcam (needs a trained classifier)")
    lv.add_argument("--camera", type=int, default=0)
    lv.add_argument("--lang", choices=LANGUAGES, default="en")

    sub.add_parser("languages", help="list supported languages")

    args = p.parse_args(argv)
    if args.cmd == "languages":
        for code, info in LANGUAGES.items():
            voice = info["sarvam"] or f"{info['tts_fallback']} voice (fallback)"
            print(f"{code:4} {info['name']:10} offline TTS: yes   Sarvam: {voice}")
        return 0

    if args.cmd == "say":
        from synaptix.core.language import detect
        from synaptix.outputs.tts import RuleTTS
        tts, lang = RuleTTS(), args.lang or detect(args.text)
        if args.phonemes:
            print(" ".join(tts.phonemes(args.text, lang)))
        print("Audio:", tts.synthesize(args.text, lang))
        return 0

    from synaptix.core.pipeline import Pipeline
    pipe = Pipeline()
    if args.cmd == "sign-live":
        from synaptix.inputs.sign.live import recognize
        glosses = recognize(args.camera)
        print("Signs:", " ".join(glosses) or "(none recognised)")
        if not glosses:
            return 1
        ans = pipe.ask_signs(glosses, args.lang)
    elif args.cmd == "ask":
        ans = pipe.ask(args.text, args.lang, args.skill)
    else:
        ans = pipe.ask_signs(args.glosses)
    print(f"[{ans.skill}] " + ans.render())
    if getattr(args, "speak", False):
        from synaptix.outputs.tts import get_tts
        print("Audio:", get_tts(args.voice).synthesize(ans.text, ans.lang))
    return 0


if __name__ == "__main__":
    sys.exit(main())
