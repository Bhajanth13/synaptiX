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

    a.add_argument("--district", help="your district (e.g. Mysuru); names the local offices to approach")
    a.add_argument("--state", help="your state (e.g. karnataka)")

    t = sub.add_parser("say", help="speak any text with the offline rule-based TTS")
    t.add_argument("text")
    t.add_argument("--lang", choices=LANGUAGES)
    t.add_argument("--phonemes", action="store_true", help="print the phonemes too")

    s = sub.add_parser("signs", help="answer from a sequence of ISL glosses")
    s.add_argument("glosses", nargs="+")

    lv = sub.add_parser("sign-live", help="sign questions to the webcam (needs a trained classifier)")
    lv.add_argument("--camera", type=int, default=0)
    lv.add_argument("--lang", choices=LANGUAGES, default="en")
    lv.add_argument("--model", choices=["include", "custom"], default="include",
                    help="include = pretrained 263-sign ISL model (default); custom = your trained classifier")

    sp = sub.add_parser("spell-live", help="fingerspell letters/digits to the webcam, then ask the spelled text")
    sp.add_argument("--camera", type=int, default=0)
    sp.add_argument("--lang", choices=LANGUAGES, default="en")

    si = sub.add_parser("spell-image", help="read fingerspelled letters/digits from image files")
    si.add_argument("images", nargs="+")

    sf = sub.add_parser("sign-file", help="recognise ISL signs in a video file")
    sf.add_argument("video")
    sf.add_argument("--model", choices=["include", "custom"], default="include")
    sf.add_argument("--ask", action="store_true", help="send the recognised signs to the pipeline")
    sf.add_argument("--whole", action="store_true", help="treat the whole clip as a single sign")

    sub.add_parser("languages", help="list supported languages")

    d = sub.add_parser("districts", help="list the districts the knowledge base knows")
    d.add_argument("--state", default="karnataka")

    lc = sub.add_parser("location", help="show, save or clear your default district")
    lc.add_argument("name", nargs="?")
    lc.add_argument("--clear", action="store_true")

    pr = sub.add_parser("prerender", help="rewrite verified entries with the SLM and store the passing ones")
    pr.add_argument("--force", action="store_true", help="redo entries that already have a rewrite")
    pr.add_argument("--out", help="output YAML (default data/rewrites/en.yaml)")

    ev = sub.add_parser("eval-rewrites", help="score SLM rewrites of the knowledge base with the guard")
    ev.add_argument("--prerendered", action="store_true", help="score only stored rewrites (no model needed)")
    ev.add_argument("--json", action="store_true")

    args = p.parse_args(argv)
    if args.cmd == "languages":
        for code, info in LANGUAGES.items():
            voice = info["sarvam"] or f"{info['tts_fallback']} voice (fallback)"
            print(f"{code:4} {info['name']:10} offline TTS: yes   Sarvam: {voice}")
        return 0

    if args.cmd in ("districts", "location", "prerender", "eval-rewrites") or (
            args.cmd == "ask" and (args.district or args.state)):
        rc = _location_commands(args)
        if rc is not None:
            return rc

    if args.cmd == "say":
        from synaptix.core.language import detect
        from synaptix.outputs.tts import RuleTTS
        tts, lang = RuleTTS(), args.lang or detect(args.text)
        if args.phonemes:
            print(" ".join(tts.phonemes(args.text, lang)))
        print("Audio:", tts.synthesize(args.text, lang))
        return 0

    if args.cmd == "spell-image":
        import cv2
        from synaptix.inputs.sign.fingerspell import FingerspellModel, HandDetector
        model, det = FingerspellModel(), HandDetector()
        for path in args.images:
            img = cv2.imread(path)
            if img is not None and max(img.shape[:2]) < 256:
                f = 256 / max(img.shape[:2])
                img = cv2.resize(img, None, fx=f, fy=f)
            letter, conf = model.predict(det.hands(img) if img is not None else [], min_conf=0.0)
            print(f"{path}: {letter or '(no hand found)'} {conf:.2f}")
        return 0

    if args.cmd == "sign-file":
        from synaptix.inputs.sign.live import recognize_file
        preds = recognize_file(args.video, args.model, args.whole)
        for g, c in preds:
            print(f"{g:15} {c:.2f}")
        if not (args.ask and preds):
            return 0 if preds else 1

    from synaptix.core.pipeline import Pipeline
    pipe = Pipeline()
    if args.cmd == "spell-live":
        from synaptix.inputs.sign.fingerspell import spell_live
        text = spell_live(args.camera)
        print("Spelled:", text or "(nothing)")
        if not text:
            return 1
        ans = pipe.ask(text.lower(), args.lang)
    elif args.cmd == "sign-live":
        from synaptix.inputs.sign.live import recognize
        glosses = recognize(args.camera, model=args.model)
        print("Signs:", " ".join(glosses) or "(none recognised)")
        if not glosses:
            return 1
        ans = pipe.ask_signs(glosses, args.lang)
    elif args.cmd == "sign-file":
        ans = pipe.ask_signs([g for g, c in preds if g and c >= 0.3] or [preds[0][0]])
    elif args.cmd == "ask":
        ans = pipe.ask(args.text, args.lang, args.skill, district=args.district, state=args.state)
    else:
        ans = pipe.ask_signs(args.glosses)
    print(f"[{ans.skill}] " + ans.render())
    if getattr(args, "speak", False):
        from synaptix.outputs.tts import get_tts
        print("Audio:", get_tts(args.voice).synthesize(ans.text, ans.lang))
    return 0


def _location_commands(args):
    """Commands that need only the knowledge base. Returns an exit code, or None to continue."""
    from synaptix.core import location as loc
    from synaptix.kb.store import KnowledgeBase
    kb = KnowledgeBase()

    def resolve(name=None, district=None, state=None):
        try:
            return loc.resolve(kb, district=district, state=state)
        except loc.UnknownLocation as e:
            hint = f" Did you mean: {', '.join(e.suggestions)}?" if e.suggestions else ""
            print(f"Unknown {'district' if district else 'state'} '{e.name}'.{hint} Run 'synaptix districts'.",
                  file=sys.stderr)
            return None

    if args.cmd == "ask":
        if args.district and resolve(district=args.district, state=args.state) is None:
            return 2
        if args.state and not args.district and resolve(state=args.state) is None:
            return 2
        return None
    if args.cmd == "districts":
        sid = kb.find_state(args.state)
        if sid is None:
            print(f"Unknown state '{args.state}'.", file=sys.stderr)
            return 2
        ds = sorted(kb.states[sid]["districts"], key=lambda d: d["id"])
        for d in ds:
            print(f"{d['id']:18} {d['name']:18} {d.get('name_kn', '')}")
        print(f"{len(ds)} districts")
        return 0
    if args.cmd == "location":
        if args.clear:
            loc.clear_config()
            print("Cleared.")
            return 0
        if not args.name:
            cur = loc.load_config(kb)
            print(f"Current: {cur.label}, {kb.states[cur.state]['meta']['name']}" if cur.district
                  else (f"Current: {cur.label}" if cur else "Current: none"))
            print(f"Config file: {loc.config_path()}")
            return 0
        try:
            new = loc.resolve(kb, district=args.name)
        except loc.UnknownLocation:
            new = resolve(state=args.name)
            if new is None:
                return 2
        loc.save_config(new)
        print(f"Saved: {new.label}" + (f", {kb.states[new.state]['meta']['name']}" if new.district else ""))
        return 0
    from synaptix.llm import evaluate, rewrites
    from synaptix.llm.slm import SLM
    if args.cmd == "prerender":
        slm = SLM()
        if not slm.available:
            print("No SLM installed (models/slm.gguf); nothing to prerender.")
            return 1
        written, rejected = evaluate.prerender(kb, slm, args.out or rewrites.DEFAULT_PATH, args.force)
        print(f"{written} written, {len(rejected)} rejected")
        for ref, reasons in rejected:
            print(f"  {ref}: {', '.join(reasons)}")
        return 0
    return evaluate.main((["--prerendered"] if args.prerendered else []) + (["--json"] if args.json else []))


if __name__ == "__main__":
    sys.exit(main())
