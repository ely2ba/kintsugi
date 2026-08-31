# Reused without item/checker changes from kintsugi-v1 d2aa7cd94a0a618169496f0235fa021ea46c0372.
"""Frozen deterministic instruction-following suite for Kintsugi."""
import hashlib
import json
import math
import re
import unicodedata
from protocol import IF_MAX_TOKENS, IF_RECOVERY_FRACTION
CATEGORIES = ("json", "length", "forbidden", "multilingual")
def _item(item_id, split, category, prompt, checker, expected=None, **rules):
    item = {"id": item_id, "split": split, "category": category, "prompt": prompt, "checker": checker, "max_tokens": IF_MAX_TOKENS}
    if expected is not None: item["expected"] = expected
    if rules: item["rules"] = rules
    return item
def _json_rows(split, rows): return tuple(_item(item_id, split, "json", prompt, "json", expected) for item_id, prompt, expected in rows)
def _exact_rows(split, category, rows): return tuple(_item(item_id, split, category, prompt, "exact", expected) for item_id, prompt, expected in rows)
JSON_CRITERION = _json_rows("criterion", (
    ("c-json-01", "Return only JSON with exactly the keys animal and count, and no others. Animal is otter; count is 3.", {"animal": "otter", "count": 3}),
    ("c-json-02", "Return only JSON with exactly the keys city and coastal, and no others. City is Lima; coastal is true.", {"city": "Lima", "coastal": True}),
    ("c-json-03", "Return only a JSON array containing north, east, south in that order.", ["north", "east", "south"]),
    ("c-json-04", "Return only JSON whose sole key is colors, mapped to an array with amber then teal.", {"colors": ["amber", "teal"]}),
    ("c-json-05", "Return only JSON with exactly the keys name and scores, and no others. Name is Ada; scores is an array containing 8, 13, 21.", {"name": "Ada", "scores": [8, 13, 21]}),
    ("c-json-06", "Return only JSON whose sole outer key is book; book has exactly title Dune and pages 412.", {"book": {"title": "Dune", "pages": 412}}),
    ("c-json-07", "Return only JSON with exactly ready false, note null, and retries 0, with no other keys.", {"ready": False, "note": None, "retries": 0}),
    ("c-json-08", "Return only a JSON array of two objects: id 1 has tag A; id 2 has tag B.", [{"id": 1, "tag": "A"}, {"id": 2, "tag": "B"}]),
    ("c-json-09", "Return only JSON with exactly x 4, y -2, and unit cm, with no other keys.", {"x": 4, "y": -2, "unit": "cm"}),
    ("c-json-10", "Return only JSON whose sole outer key is user; user has exactly id 7 and active true.", {"user": {"id": 7, "active": True}}),
    ("c-json-11", "Return only JSON with exactly the keys weekdays and total. Weekdays is an array containing Monday then Friday; total is 2.", {"weekdays": ["Monday", "Friday"], "total": 2}),
    ("c-json-12", "Return only JSON with exactly code X9 and valid false, with no other keys.", {"code": "X9", "valid": False}),
    ("c-json-13", "Return only a JSON object whose only key is empty and whose value is an empty array.", {"empty": []}),
    ("c-json-14", "Return only JSON with exactly temperature 18 and scale C, with no other keys.", {"temperature": 18, "scale": "C"}),
    ("c-json-15", "Return only JSON whose sole outer key is route; route has exactly from Oslo, to Bergen, and stops 1.", {"route": {"from": "Oslo", "to": "Bergen", "stops": 1}}),
))
JSON_HELDOUT = _json_rows("heldout", (
    ("h-json-01", "Return only JSON with exactly the keys bird and count, and no others. Bird is heron; count is 2.", {"bird": "heron", "count": 2}),
    ("h-json-02", "Return only a JSON array containing copper, silver, gold in that order.", ["copper", "silver", "gold"]),
    ("h-json-03", "Return only JSON whose sole outer key is station; station has exactly name Oak and open false.", {"station": {"name": "Oak", "open": False}}),
    ("h-json-04", "Return only JSON with exactly value null and measured true, with no other keys.", {"value": None, "measured": True}),
    ("h-json-05", "Return only JSON whose sole key is primes, mapped to an array containing 2, 3, 5.", {"primes": [2, 3, 5]}),
    ("h-json-06", "Return only a JSON array of two objects: slot A has full true; slot B has full false.", [{"slot": "A", "full": True}, {"slot": "B", "full": False}]),
    ("h-json-07", "Return only JSON with exactly width 6, height 9, and unit px, with no other keys.", {"width": 6, "height": 9, "unit": "px"}),
    ("h-json-08", "Return only JSON whose sole outer key is trip; trip has exactly origin Kyoto and nights 4.", {"trip": {"origin": "Kyoto", "nights": 4}}),
))
LENGTH_CRITERION = _exact_rows("criterion", "length", (
    ("c-length-01", "Reply with exactly three words: blue river stone", "blue river stone"),
    ("c-length-02", "Reply with exactly four words: quiet lamps glow nightly", "quiet lamps glow nightly"),
    ("c-length-03", "Reply with exactly two words: winter pears", "winter pears"),
    ("c-length-04", "Reply with exactly five words: small birds cross the valley", "small birds cross the valley"),
    ("c-length-05", "Reply with exactly six words: one two three four five six", "one two three four five six"),
    ("c-length-06", "Reply with exactly two lines. First line: Name: Ada. Second line: Role: analyst.", "Name: Ada\nRole: analyst"),
    ("c-length-07", "Reply with exactly three lines: red on line one, green on line two, blue on line three.", "red\ngreen\nblue"),
    ("c-length-08", "Reply with exactly two lines. First: Status: ready. Second: Code: 17.", "Status: ready\nCode: 17"),
    ("c-length-09", "Reply as exactly three numbered lines: 1. oak, 2. elm, 3. ash.", "1. oak\n2. elm\n3. ash"),
    ("c-length-10", "Reply with exactly two lines: BEGIN then END.", "BEGIN\nEND"),
    ("c-length-11", "Reply with exactly five characters: KITE!", "KITE!"),
    ("c-length-12", "Reply with exactly six characters: orbit?", "orbit?"),
    ("c-length-13", "Reply with exactly four characters: 7x7=", "7x7="),
    ("c-length-14", "Reply with exactly seven characters: SUN-204", "SUN-204"),
    ("c-length-15", "Reply with exactly eight characters: mint_tea", "mint_tea"),
))
LENGTH_HELDOUT = _exact_rows("heldout", "length", (
    ("h-length-01", "Reply with exactly three words: silver clouds gather", "silver clouds gather"),
    ("h-length-02", "Reply with exactly five words: seven cranes circle at dawn", "seven cranes circle at dawn"),
    ("h-length-03", "Reply with exactly two words: paper lantern", "paper lantern"),
    ("h-length-04", "Reply with exactly two lines. First: Item: compass. Second: Qty: 2.", "Item: compass\nQty: 2"),
    ("h-length-05", "Reply as exactly three labeled lines: A: sun, B: moon, C: star.", "A: sun\nB: moon\nC: star"),
    ("h-length-06", "Reply with exactly two lines: OPEN then CLOSED.", "OPEN\nCLOSED"),
    ("h-length-07", "Reply with exactly five characters: WAVE?", "WAVE?"),
    ("h-length-08", "Reply with exactly seven characters: map-404", "map-404"),
))
FORBIDDEN_CRITERION = _exact_rows("criterion", "forbidden", (
    ("c-forbidden-01", "Replace red with blue in 'the red kite'; output only the result.", "the blue kite"),
    ("c-forbidden-02", "Replace slow with swift in 'a slow river'; output only the result.", "a swift river"),
    ("c-forbidden-03", "Replace every cat with fox in 'cat and cat'; output only the result.", "fox and fox"),
    ("c-forbidden-04", "Remove the word very from 'a very calm lake'; output only the result.", "a calm lake"),
    ("c-forbidden-05", "Change Monday to Tuesday in 'Meet on Monday'; output only the result.", "Meet on Tuesday"),
    ("c-forbidden-11", "Choose cedar, not pine. Reply with that one word only.", "cedar"),
    ("c-forbidden-12", "Choose circle, not square. Reply with that one word only.", "circle"),
    ("c-forbidden-13", "Reply exactly YES; do not write no.", "YES"),
    ("c-forbidden-14", "Name the allowed color only: amber. Do not use violet.", "amber"),
    ("c-forbidden-15", "Return the permitted code only: R7. Do not return Q4.", "R7"),
)) + tuple(
    _item(item_id, "criterion", "forbidden", prompt, "tokens", required=required, forbidden=forbidden, max_words=max_words)
    for item_id, prompt, required, forbidden, max_words in (
        ("c-forbidden-06", "In at most six words, mention cedar and rain; do not use wet.", ["cedar", "rain"], ["wet"], 6),
        ("c-forbidden-07", "In at most five words, include moon and quiet; never use night.", ["moon", "quiet"], ["night"], 5),
        ("c-forbidden-08", "Write at most seven words containing tea and warm, without hot.", ["tea", "warm"], ["hot"], 7),
        ("c-forbidden-09", "In at most six words, include train and early; omit late.", ["train", "early"], ["late"], 6),
        ("c-forbidden-10", "Write at most five words containing kind and firm; avoid rude.", ["kind", "firm"], ["rude"], 5),
    )
)
FORBIDDEN_HELDOUT = _exact_rows("heldout", "forbidden", (
    ("h-forbidden-01", "Replace cold with mild in 'a cold morning'; output only the result.", "a mild morning"),
    ("h-forbidden-02", "Remove the word quite from 'a quite narrow road'; output only the result.", "a narrow road"),
    ("h-forbidden-06", "Choose granite, not marble. Reply with that one word only.", "granite"),
    ("h-forbidden-07", "Return the permitted code only: T2. Do not return P8.", "T2"),
)) + tuple(
    _item(item_id, "heldout", "forbidden", prompt, "tokens", required=required, forbidden=forbidden, max_words=max_words)
    for item_id, prompt, required, forbidden, max_words in (
        ("h-forbidden-03", "In at most six words, mention willow and breeze; do not use wind.", ["willow", "breeze"], ["wind"], 6),
        ("h-forbidden-04", "Write at most five words containing gentle and clear, without harsh.", ["gentle", "clear"], ["harsh"], 5),
        ("h-forbidden-05", "In at most seven words, include bread and fresh; omit stale.", ["bread", "fresh"], ["stale"], 7),
    )
)
MULTILINGUAL_CRITERION = _exact_rows("criterion", "multilingual", (
    ("c-multi-01", "Translate 'good morning' into French. Reply with only the translation in lowercase, no punctuation.", "bonjour"),
    ("c-multi-02", "Translate 'thank you' into Spanish. Reply with only the translation in lowercase, no punctuation.", "gracias"),
    ("c-multi-03", "Translate 'water' into German. Reply with only the German noun, capitalized, no punctuation.", "Wasser"),
    ("c-multi-04", "Translate 'book' into Italian. Reply with only the translation in lowercase, no punctuation.", "libro"),
    ("c-multi-05", "Translate 'friend' into Portuguese. Reply with only the masculine singular translation, lowercase.", "amigo"),
    ("c-multi-06", "Reply in exactly two lines: English 'yes' on line one, French 'yes' on line two.", "yes\noui"),
    ("c-multi-07", "Reply in exactly two lines: Spanish 'hello' on line one, Italian 'hello' on line two.", "hola\nciao"),
    ("c-multi-08", "Reply in exactly two lines: German 'no' on line one, Dutch 'no' on line two.", "nein\nnee"),
    ("c-multi-09", "Reply in exactly two lines: English 'moon' on line one, Spanish 'moon' on line two.", "moon\nluna"),
    ("c-multi-10", "Reply in exactly two lines: French 'red' on line one, German 'red' on line two.", "rouge\nrot"),
    ("c-multi-11", "Reply with only the Japanese word for cat in Japanese script.", "猫"),
    ("c-multi-12", "Reply with only the Arabic word for book in Arabic script.", "كتاب"),
    ("c-multi-13", "Reply with only the Korean word for water in Hangul.", "물"),
    ("c-multi-14", "Reply with only the Greek word for yes in Greek script, lowercase.", "ναι"),
    ("c-multi-15", "Reply with only the Hindi word for house in Devanagari.", "घर"),
))
MULTILINGUAL_HELDOUT = _exact_rows("heldout", "multilingual", (
    ("h-multi-01", "Translate 'night' into Spanish. Reply with only the translation in lowercase, no punctuation.", "noche"),
    ("h-multi-02", "Translate 'apple' into French. Reply with only the translation in lowercase, no punctuation.", "pomme"),
    ("h-multi-03", "Translate 'sun' into Italian. Reply with only the translation in lowercase, no punctuation.", "sole"),
    ("h-multi-04", "Reply in exactly two lines: English 'blue' on line one, French 'blue' on line two.", "blue\nbleu"),
    ("h-multi-05", "Reply in exactly two lines: Spanish 'one' on line one, German 'one' on line two.", "uno\neins"),
    ("h-multi-06", "Reply with only the Japanese word for mountain in Japanese script.", "山"),
    ("h-multi-07", "Reply with only the Arabic word for moon in Arabic script.", "قمر"),
))
ITEMS = (
    JSON_CRITERION + LENGTH_CRITERION + FORBIDDEN_CRITERION + MULTILINGUAL_CRITERION
    + JSON_HELDOUT + LENGTH_HELDOUT + FORBIDDEN_HELDOUT + MULTILINGUAL_HELDOUT
)
def items(split=None, category=None):
    """Return suite items selected by split and/or category."""
    return tuple(item for item in ITEMS if (split is None or item["split"] == split) and (category is None or item["category"] == category))
def _text(value, strip=True):
    value = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    return value.strip() if strip else value
def check_output(item, output):
    """Apply one deterministic checker; malformed or non-string output fails."""
    if not isinstance(output, str): return False
    if item["checker"] == "exact": return _text(output, strip=False) == item["expected"]
    if item["checker"] == "json":
        try:
            def unique_object(pairs):
                if len({key for key, _ in pairs}) != len(pairs): raise ValueError("duplicate JSON key")
                return dict(pairs)
            parsed = json.loads(_text(output), object_pairs_hook=unique_object)
        except (json.JSONDecodeError, TypeError, ValueError): return False
        canonical = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return canonical(parsed) == canonical(item["expected"])
    if item["checker"] == "tokens":
        words = re.findall(r"[^\W_]+", _text(output).casefold(), flags=re.UNICODE); rules = item["rules"]
        return len(words) <= rules["max_words"] and all(word.casefold() in words for word in rules["required"]) and not any(word.casefold() in words for word in rules["forbidden"])
    raise ValueError(f"unknown checker: {item['checker']}")
def evaluate(outputs, split="criterion"):
    """Score an id->output mapping. Held-out results never enter criterion scoring."""
    selected = items(split=split)
    results = {item["id"]: check_output(item, outputs.get(item["id"], "")) for item in selected}
    passed = sum(results.values())
    return {"split": split, "passed": passed, "total": len(selected), "score": passed / len(selected), "results": results}
def criterion_target(cycle0_passes):
    if not 0 <= cycle0_passes <= len(items(split="criterion")): raise ValueError("cycle0_passes must be between 0 and 60")
    return math.ceil(IF_RECOVERY_FRACTION * cycle0_passes)
def prompt_hash(prompt): return hashlib.sha256(_text(prompt).encode("utf-8")).hexdigest()
def prompt_hashes(prompts):
    """Hash strings or dictionaries containing a string `prompt` field."""
    texts = (value if isinstance(value, str) else value["prompt"] for value in prompts)
    return frozenset(prompt_hash(value) for value in texts)
def assert_prompt_hash_disjoint(repair_prompts, suite_items=ITEMS):
    """Raise if any normalized repair-pool prompt duplicates an IF-suite prompt."""
    overlap = prompt_hashes(repair_prompts) & prompt_hashes(suite_items)
    if overlap: raise ValueError(f"repair pool overlaps IF suite at {len(overlap)} prompt hash(es)")
    return True
def manifest_hash(split=None):
    payload = json.dumps(items(split=split), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
def manifest_hashes(): return {"all": manifest_hash(), "criterion": manifest_hash("criterion"), "heldout": manifest_hash("heldout")}
def _validate_manifest():
    expected = {"criterion": {category: 15 for category in CATEGORIES}, "heldout": {"json": 8, "length": 8, "forbidden": 7, "multilingual": 7}}
    if len(ITEMS) != 90 or len({item["id"] for item in ITEMS}) != 90: raise ValueError("IF suite must contain 90 uniquely identified items")
    for split, counts in expected.items():
        actual = {category: len(items(split, category)) for category in CATEGORIES}
        if actual != counts: raise ValueError(f"bad {split} category counts: {actual}")
    if any(item["max_tokens"] != IF_MAX_TOKENS for item in ITEMS): raise ValueError("all IF items must use max_tokens=96")
    if len(prompt_hashes(ITEMS)) != len(ITEMS): raise ValueError("IF prompts must be unique after normalization")
_validate_manifest()
