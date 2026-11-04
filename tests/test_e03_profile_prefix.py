from __future__ import annotations

import json

from src.evaluation.profile_prefix import ProfilePrefixStopper, parse_profile_prefix


ENUMS = {
    "emotion_label": ["平静", "开心"],
    "personality_traits": ["友善", "谨慎"],
    "interests": ["音乐", "阅读"],
    "style": ["温和", "直接"],
}


def _prefix() -> str:
    return (
        '{"emotion_label":"平静","user_profile":'
        '{"personality_traits":["友善"],"interests":[],"style":["温和"]}'
    )


def test_accepts_closed_profile_and_stops_before_long_response_json():
    response = '包含 {"key": "}"}、Unicode 🙂 和转义 \\"，继续。'
    full = _prefix() + ',"response_text":' + json.dumps(response, ensure_ascii=False) + ',"memory_refs":[]}'
    result = parse_profile_prefix(full, ENUMS)
    assert result.status == "valid"
    assert result.reason == "profile_prefix_closed"
    assert full[:result.boundary] == _prefix()
    assert result.emotion_label == "平静"
    assert result.user_profile == {
        "personality_traits": ["友善"], "interests": [], "style": ["温和"]
    }


def test_escaped_top_level_key_is_decoded_before_order_check():
    text = '{"emotion_\\u006cabel":"开心","user_profile":{"personality_traits":[],"interests":[],"style":[]}'
    assert parse_profile_prefix(text, ENUMS).valid


def test_reordered_fields_are_unsupported_and_never_accepted():
    result = parse_profile_prefix(
        '{"user_profile":{"personality_traits":[],"interests":[],"style":[]},"emotion_label":"平静"}',
        ENUMS,
    )
    assert result.status == "unsupported_order"


def test_duplicate_root_and_nested_keys_are_rejected():
    root = '{"emotion_label":"平静","emotion_label":"开心","user_profile":{"personality_traits":[],"interests":[],"style":[]}'
    nested = '{"emotion_label":"平静","user_profile":{"personality_traits":[],"personality_traits":[],"interests":[],"style":[]}'
    assert parse_profile_prefix(root, ENUMS).reason == "duplicate_top_level_key"
    nested_result = parse_profile_prefix(nested, ENUMS)
    assert nested_result.status == "invalid"
    assert "duplicate JSON key" in nested_result.reason


def test_arrays_wrong_dimensions_and_unknown_labels_are_rejected():
    assert parse_profile_prefix('["平静"]', ENUMS).status == "invalid"
    wrong_profile = '{"emotion_label":"平静","user_profile":[]}'
    assert parse_profile_prefix(wrong_profile, ENUMS).reason == "profile_not_object"
    unknown = '{"emotion_label":"平静","user_profile":{"personality_traits":["未知"],"interests":[],"style":[]}'
    assert parse_profile_prefix(unknown, ENUMS).reason == "unknown_profile_label:personality_traits"


def test_truncated_object_or_string_stays_incomplete():
    assert parse_profile_prefix('{"emotion_label":"平静",', ENUMS).status == "incomplete"
    assert parse_profile_prefix('{"emotion_label":"平静","user_profile":{"personality_traits":["友善', ENUMS).status == "incomplete"


def test_token_stopper_waits_across_token_boundaries_and_stores_exact_ids():
    text = _prefix() + ',"response_text":"large payload"}'
    stopper = ProfilePrefixStopper(lambda ids: "".join(chr(token) for token in ids), ENUMS)
    stop_at = None
    for index, character in enumerate(text, 1):
        if stopper.inspect([ord(item) for item in text[:index]]):
            stop_at = index
            break
    assert stop_at == len(_prefix())
    assert stopper.stop_reason == "profile_prefix_closed"
    assert stopper.generated_token_ids == tuple(ord(item) for item in _prefix())
    assert stopper.result is not None and stopper.result.valid


def test_eos_and_token_limit_are_explicitly_incomplete():
    stopper = ProfilePrefixStopper(lambda ids: "".join(chr(token) for token in ids), ENUMS)
    assert stopper.finish("eos", [ord("{")]).reason == "early_eos"
    assert stopper.finish("max_new_tokens", [ord("{")]).reason == "token_limit"
