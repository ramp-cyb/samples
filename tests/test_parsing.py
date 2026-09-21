from homesim.agents.parsing import parse_assistant_text


def test_parses_qwen_tool_call_block():
    text = 'Looking it up.\n<tool_call>\n{"name": "search_context", "arguments": {"query": "movie"}}\n</tool_call>'
    m = parse_assistant_text(text)
    assert m["content"] == "Looking it up."
    assert m["tool_calls"][0]["function"] == {"name": "search_context", "arguments": {"query": "movie"}}


def test_parses_multiple_calls_and_strips_think():
    text = ('<think>hmm</think><tool_call>{"name":"set_device","arguments":{"device_id":"a.b","attribute":"power","value":"on"}}</tool_call>'
            '<tool_call>{"name":"done","arguments":{"message":"ok"}}</tool_call>')
    m = parse_assistant_text(text)
    assert [c["function"]["name"] for c in m["tool_calls"]] == ["set_device", "done"]
    assert m["content"] == ""


def test_fallback_bare_json_and_string_arguments():
    m = parse_assistant_text('Sure: {"name": "done", "arguments": "{\\"message\\": \\"bye\\"}"}')
    assert m["tool_calls"][0]["function"] == {"name": "done", "arguments": {"message": "bye"}}


def test_plain_text_has_no_calls():
    m = parse_assistant_text("I will turn the lights off now.")
    assert m["tool_calls"] == [] and m["content"].startswith("I will")
