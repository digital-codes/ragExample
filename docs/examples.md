# LLAMACPP

## Phi-4-mini-reasoning

### Description

Primary Use Cases

Phi-4-mini-reasoning is designed for multi-step, logic-intensive mathematical problem-solving tasks under memory/compute constrained environments and latency bound scenarios. Some of the use cases include formal proof generation, symbolic computation, advanced word problems, and a wide range of mathematical reasoning scenarios. These models excel at maintaining context across steps, applying structured logic, and delivering accurate, reliable solutions in domains that require deep analytical thinking.

**llamacpp running with normal template (no --ninja)**

 * 1 Simple question, little thinking

> http -j post localhost:8080/v1/chat/completions model=Phi-4-mini-reasoning   messages:='[{"role":"control","content":"thinking"},{"role":"user","content":"need best color. use the following information to answer:\nblue is bad\ngreen is best\n"}]' 
    HTTP/1.1 200 OK

``` 
{
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "<think>\nOkay, let's see. The user wants the \"best color\" based on the given information. The problem states that blue is bad and green is best. Hmm, so first, I need to understand what criteria they're using here. Maybe it's a straightforward comparison between the two colors.\n\nWait, the user says green is best. But why? Maybe there's a rule or a preference here. Let me break it down. The information given is two facts: blue is bad, and green is best. So if we take these at face value, the best color would be green because it's the one that's labeled as \"best\" directly.\n\nBut wait, could there be more to it? Like, maybe there's a hidden context. For example, sometimes in decision-making, you have to consider all options. If blue is bad, but maybe there are other colors not mentioned here. But the problem only provides information about blue and green. So according to the given data, green is the best because the statement says so. There's no conflicting information. So the answer should be green. Let me make sure I'm not overcomplicating it. The problem is straightforward. The user gives two pieces of information: blue is bad, green is best. So the best color available per the info is green. Yeah, that makes sense. No need to think beyond that. So the answer is green.\n</think>\n\nThe problem states that \"green is best\" directly. Since no additional context or constraints are provided, the best color is explicitly given. \n\n**Answer:**  \n\\boxed{green}",
                "role": "assistant"
            }
        }
    ],
    "created": 1746471181,
    "id": "chatcmpl-NljJQaocpSbFvJN6ZxHQ9XeNpagru5Da",
    "model": "Phi-4-mini-reasoning",
    "object": "chat.completion",
    "system_fingerprint": "b0-unknown",
    "timings": {
        "predicted_ms": 23867.456000000002,
        "predicted_n": 321,
        "predicted_per_second": 13.44927586752438,
        "predicted_per_token_ms": 74.35344548286605,
        "prompt_ms": 545.2280000000001,
        "prompt_n": 20,
        "prompt_per_second": 36.68190188324884,
        "prompt_per_token_ms": 27.261400000000002
    },
    "usage": {
        "completion_tokens": 321,
        "prompt_tokens": 33,
        "total_tokens": 354
    }
}

```

 * 2 Complicated question, much thinking

>  http -j post localhost:8080/v1/chat/completions model=Phi-4-mini-reasoning  messages:='[{"role":"control","content":"thinking,citations"},{"role":"user","content":"need best color. use only information from following context"},{"role":"user","content":"blue is bad\nred id good\nyellow rules"}]'

{
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "<think>\nOkay, so I need to figure out the best color based on these four lines: \"blue is bad,\" \"red id good,\" and \"yellow rules.\" Wait, the third one says \"yellow rules,\" but maybe there's a typo? Because \"id good\" doesn't make sense grammatically. Maybe \"red is good\"? That would make more sense. Let me assume that's the case. So the clues are:\n\n1. Blue is bad\n2. Red is good\n3. Yellow rules\n\nNow, I have to figure out the best color here. Let me think. These could be clues pointing to something about colors. Maybe it's a riddle where each line gives a part of the answer.\n\nFirst, \"blue is bad.\" Maybe blue is associated with negative things. Then \"red is good,\" so red is positive. \"Yellow rules,\" which could mean yellow is dominant or important. Alternatively, \"yellow rules\" might be a play on words. Maybe \"rules\" as in governs or controls.\n\nAlternatively, maybe it's a reference to traffic lights. Because in traffic lights, the order is red, yellow, green. But here we have red, yellow, and maybe green implied by \"rules\"? Wait, the first line says blue is bad. If we consider that in some contexts, blue might be associated with stopping, like the stop signal in traffic is red and green, but yellow is caution. Hmm, not sure.\n\nWait, the riddle says \"best color.\" So perhaps combining these clues. If red is good, yellow is next, and blue is bad. But in traffic lights, after red comes green, but here it's red, yellow, and maybe green isn't mentioned. Alternatively, if yellow rules, maybe yellow is the answer. But why?\n\nAlternatively, think of the colors of the Chinese political party flags. The Communist Party uses red, the Democratic Party uses blue, and the Green Party uses yellow. But the clues here are blue is bad, red is good, yellow rules. So maybe red is good, yellow is next in line, so the best color would be red? But the question is asking for the best color based on these clues.\n\nAlternatively, think of the colors in terms of danger. Red is bad (like in speed limits), but wait, no. Wait, \"blue is bad\" could mean avoiding blue, \"red is good\" so red is preferred, and \"yellow rules\" meaning yellow is the most important. So the best color would be yellow? But why?\n\nAlternatively, maybe it's a play on the phrases. For example, \"red id good\" could be a typo for \"red is good,\" and \"yellow rules\" could be a play on words. Maybe \"rules\" as in \"reuleaux,\" which is a color term for a gradient of a single color. But that might be a stretch.\n\nAlternatively, think of the colors in terms of the rainbow. Red, blue, yellow are all rainbow colors. But how does that help?\n\nWait, another angle: \"best color\" could be referring to the flag of a country. For example, if red is good, yellow rules, then maybe a country that has red and yellow as main colors. Like Japan's flag has red and white, but that's not yellow. Brazil's flag has yellow, blue, green. Maybe not.\n\nAlternatively, think of the phrase \"red is good, yellow is bad.\" But the first line says blue is bad. Hmm. Wait, in some contexts, yellow is considered bad in certain situations. For example, in some cultures, white is considered bad for weddings, but that's different.\n\nWait, maybe it's a riddle where each line gives a part of the answer. For example, \"blue is bad\" (avoid blue), \"red is good,\" \"yellow rules.\" If you take the first letters: B, R, Y. But not sure. Alternatively, take the colors in order. The best color would be the one that's good, so red. But then why mention yellow rules?\n\nAlternatively, think of the order of the clues. If red is good, yellow rules, then yellow is next in line after red, so the best color would be yellow. But in traffic lights, after red comes green, but maybe in this context, yellow is the answer.\n\nAlternatively, think of the terms as parts of a sentence. Like \"Red is good, yellow rules,\" so the best color is yellow. But the first line says blue is bad, so maybe exclude blue. So the best color would be red or yellow. But the second line says red is good, third says yellow rules. So maybe yellow is the best.\n\nAlternatively, think of the word \"rules.\" If yellow rules, maybe it's the dominant color. So if red is good and yellow is dominant, then the best color is yellow. But I'm not sure.\n\nWait, another approach: maybe these are clues about the color hierarchy. If red is good, yellow rules (is primary or main), and blue is bad (secondary or tertiary). But in color theory, primary colors are red, blue, and yellow. So if yellow is primary, maybe it's the best.\n\nAlternatively, the phrases could be hints towards the order. \"Red is good\" (so red comes first), \"yellow rules\" (yellow is next), so the best color would be yellow. But then why mention blue is bad? Maybe blue is excluded, so the answer is yellow.\n\nAlternatively, the answer is red because it's mentioned as good. But the third clue complicates that.\n\nAlternatively, think of the word \"rules\" as \"reuleaux,\" which is a gradient of a single color, but that's a stretch.\n\nWait, maybe it's a play on the words \"red id good\" which might be \"red is good,\" \"yellow rules\" which could be \"yellow is right.\" So combining, the best color would be red or yellow. But without more context, it's hard to tell.\n\nAlternatively, think of the phrases as parts of a sentence. \"Blue is bad. Red is good. Yellow rules.\" So the answer is the color that comes after red in some sequence. Like in a game where red is good, then yellow rules. So the best color is yellow.\n\nAlternatively, think of the terms in terms of speed limits. In some countries, the order of signals is red, yellow, green. So after red comes caution (yellow), then go (green). But blue isn't part of that. If the clues are about signals, then maybe yellow rules here, but not sure.\n\nAlternatively, think of the colors in terms of the Olympic rings. Blue, yellow, black, green, red. But the clues don't align.\n\nAlternatively, think of the colors in terms of danger. Red is stop, yellow is caution, green is go. But blue isn't typically part of that. If blue is bad, maybe avoid it. So the best colors would be red and yellow. But the riddle asks for the best color.\n\nAlternatively, the answer is yellow because \"yellow rules,\" meaning it's the most important or dominant. So even though red is good, yellow is the top choice.\n\nAlternatively, the answer is red because it's explicitly stated as good, and the other clues are distractions.\n\nHmm, this is tricky. Let me try to look for patterns. The lines are:\n\n1. Blue is bad\n2. Red is good\n3. Yellow rules\n\nIf we consider that each line gives a preference, then red is the best. But the third line might indicate that yellow is the top choice. Alternatively, maybe the answer is a combination. But the question says \"best color,\" singular.\n\nAlternatively, the answer is red because it's the first one stated as good. But \"yellow rules\" could override that.\n\nAlternatively, think of the phrases as parts of a sentence: \"Red is good, yellow rules.\" So the best color would be yellow. But why the first line about blue?\n\nMaybe the first line is to eliminate blue. So from the remaining colors, red and yellow, which is best? The second line says red is good, third says yellow rules. So maybe the answer is yellow.\n\nAlternatively, in some contexts, \"yellow\" can mean caution, but here it's \"rules,\" which might imply it's the main or correct choice. So the answer would be yellow.\n\nAlternatively, think of the terms as parts of a hierarchy. If red is good, and yellow rules (is superior), then yellow is the best.\n\nAlternatively, the answer could be red, as it's commonly considered a positive color. But the third clue might change that.\n\nAlternatively, \"yellow rules\" could be a play on words. If you rearrange \"yellow,\" maybe \"yellow\" sounds like \"yellow\" which rules. Not sure.\n\nAlternatively, think of the terms as colors in the rainbow. Red, blue, and yellow are all present. But the clues point to red being good, yellow being rules, so perhaps yellow is the answer.\n\nWait, another angle: the words \"id good\" might be a typo. If it's \"id good,\" that's a psychological term (id is the instinctual part of the psyche). But that doesn't make sense here. So probably a typo for \"red is good.\"\n\nAssuming that, then the clues are:\n\n1. Blue is bad\n2. Red is good\n3. Yellow rules\n\nSo the best color would be the one that's good and rules. But how do these three clues interact? Maybe the answer is the color that is mentioned in the last clue, yellow, since it \"rules.\" So even though red is good, yellow is the best because it rules.\n\nAlternatively, the answer is red, as it's explicitly stated to be good, and the other clues are just additional info. But the presence of \"yellow rules\" complicates that.\n\nAlternatively, the answer is the combination of red and yellow, but since the question asks for a single color, it's probably one of them.\n\nGiven that \"yellow rules,\" which could mean yellow is the most important or dominant, the best color would be yellow. Even though red is good, yellow being the one that \"rules\" makes it the top choice.\n\nAlternatively, in some contexts, when someone says \"X rules,\" it means X is the main or most important. So if yellow rules here, it's the best choice.\n\nTherefore, the best color is yellow.\n</think>\n\nThe best color, based on the clues provided, is **yellow**. \n\n1. **Blue is bad**: Eliminates blue from consideration.  \n2. **Red is good**: Highlights red as a positive option.  \n3. **Yellow rules**: Indicates yellow holds precedence or dominance (\"rules\" here implies superiority).  \n\nWhile red is explicitly \"good,\" the final clue \"yellow rules\" suggests yellow is the overriding choice. Eliminating blue and prioritizing the latter clue, **yellow** emerges as the best color.\n\n\\boxed{yellow}",
                "role": "assistant"
            }
        }
    ],
    "created": 1746472739,
    "id": "chatcmpl-CSLmPAzaLWz8vJt6ZNHHI0Exk7Z5alfO",
    "model": "Phi-4-mini-reasoning",
    "object": "chat.completion",
    "system_fingerprint": "b0-unknown",
    "timings": {
        "predicted_ms": 184037.226,
        "predicted_n": 2239,
        "predicted_per_second": 12.166016890517573,
        "predicted_per_token_ms": 82.1961706118803,
        "prompt_ms": 1837.13,
        "prompt_n": 40,
        "prompt_per_second": 21.773091724592163,
        "prompt_per_token_ms": 45.928250000000006
    },
    "usage": {
        "completion_tokens": 2239,
        "prompt_tokens": 40,
        "total_tokens": 2279
    }
}



## Granite-3.3-2b-instruct

>    http -j post localhost:8080/v1/chat/completions model=granite-3.3-2b-instruct  messages:='[{"role":"control","content":"thinking"},{"role":"user","content":"need best color. use the following information to answer:\nblue is bad\ngreen is best\n"}]' 

*Faster* but no thinking. Hower, on ollama the model should think

```
{
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Based on the information provided, the best color would be green. Blue is considered less favorable in this context.",
                "role": "assistant"
            }
        }
    ],
    "created": 1746471670,
    "id": "chatcmpl-USaGhzveaNFimdPtWsEADFmN3PioErVM",
    "model": "granite-3.3-2b-instruct",
    "object": "chat.completion",
    "system_fingerprint": "b0-unknown",
    "timings": {
        "predicted_ms": 1203.276,
        "predicted_n": 24,
        "predicted_per_second": 19.94554865217955,
        "predicted_per_token_ms": 50.136500000000005,
        "prompt_ms": 1139.845,
        "prompt_n": 36,
        "prompt_per_second": 31.583241581092164,
        "prompt_per_token_ms": 31.66236111111111
    },
    "usage": {
        "completion_tokens": 24,
        "prompt_tokens": 36,
        "total_tokens": 60
    }
}

```


## Llama-3-2-3B-instruct

>    http -j post localhost:8080/v1/chat/completions model=Llama-3.2-3B-Instruct   messages:='[{"role":"control","content":"thinking"},{"role":"user","content":"need best color. use the following information to answer:\nblue is bad\ngreen is best\n"}]' 
    HTTP/1.1 200 OK
    Access-Control-Allow-Origin: 
    Content-Length: 657
    Content-Type: application/json; charset=utf-8
    Keep-Alive: timeout=5, max=100
    Server: llama.cpp


**Much** faster, but no thinking

```
{
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Based on the information provided, I would recommend the color... GREEN!",
                "role": "assistant"
            }
        }
    ],
    "created": 1746471482,
    "id": "chatcmpl-aLHV0p8Y72sk1HvLGSqytxzhoYeO7Ajx",
    "model": "Llama-3.2-3B-Instruct",
    "object": "chat.completion",
    "system_fingerprint": "b0-unknown",
    "timings": {
        "predicted_ms": 838.393,
        "predicted_n": 15,
        "predicted_per_second": 17.891370753334055,
        "predicted_per_token_ms": 55.89286666666667,
        "prompt_ms": 1599.3220000000001,
        "prompt_n": 34,
        "prompt_per_second": 21.25900850485393,
        "prompt_per_token_ms": 47.03888235294118
    },
    "usage": {
        "completion_tokens": 15,
        "prompt_tokens": 34,
        "total_tokens": 49
    }
}

```


### Tool calling

server must be started with **--jinja** flag

works also for granite. thinking cannot be enabled however with tools? 

**tool is not called**

>   http -j post localhost:8080/v1/chat/completions model=Phi-4-mini-reasoning tools:='[ { "function": { "description": "Returns the criteria to rank colors.", "name": "Elasticsearch-get_latest_version", "strict": false, "parameters": { "type": "object", "required": [], "properties": { "majorVersion": { "description": "Major version to filter by (e.g. 7, 8). Defaults to latest", "type": "integer" } } } }, "type": "function" } ]' controls:='["thinking"]' messages:='[{"role":"control","content":"thinking"},{"role":"user","content":"need best color. if required, use provided tool for color ranging. the following information is available to answer:\nblue is bad\ngreen is best\n"}]' 


{
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "green",
                "role": "assistant"
            }
        }
    ],
    "created": 1746546331,
    "id": "chatcmpl-0jDgohUXIHM665cI3wSG5B6Ws9zBMtpN",
    "model": "Phi-4-mini-reasoning",
    "object": "chat.completion",
    "system_fingerprint": "b0-unknown",
    "timings": {
        "predicted_ms": 726.894,
        "predicted_n": 10,
        "predicted_per_second": 13.757164043175484,
        "predicted_per_token_ms": 72.6894,
        "prompt_ms": 93.871,
        "prompt_n": 1,
        "prompt_per_second": 10.65291730140299,
        "prompt_per_token_ms": 93.871
    },
    "usage": {
        "completion_tokens": 10,
        "prompt_tokens": 265,
        "total_tokens": 275
    }
}


**tool is called**
>  http -j post localhost:8080/v1/chat/completions model=Phi-4-mini-reasoning tools:='[ { "function": { "description": "Returns the criteria to rank colors.", "name": "Elasticsearch-get_latest_version", "strict": false, "parameters": { "type": "object", "required": [], "properties": { "majorVersion": { "description": "Major version to filter by (e.g. 7, 8). Defaults to latest", "type": "integer" } } } }, "type": "function" } ]' controls:='["thinking"]' messages:='[{"role":"control","content":"thinking"},{"role":"user","content":"need best color. use provided tool for color ranging if required. the following information is available to answer:\noptions are blube,green, white"}]' 

{
    "choices": [
        {
            "finish_reason": "tool_calls",
            "index": 0,
            "message": {
                "content": null,
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "arguments": "{\"majorVersion\":7}",
                            "name": "Elasticsearch-get_latest_version"
                        },
                        "id": "x5WmgwDZ4f8IldnoCgzZCmMhKNgxqtHT",
                        "type": "function"
                    }
                ]
            }
        }
    ],
    "created": 1746546594,
    "id": "chatcmpl-BOIiObxB7LiTgsyQopUY9ProFMdEPVyE",
    "model": "Phi-4-mini-reasoning",
    "object": "chat.completion",
    "system_fingerprint": "b0-unknown",
    "timings": {
        "predicted_ms": 2591.0950000000003,
        "predicted_n": 33,
        "predicted_per_second": 12.735928246552133,
        "predicted_per_token_ms": 78.51803030303032,
        "prompt_ms": 314.576,
        "prompt_n": 10,
        "prompt_per_second": 31.788820507603884,
        "prompt_per_token_ms": 31.457600000000003
    },
    "usage": {
        "completion_tokens": 33,
        "prompt_tokens": 262,
        "total_tokens": 295
    }
}


## Granite


**tool not called, in this case**
>  http -j post localhost:8080/v1/chat/completions model=granite-3.3-2b-instruct  tools:='[ { "function": { "description": "Returns ranked list of colors. best first.", "name": "GetColorRank", "strict": false, "parameters": { "type": "object", "required": [], "properties": { "colorSet": { "description": "List of colors to check for. Defaults to [red]", "type": "array" } } } }, "type": "function" } ]'  messages:='[{"role":"control","content":"thinking"},{"role":"user","content":"need best color.use provided documents. use provided tools only if required"},{"role":"user","content":"document_1:blue is bad\ndocument_2:green is nice" }]' 

{
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "green is a nice color according to the provided documents. It is recommended as the best choice.",
                "role": "assistant"
            }
        }
    ],
    "created": 1746551949,
    "id": "chatcmpl-yvt3mzE0ebVWjUqnMakAmYRbwh6sGV8E",
    "model": "granite-3.3-2b-instruct",
    "object": "chat.completion",
    "system_fingerprint": "b0-unknown",
    "timings": {
        "predicted_ms": 1261.588,
        "predicted_n": 25,
        "predicted_per_second": 19.816295018659023,
        "predicted_per_token_ms": 50.463519999999995,
        "prompt_ms": 573.801,
        "prompt_n": 28,
        "prompt_per_second": 48.79740537224578,
        "prompt_per_token_ms": 20.49289285714286
    },
    "usage": {
        "completion_tokens": 25,
        "prompt_tokens": 196,
        "total_tokens": 221
    }
}

**tool called**

>  http -j post localhost:8080/v1/chat/completions model=granite-3.3-2b-instruct  tools:='[ { "function": { "description": "Returns ranked list of colors. best first.", "name": "GetColorRank", "strict": false, "parameters": { "type": "object", "required": [], "properties": { "colorSet": { "description": "List of colors to check for. Defaults to [red]", "type": "array" } } } }, "type": "function" } ]'  messages:='[{"role":"control","content":"thinking"},{"role":"user","content":"need best color.use provided documents. use provided tools only if required"},{"role":"user","content":"document_1:blue is bad\ndocument_2:green is not good" }]' 

{
    "choices": [
        {
            "finish_reason": "tool_calls",
            "index": 0,
            "message": {
                "content": null,
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "arguments": "{\"colorSet\":[\"blue\",\"green\"]}",
                            "name": "GetColorRank"
                        },
                        "id": "IuKcHkSjMEdcoXsLwl7FIAArBGSdGccB",
                        "type": "function"
                    }
                ]
            }
        }
    ],
    "created": 1746551966,
    "id": "chatcmpl-mw1cmZrN30RGFsq2T2MFGNmxB8BTvmEq",
    "model": "granite-3.3-2b-instruct",
    "object": "chat.completion",
    "system_fingerprint": "b0-unknown",
    "timings": {
        "predicted_ms": 1501.561,
        "predicted_n": 29,
        "predicted_per_second": 19.313234693762027,
        "predicted_per_token_ms": 51.77796551724138,
        "prompt_ms": 190.787,
        "prompt_n": 7,
        "prompt_per_second": 36.69013087893829,
        "prompt_per_token_ms": 27.255285714285716
    },
    "usage": {
        "completion_tokens": 29,
        "prompt_tokens": 197,
        "total_tokens": 226
    }
}

**tool result fed back to llm**

*Make sure to return a precise description of the result and repeat the initial query*

>  http -j post localhost:8080/v1/chat/completions model=granite-3.3-2b-instruct  messages:='[{  "role": "tool",  "content": "{\"ranked list of colors. first is best, last is worst\": [\"green\", \"blue\"]}",  "tool_call_id": "IuKcHkSjMEdcoXsLwl7FIAArBGSdGccB"},{"role":"user","content":"need best color"}]'

{
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Based on the tool's response, the best color according to the tool's analysis is \"green\".",
                "role": "assistant"
            }
        }
    ],
    "created": 1746553036,
    "id": "chatcmpl-9igwUqK6HqrhbEQui9VSKMLhiGywAKQ1",
    "model": "granite-3.3-2b-instruct",
    "object": "chat.completion",
    "system_fingerprint": "b0-unknown",
    "timings": {
        "predicted_ms": 1102.575,
        "predicted_n": 21,
        "predicted_per_second": 19.046323379361947,
        "predicted_per_token_ms": 52.50357142857143,
        "prompt_ms": 248.372,
        "prompt_n": 10,
        "prompt_per_second": 40.262187364115114,
        "prompt_per_token_ms": 24.837200000000003
    },
    "usage": {
        "completion_tokens": 21,
        "prompt_tokens": 135,
        "total_tokens": 156
    }
}


