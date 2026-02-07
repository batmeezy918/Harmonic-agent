#!/usr/bin/env python3
"""
Harmonic Automation Agent:
(1) Tool-using agent (function calling)
(3) Natural-language infra control via shell tool
(6) CI/CD intelligence via log analysis + patch suggestions
(7) Voice: speech-to-text (transcribe) and text-to-speech (speak)

Security:
- Requires OPENAI_API_KEY in environment.
- Shell tool is allowlisted (basic safety guardrails).
"""

import os
import json
import subprocess
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
MODEL_AGENT = os.getenv("OPENAI_AGENT_MODEL", "gpt-5.2")  # pick your preferred model
TRANSCRIBE_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")  # voice options vary by model

WORKDIR = Path(os.getenv("AGENT_WORKDIR", ".")).resolve()
ARTIFACTS = WORKDIR / "agent_artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Basic shell allowlist: keep it tight. Expand deliberately.
ALLOWED_SHELL_PREFIXES = [
    "ls", "pwd", "cat", "sed", "awk", "grep", "rg", "find",
    "python", "python3", "pip", "pip3",
    "node", "npm",
    "git",
    "docker",
    "bash", "sh",
    "uname", "whoami",
    "make", "cmake",
    "gradle", "./gradlew",
]

client = OpenAI()

# ----------------------------
# Tool implementations
# ----------------------------
def _shell_allowed(command: str) -> bool:
    cmd = command.strip()
    if not cmd:
        return False
    first = shlex.split(cmd)[0]
    return any(first == p or first.startswith(p + "/") for p in ALLOWED_SHELL_PREFIXES)

def tool_run_shell(command: str, timeout_sec: int = 120) -> Dict[str, Any]:
    """Run a shell command inside WORKDIR (allowlisted)."""
    if not _shell_allowed(command):
        return {
            "ok": False,
            "error": "Command not allowed by allowlist.",
            "command": command,
            "allowed_prefixes": ALLOWED_SHELL_PREFIXES,
        }

    try:
        proc = subprocess.run(
            command,
            cwd=str(WORKDIR),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "command": command,
            "stdout": proc.stdout[-20000:],  # cap output
            "stderr": proc.stderr[-20000:],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Timeout", "command": command}

def tool_save_text(filename: str, content: str) -> Dict[str, Any]:
    """Save text artifact under agent_artifacts/."""
    p = (ARTIFACTS / filename).resolve()
    if ARTIFACTS not in p.parents:
        return {"ok": False, "error": "Invalid path"}
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": str(p)}

def tool_transcribe_audio(path: str) -> Dict[str, Any]:
    """Speech-to-text from an audio file (wav/mp3/m4a...)."""
    ap = Path(path).expanduser().resolve()
    if not ap.exists():
        return {"ok": False, "error": "File not found", "path": str(ap)}

    with ap.open("rb") as f:
        # Audio API: speech-to-text guide
        # (Model names vary; default here is gpt-4o-mini-transcribe)
        out = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=f,
        )
    text = out.text if hasattr(out, "text") else str(out)
    return {"ok": True, "text": text}

def tool_speak_text(text: str, out_file: str = "tts_output.mp3") -> Dict[str, Any]:
    """Text-to-speech to an MP3 file under agent_artifacts/."""
    out_path = (ARTIFACTS / out_file).resolve()
    if ARTIFACTS not in out_path.parents:
        return {"ok": False, "error": "Invalid out_file"}

    audio = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
    )
    # The SDK returns bytes-like audio
    out_path.write_bytes(audio.read())
    return {"ok": True, "path": str(out_path)}

# ----------------------------
# Tool schemas (function calling)
# ----------------------------
TOOLS = [
    {
        "type": "function",
        "name": "run_shell",
        "description": "Run an allowlisted shell command in the repo/workdir and return stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_sec": {"type": "integer", "default": 120},
            },
            "required": ["command"],
        },
    },
    {
        "type": "function",
        "name": "save_text",
        "description": "Save a text file under agent_artifacts/.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["filename", "content"],
        },
    },
    {
        "type": "function",
        "name": "transcribe_audio",
        "description": "Transcribe an audio file to text.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "speak_text",
        "description": "Convert text to speech and write an MP3 file under agent_artifacts/.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "out_file": {"type": "string", "default": "tts_output.mp3"},
            },
            "required": ["text"],
        },
    },
]

def dispatch_tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name == "run_shell":
        return tool_run_shell(**arguments)
    if name == "save_text":
        return tool_save_text(**arguments)
    if name == "transcribe_audio":
        return tool_transcribe_audio(**arguments)
    if name == "speak_text":
        return tool_speak_text(**arguments)
    return {"ok": False, "error": f"Unknown tool: {name}"}

# ----------------------------
# Agent loop
# ----------------------------
SYSTEM_INSTRUCTIONS = """You are an automation agent.
Goals:
1) Execute tasks by calling tools (run_shell/save_text/transcribe_audio/speak_text).
2) When a command fails, diagnose from stderr and propose a fix; prefer minimal changes.
3) For CI/CD logs, extract root cause, list concrete patches, and output a 'next commands' checklist.
4) Be explicit and deterministic: exact commands, exact filenames.
"""

def run_agent(user_request: str) -> str:
    # Initial request to model with tools available
    resp = client.responses.create(
        model=MODEL_AGENT,
        input=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_request},
        ],
        tools=TOOLS,
    )

    # Tool-calling loop
    while True:
        tool_calls = []
        for item in resp.output:
            if item.type == "tool_call":
                tool_calls.append(item)

        if not tool_calls:
            # Final answer
            return resp.output_text

        tool_outputs = []
        for call in tool_calls:
            name = call.name
            args = call.arguments or {}
            if isinstance(args, str):
                args = json.loads(args)
            result = dispatch_tool_call(name, args)
            tool_outputs.append(
                {"type": "tool_output", "tool_call_id": call.id, "output": json.dumps(result)}
            )

        # Send tool outputs back
        resp = client.responses.create(
            model=MODEL_AGENT,
            input=tool_outputs,
        )

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY env var. Set it before running.")

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="What you want the agent to do.")
    args = ap.parse_args()

    print(run_agent(args.task))

if __name__ == "__main__":
    main()
