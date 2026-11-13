"""
A LiveKit voice agent that uses turn-blend for voice activity and end-of-turn detection.

  microphone -> Deepgram streaming STT -> turn-blend decides the turn -> OpenAI LLM -> Deepgram Aura-2 TTS

Setup:
    pip install -r examples/requirements-livekit.txt
    cp examples/.env.example .env      # fill in LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, DEEPGRAM_API_KEY, OPENAI_API_KEY
Run:
    python examples/livekit_agent.py dev        # connects to your LiveKit project; talk to it from the Agents Playground
    python examples/livekit_agent.py console    # local microphone and speaker, no LiveKit room needed

The only turn-blend specific lines are the `vad=` and `turn_handling=` arguments below.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "integrations")]

from dotenv import load_dotenv
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import deepgram, openai

from livekit_turn_blend import TurnBlendVAD

load_dotenv(ROOT / ".env")

server = AgentServer()
turn_blend = TurnBlendVAD(str(ROOT / "turn-blend.onnx"))    # loaded once, shared by every call


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="en"),
        llm=openai.LLM(model="gpt-5.4-nano", reasoning_effort="none"),
        tts=deepgram.TTS(model="aura-2-thalia-en"),
        vad=turn_blend,
        turn_handling={
            "turn_detection": "vad",       # the turn ends when turn-blend reports end of speech
            "endpointing": {"mode": "fixed", "min_delay": 0.0, "max_delay": 3.0},   # no extra wait on top of it
        },
    )
    await session.start(
        room=ctx.room,
        agent=Agent(instructions=(
            "You are a friendly voice assistant in a live phone-style conversation. Your replies are spoken "
            "aloud, so keep them to one or two short sentences, with no lists, markdown or emoji."
        )),
    )
    await session.generate_reply(instructions="Greet the caller in one short sentence.")


if __name__ == "__main__":
    cli.run_app(server)
