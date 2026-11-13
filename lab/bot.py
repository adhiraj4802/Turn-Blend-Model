"""
Voice bot for testing turn-blend in a real call, built on Pipecat.

  microphone (WebRTC) -> Deepgram streaming STT -> our Silero VAD + turn-blend decide the turn
                      -> OpenAI gpt-5.4-nano (no reasoning) -> Deepgram Aura-2 TTS -> speaker

Run:  ./run_bot.sh                        local WebRTC: open http://localhost:7860 and press Connect
      ./run_bot.sh -t daily               Daily room instead (the Daily account needs a payment method)
      TURN_MODE=smart_turn ./run_bot.sh   same bot with Pipecat's stock Silero + Smart Turn v3, to compare

Settings console: http://localhost:7861 - edit the VAD and blend settings while you talk and see each
turn's measured results. Edits apply to the call in progress and are saved to bot_settings.json.

Keys are read from .env: DEEPGRAM_API_KEY, OPENAI_API_KEY (and DAILY_API_KEY for Daily).
Per-turn results are appended to bot_turns.jsonl.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

import console
from loguru import logger
from pipecat.frames.frames import LLMRunFrame, MetricsFrame
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import DailyRunnerArguments, RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

TURN_MODE = os.environ.get("TURN_MODE", "blend")            # blend | smart_turn
STT_MODEL = os.environ.get("STT_MODEL", "nova-3")
TTS_VOICE = os.environ.get("TTS_VOICE", "aura-2-thalia-en")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.4-nano")
console.STATE["mode"] = TURN_MODE

SYSTEM = (
    "You are a friendly voice assistant in a live phone-style conversation. Your replies are spoken aloud, "
    "so keep them to one or two short sentences, with no lists, markdown or emoji. Ask a short follow-up "
    "question when it keeps the conversation going."
)


def vad_and_turn():
    """The VAD that starts turns and the analyzer that ends them."""
    if TURN_MODE == "smart_turn":   # Pipecat's stock pair, for comparison
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        vad = SileroVADAnalyzer(params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=0.2))
        return vad, LocalSmartTurnAnalyzerV3()   # the console's settings do not apply in this mode
    from blend_turn import BlendTurnAnalyzer, LabSileroVADAnalyzer
    return LabSileroVADAnalyzer(), BlendTurnAnalyzer()


def make_transport(runner_args: RunnerArguments):
    if isinstance(runner_args, SmallWebRTCRunnerArguments):
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
        return SmallWebRTCTransport(webrtc_connection=runner_args.webrtc_connection,
                                    params=TransportParams(audio_in_enabled=True, audio_out_enabled=True))
    if isinstance(runner_args, DailyRunnerArguments):
        from pipecat.transports.daily.transport import DailyParams, DailyTransport
        return DailyTransport(runner_args.room_url, runner_args.token, "turn-blend bot", DailyParams(
            audio_in_enabled=True, audio_out_enabled=True, camera_out_enabled=False))
    raise SystemExit(f"Unsupported transport: {type(runner_args).__name__}. Use -t webrtc or -t daily.")


async def bot(runner_args: RunnerArguments):
    """Entry point called by the Pipecat runner for every client that connects."""
    for key in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY"):
        if not os.environ.get(key):
            raise SystemExit(f"{key} is missing. Put it in {ROOT / '.env'}")

    transport = make_transport(runner_args)
    stt = DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramSTTService.Settings(model=STT_MODEL, language="en"),
    )
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(model=LLM_MODEL, extra={"reasoning_effort": "none"}),
    )
    tts = DeepgramTTSService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramTTSService.Settings(voice=TTS_VOICE),
    )

    vad, turn = vad_and_turn()
    context = LLMContext([{"role": "system", "content": SYSTEM}])
    aggregators = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
        vad_analyzer=vad,   # starts turns and detects interruptions
        user_turn_strategies=UserTurnStrategies(stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn)]),
    ))

    latency = UserBotLatencyObserver()

    async def show_metric(name, seconds):
        """Send a timing to the page's Metrics tab, where it is charted next to the per-service ones."""
        await worker.queue_frames([MetricsFrame(data=[TTFBMetricsData(processor=name, value=seconds)])])

    @latency.event_handler("on_latency_measured")
    async def on_latency(_, seconds):
        # Full turn latency: from the moment the user stopped speaking to the bot's first audio.
        # It covers turn detection, transcription, the LLM's first sentence and speech synthesis.
        detect_ms = getattr(turn, "last_delay_ms", None)
        logger.info(f"FULL TURN LATENCY: {seconds * 1000:.0f} ms (user stopped speaking -> bot started speaking)"
                    + (f", of which turn detection {detect_ms:.0f} ms" if detect_ms is not None else ""))
        console.record("full_turn_latency", ms=round(seconds * 1000),
                  turn_detection_ms=None if detect_ms is None else round(detect_ms))
        await show_metric("Full turn latency", seconds)

    @latency.event_handler("on_latency_breakdown")
    async def on_breakdown(_, breakdown):
        lines = list(breakdown.turn_contribution_lines())
        for line in lines:
            logger.info(line)
        console.record("latency_breakdown", lines=lines)

    user_agg = aggregators.user()

    @user_agg.event_handler("on_user_turn_stopped")
    async def on_turn_stopped(_, strategy, message):
        logger.info(f"user turn: {message.content!r}")
        console.record("user_turn", text=message.content)
        if getattr(turn, "last_delay_ms", None) is not None:
            await show_metric("Turn detection", turn.last_delay_ms / 1000)

    pipeline = Pipeline([transport.input(), stt, user_agg, llm, tts, transport.output(), aggregators.assistant()])
    worker = PipelineWorker(pipeline, params=PipelineParams(
        audio_in_sample_rate=16000, enable_metrics=True, enable_usage_metrics=True,
    ), observers=[latency])

    @transport.event_handler("on_client_connected")
    async def on_connected(_, client):
        console.STATE["calls"] += 1
        logger.info(f"client connected | turn detection: {TURN_MODE}")
        context.add_message({"role": "system", "content": "Greet the caller in one short sentence."})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_, client):
        await worker.cancel()

    await PipelineRunner(handle_sigint=runner_args.handle_sigint).run(worker)


if __name__ == "__main__":
    from pipecat.runner.run import main

    console.load_saved()
    console.start(int(os.environ.get("CONSOLE_PORT", "7861")))
    main()
