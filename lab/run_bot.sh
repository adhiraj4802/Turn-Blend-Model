#!/usr/bin/env bash
# Starts the Pipecat voice bot that uses our VAD and turn-blend to decide when you have finished speaking.
#
#   ./run_bot.sh                        local WebRTC: open http://localhost:7860 and press Connect
#   ./run_bot.sh -t daily               Daily room instead (needs DAILY_API_KEY and a payment method on Daily)
#   TURN_MODE=smart_turn ./run_bot.sh   Pipecat's stock Silero + Smart Turn v3, to compare
#
# Needs .env with DEEPGRAM_API_KEY and OPENAI_API_KEY.
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] || { echo "Create .env with DEEPGRAM_API_KEY and OPENAI_API_KEY first." >&2; exit 1; }

if [ ! -x .venv/bin/python ]; then
  echo "Creating Python environment in .venv ..."
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
fi
if [ ! -f .venv/.installed-bot2 ]; then
  echo "Installing packages from requirements-bot.txt ..."
  .venv/bin/pip install -q -r requirements-bot.txt
  touch .venv/.installed-bot2
fi

[ $# -eq 0 ] && set -- -t webrtc
exec .venv/bin/python -u bot.py "$@"
