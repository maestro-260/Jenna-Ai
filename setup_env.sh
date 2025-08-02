#!/bin/bash

# Check Python version
if ! python3 --version | grep -q "3.11"; then
    echo "Python 3.11 is required. Install it first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip not found. Install Python packages manually."
    exit 1
fi

# Install requirements
pip install -r requirements.txt || { echo "Failed to install dependencies"; exit 1; }

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Installing now..."
    curl -fsSL https://ollama.com/install.sh | sh || { echo "Failed to install Ollama"; exit 1; }
fi

# Ensure required models are available
ollama list | grep mistral || ollama pull mistral || { echo "Failed to pull mistral model"; exit 1; }

echo "Setup completed successfully."

curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

pyenv install 3.11.7
pyenv global 3.11.7

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt || { echo "Failed to install requirements"; exit 1; }

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html || { echo "Failed to install PyTorch"; exit 1; }

mkdir -p assets/voices
ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 3 -acodec pcm_s16le assets/voices/reference_audio.wav || true

pip install langchain-community browser-use || { echo "Failed to install additional packages"; exit 1; }

pip install pytest
pip install sentence-transformers

export PORCUPINE_KEY="your_porcupine_key"
export OPENVOICE_KEY="your_openvoice_key"

# Install required packages
pip install sounddevice vosk numpy --upgrade

# Download VOSK model (English small)
mkdir -p models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip -P models
unzip models/vosk-model-small-en-us-0.15.zip -d models/
mv models/vosk-model-small-en-us-0.15 models/vosk-model-small-en-us

# OpenVoice setup
if [ ! -d "checkpoints/converter" ]; then
    echo "Downloading OpenVoice checkpoints..."
    mkdir -p checkpoints
    wget https://github.com/myshell-ai/OpenVoice/releases/download/v0.0.1/converter.zip -O checkpoints/converter.zip
    unzip checkpoints/converter.zip -d checkpoints/
    rm checkpoints/converter.zip
fi

pip install rasa
pip install deepspeech
pip install google-auth-oauthlib google-api-python-client httpx ollama

# Initialize directories
mkdir -p memory assets/voices models config logs

# Download required models
python -m utils.model_download

# Verify setup
python -m utils.system_check
