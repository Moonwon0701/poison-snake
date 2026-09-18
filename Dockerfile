# The deployed snake plays with numpy, so this image needs neither PyTorch
# nor a GPU: requirements.txt alone, about 100 MB.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ agent/
COPY gym_env/ gym_env/
COPY rl/numpy_agent.py rl/__init__.py rl/
COPY models/ models/
COPY server.py .

ENV HOST=0.0.0.0 PORT=8000 SNAKE_BRAIN=rl SNAKE_MODEL=models/snake.npz
EXPOSE 8000
CMD ["python", "server.py"]
