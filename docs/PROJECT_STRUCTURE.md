# Project Structure

This project is organized so each folder has one clear responsibility:

- `src/deraining/data/`: dataset loading and image pairing logic
- `src/deraining/models/`: neural network definitions
- `src/deraining/utils/`: shared image read/write helpers
- `src/deraining/cli/`: runnable training, testing, and demo app logic
- `train.py`, `test.py`, `app.py`: simple entry points that call the CLI modules
- `assets/results/`: sample outputs used in the README or presentation
- `outputs/`: generated checkpoints and evaluation results
- `Rain100L/`: local dataset folder, not tracked in Git

## How To Explain It

You can explain the project in four layers:

1. `data` prepares paired rainy and clean images from Rain100L.
2. `models` defines the deraining networks: an autoencoder and a Pix2Pix GAN.
3. `cli` runs the workflows: training, testing, and single-image inference.
4. `outputs` stores everything produced by the workflows, such as checkpoints and result images.

## Short Viva Version

"My project follows a clean ML structure. The reusable code is inside `src/deraining`, where `data` handles dataset loading, `models` defines the networks, `utils` stores shared image helpers, and `cli` contains the training, testing, and inference workflows. The root scripts are just launch files, so the project is easy to read, run, and explain."
