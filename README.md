# Python Tetris bot for TETR.IO
This bot retrieves game data through color matching and calculates optimal moves using multiprocessing. It's optimized for back-to-back (b2b) moves, including T-spins and other advanced spins.

## Demo
https://youtu.be/nqyY3mnWVAE

## Quick Start

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the calibration wizard (first time setup):
    ```bash
    python bot.py --calibrate
    ```

3. Start the bot with your saved configuration:
    ```bash
    python bot.py --use-config
    ```

## Calibration

The bot includes an interactive calibration wizard that captures your screen coordinates for accurate gameplay detection. This is much easier than manually editing coordinate values!

### Running Calibration

```bash
python bot.py --calibrate
```

### Calibration Steps

The wizard guides you through 5 steps to capture the necessary screen positions:

| Step | What to Capture | Description |
|------|-----------------|-------------|
| 1 | Board Top-Left | Move mouse to the top-left corner of the Tetris board |
| 2 | Board Bottom-Right | Move mouse to the bottom-right corner of the Tetris board |
| 3 | Next Piece #1 | Move mouse to the center of the first (topmost) next piece preview |
| 4 | Next Piece #5 | Move mouse to the center of the fifth (bottommost) next piece preview |
| 5 | Held Piece | Move mouse to the center of the held piece display |

### Controls During Calibration

- **Press `=`** - Capture the current mouse position
- **Press `Escape`** - Cancel calibration and exit

### Configuration File

After successful calibration, your settings are saved to `config.json`. This file stores:
- Screen resolution
- Board coordinates
- Next piece positions
- Held piece position
- AI parameters (multiprocessing workers, pruning settings)

## Usage

### Command Line Options

| Option | Description |
|--------|-------------|
| `--calibrate` | Run the interactive calibration wizard |
| `--use-config` | Load settings from `config.json` |
| `--mp N` | Override multiprocessing workers (default: 4) |
| `--pruning-moves N` | Override pruning moves parameter |
| `--pruning-breadth N` | Override pruning breadth parameter |

### Examples

```bash
# First time setup - calibrate your screen
python bot.py --calibrate

# Run bot with saved configuration
python bot.py --use-config

# Run with custom performance settings
python bot.py --use-config --mp 8

# Run with custom AI parameters
python bot.py --use-config --mp 8 --pruning-moves 3 --pruning-breadth 5

# Run without config (uses hardcoded defaults)
python bot.py
```

### Manual Configuration (Legacy)

If you prefer not to use the calibration wizard, you can still manually adjust the parameters in `bot.py`:
```python
screen_resolution=(1920, 1080),
board_top_left=(787, 220),
board_bottom_right=(1133, 899),
next_piece_xy_0=(1260, 300),
next_piece_xy_4=(1260, 721),
held_piece_xy=(691, 300),
```

## Game Settings

For optimal bot performance, configure TETR.IO with these settings:
- **ARR**: 0ms
- **DAS**: 40ms
- **SDF**: max

## Troubleshooting

### Calibration Issues

- **Mouse position not capturing**: Ensure no other application is intercepting the `=` key
- **Wrong coordinates saved**: Re-run `python bot.py --calibrate` to recalibrate
- **Config file not loading**: Check that `config.json` exists and is valid JSON

### Bot Performance

- **Bot moves incorrectly**: Re-calibrate to ensure accurate board detection
- **Bot is slow**: Increase `--mp` value for more parallel processing
- **Bot misses pieces**: Ensure the next piece and held piece positions are correctly calibrated

## Dependencies

- Python 3.x
- pyautogui
- mss
- numpy
- See `requirements.txt` for full list

## Disclaimer

Use this bot at your own discretion. Using it in multiplayer mode could result in your account and IP address being banned.
