from multiprocessing import Pool
import argparse
import json
import os
import random

import keyboard
import pyautogui
import time
import numpy as np
import math
from PIL import ImageGrab
from PIL.Image import Image

from constants import colors, colors_name, tetris_pieces, NUM_ROW, NUM_COL
from tetris_ai import find_best_move

CONFIG_FILE = "config.json"

# Default delay settings (in milliseconds)
DEFAULT_MOVE_DELAY_MS = 30
DEFAULT_ACTION_DELAY_MS = 50
DEFAULT_DELAY_VARIANCE_PERCENT = 20

def load_config():
    """Load configuration from config.json if it exists.
    Applies default values for any missing delay settings."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            # Apply defaults for delay settings if not present
            if 'move_delay_ms' not in config:
                config['move_delay_ms'] = DEFAULT_MOVE_DELAY_MS
            if 'action_delay_ms' not in config:
                config['action_delay_ms'] = DEFAULT_ACTION_DELAY_MS
            if 'delay_variance_percent' not in config:
                config['delay_variance_percent'] = DEFAULT_DELAY_VARIANCE_PERCENT
            return config
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file: {e}")
    return None


def save_config(config):
    """Save configuration to config.json."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguration saved to {CONFIG_FILE}")


def run_calibration_wizard():
    """
    Interactive calibration wizard to capture screen coordinates for TetrioBot.
    Guides the user through clicking on specific positions on the game screen.
    """
    print("\n" + "=" * 60)
    print("       TETRIOBOT CALIBRATION WIZARD")
    print("=" * 60)
    print("\nThis wizard will help you configure the screen coordinates")
    print("for the TetrioBot. You will be guided through 5 steps.\n")
    print("INSTRUCTIONS:")
    print("  1. Position your TETR.IO game window where you want it")
    print("  2. For each step, move your mouse to the indicated position")
    print("  3. Press '=' key to capture the current mouse position")
    print("  4. Press 'Escape' to cancel calibration at any time\n")
    
    # Detect screen offset for multi-monitor setups
    screens = pyautogui.size()
    print(f"Detected primary screen resolution: {screens[0]}x{screens[1]}")
    
    # Try to detect monitor offset
    # pyautogui.position() returns absolute coordinates across all monitors
    print("\nTo detect your monitor offset, please move your mouse to the")
    print("TOP-LEFT corner of your TETR.IO game window and press '='")
    print("(This helps with multi-monitor setups)\n")
    
    steps = [
        ("BOARD TOP-LEFT", "Click on the TOP-LEFT corner of the game board (where pieces stack)"),
        ("BOARD BOTTOM-RIGHT", "Click on the BOTTOM-RIGHT corner of the game board"),
        ("NEXT PIECE #1", "Click on the CENTER of the FIRST next piece (top of next queue)"),
        ("NEXT PIECE #5", "Click on the CENTER of the FIFTH next piece (bottom of next queue)"),
        ("HELD PIECE", "Click on the CENTER of the HELD piece display"),
    ]
    
    captured_positions = []
    
    for i, (name, instruction) in enumerate(steps, 1):
        print(f"\n--- Step {i}/5: {name} ---")
        print(f"  {instruction}")
        print("  >> Move your mouse to the position and press '=' to capture")
        print("  >> Press 'Escape' to cancel")
        
        # Wait for keypress
        while True:
            event = keyboard.read_event(suppress=False)
            if event.event_type == keyboard.KEY_DOWN:
                if event.name == '=':
                    pos = pyautogui.position()
                    captured_positions.append(pos)
                    print(f"  ✓ Captured: ({pos[0]}, {pos[1]})")
                    time.sleep(0.2)  # Small delay to prevent double-capture
                    break
                elif event.name == 'escape':
                    print("\n\nCalibration cancelled by user.")
                    return None
    
    # Process captured positions
    board_top_left = captured_positions[0]
    board_bottom_right = captured_positions[1]
    next_piece_0 = captured_positions[2]
    next_piece_4 = captured_positions[3]
    held_piece = captured_positions[4]
    
    # Calculate screen offset (assume primary monitor if board is on it)
    # Use the top-left of the board to determine which monitor
    screen_offset = (0, 0)
    
    # Check if position suggests a different monitor
    primary_width, primary_height = pyautogui.size()
    if board_top_left[0] < 0:
        # Left of primary monitor
        screen_offset = (board_top_left[0] - (board_top_left[0] % primary_width), 0)
    elif board_top_left[0] >= primary_width:
        # Right of primary monitor
        screen_offset = (primary_width, 0)
    
    # Ask user about screen offset
    print(f"\n--- Screen Offset Detection ---")
    print(f"  Detected board position: ({board_top_left[0]}, {board_top_left[1]})")
    print(f"  Suggested screen_offset: {screen_offset}")
    print(f"\n  If your game is on a secondary monitor, you may need to adjust this.")
    print(f"  Common values: (0, 0) for primary, (-1920, 0) for left monitor,")
    print(f"  (1920, 0) for right monitor")
    
    # Get screen resolution (use the board area to estimate)
    screen_resolution = (primary_width, primary_height)
    
    # Prompt for delay settings
    print(f"\n--- Step 6/6: DELAY SETTINGS ---")
    print("  Configure delays to make inputs appear more human-like.")
    print("  This helps avoid anti-cheat detection.\n")
    
    print(f"  Move Delay: Delay between each keypress (default: {DEFAULT_MOVE_DELAY_MS}ms)")
    move_delay_input = input(f"  Enter move delay in ms (or press Enter for default): ").strip()
    move_delay_ms = int(move_delay_input) if move_delay_input else DEFAULT_MOVE_DELAY_MS
    
    print(f"\n  Action Delay: Delay after actions like hold/rotate (default: {DEFAULT_ACTION_DELAY_MS}ms)")
    action_delay_input = input(f"  Enter action delay in ms (or press Enter for default): ").strip()
    action_delay_ms = int(action_delay_input) if action_delay_input else DEFAULT_ACTION_DELAY_MS
    
    print(f"\n  Delay Variance: Random variance percentage (default: {DEFAULT_DELAY_VARIANCE_PERCENT}%)")
    print("  Example: 20% variance on 30ms = delays between 24ms-36ms")
    variance_input = input(f"  Enter variance percentage (or press Enter for default): ").strip()
    delay_variance_percent = int(variance_input) if variance_input else DEFAULT_DELAY_VARIANCE_PERCENT
    
    # Build the configuration
    config = {
        "screen_offset": list(screen_offset),
        "screen_resolution": list(screen_resolution),
        "board_top_left": list(board_top_left),
        "board_bottom_right": list(board_bottom_right),
        "next_piece_xy_0": list(next_piece_0),
        "next_piece_xy_4": list(next_piece_4),
        "held_piece_xy": list(held_piece),
        "move_delay_ms": move_delay_ms,
        "action_delay_ms": action_delay_ms,
        "delay_variance_percent": delay_variance_percent,
    }
    
    # Display results
    print("\n" + "=" * 60)
    print("       CALIBRATION COMPLETE!")
    print("=" * 60)
    print("\nCaptured coordinates:\n")
    print(f"  Board Top-Left:     ({board_top_left[0]}, {board_top_left[1]})")
    print(f"  Board Bottom-Right: ({board_bottom_right[0]}, {board_bottom_right[1]})")
    print(f"  Next Piece #1:      ({next_piece_0[0]}, {next_piece_0[1]})")
    print(f"  Next Piece #5:      ({next_piece_4[0]}, {next_piece_4[1]})")
    print(f"  Held Piece:         ({held_piece[0]}, {held_piece[1]})")
    print(f"\nDelay settings:")
    print(f"  Move Delay:         {move_delay_ms}ms")
    print(f"  Action Delay:       {action_delay_ms}ms")
    print(f"  Delay Variance:     {delay_variance_percent}%")
    
    print("\n" + "-" * 60)
    print("COPY-PASTE READY CONFIGURATION:")
    print("-" * 60)
    print(f"""
bot = TetrioBot(
    screen_offset={tuple(screen_offset)},
    screen_resolution={tuple(screen_resolution)},
    board_top_left={tuple(board_top_left)},
    board_bottom_right={tuple(board_bottom_right)},
    next_piece_xy_0={tuple(next_piece_0)},
    next_piece_xy_4={tuple(next_piece_4)},
    held_piece_xy={tuple(held_piece)},
    pruning_moves=5,
    pruning_breadth=5,
    mp=16
)
""")
    
    # Save to config file
    save_config(config)
    
    print("\nYou can also run the bot with the saved config using:")
    print("  python bot.py --use-config\n")
    
    return config


# keybinds
rotate_clockwise_key = 'x'
rotate_180_key = 'a'
rotate_counterclockwise_key = 'z'
hold_key = 'c'
move_left_key = 'left'
move_right_key = 'right'
drop_key = 'space'
wait_time = 0.03
soft_drop_delay = 0.1
key_delay = 0.01
# Game Settings - DAS 40ms, ARR 0ms, SDF max, lowest graphic


def get_delay_with_variance(base_delay_ms, variance_percent):
    """Calculate delay with random variance.
    
    Args:
        base_delay_ms: Base delay in milliseconds
        variance_percent: Variance percentage (e.g., 20 for ±20%)
    
    Returns:
        Delay in seconds (for use with time.sleep)
    """
    if base_delay_ms <= 0:
        return 0
    variance = variance_percent / 100.0
    actual_delay_ms = base_delay_ms * (1 + random.uniform(-variance, variance))
    return max(0, actual_delay_ms / 1000.0)  # Convert to seconds


class TetrioBot:
    def __init__(
        self,
        screen_offset,
        screen_resolution,
        board_top_left,
        board_bottom_right,
        next_piece_xy_0,
        next_piece_xy_4,
        held_piece_xy,
        pruning_moves,
        pruning_breadth,
        mp,
        move_delay_ms=DEFAULT_MOVE_DELAY_MS,
        action_delay_ms=DEFAULT_ACTION_DELAY_MS,
        delay_variance_percent=DEFAULT_DELAY_VARIANCE_PERCENT
    ):
        self.screen_offset = screen_offset
        self.screen_resolution = screen_resolution
        self.board_top_left = board_top_left
        self.board_bottom_right = board_bottom_right

        x0, y0 = next_piece_xy_0
        x4, y4 = next_piece_xy_4
        self.next_piece_xy = (
            next_piece_xy_0,
            ((x0+x4)//2, y0 + math.floor(((y4-y0)/4)*1)),
            ((x0+x4)//2, y0 + math.floor(((y4-y0)/4)*2)),
            ((x0+x4)//2, y0 + math.floor(((y4-y0)/4)*3)),
            next_piece_xy_4
        )
        pixel_area = (y4 - y0) // NUM_COL
        self.pixel_area_half = pixel_area // 2
        self.held_piece_xy = held_piece_xy

        self.pruning_moves = pruning_moves
        self.pruning_breadth = pruning_breadth
        self.mp_pool = Pool(processes=mp) if mp > 1 else None
        
        # Delay settings
        self.move_delay_ms = move_delay_ms
        self.action_delay_ms = action_delay_ms
        self.delay_variance_percent = delay_variance_percent

        self.screen_image = Image()
        self.refresh_screen_image()

    def refresh_screen_image(self):
        # ImageGrab.grab is too heavy. We only make 1 whole screenshot and crop from this only screenshot
        self.screen_image = ImageGrab.grab(
            bbox=(
                self.screen_offset[0],
                self.screen_offset[1],
                self.screen_offset[0] + self.screen_resolution[0],
                self.screen_offset[1] + self.screen_resolution[1],
            ),
            all_screens=True
        )

    def get_next_pieces(self):
        # image.save("board.png")
        result = []
        for x, y in self.next_piece_xy:
            target_colors = np.array(self.screen_image.crop((
                x - self.pixel_area_half,
                y - self.pixel_area_half,
                x + self.pixel_area_half,
                y + self.pixel_area_half
            )))
            target_colors = target_colors.reshape(target_colors.shape[0] * target_colors.shape[1], target_colors.shape[2]).astype(np.int32)
            closest_color = (0, 0, 0)
            min_diff = float('inf')
            for tc in target_colors:
                for c in colors:
                    diff = (c[0] - tc[0]) * (c[0] - tc[0]) + (c[1] - tc[1]) * (c[1] - tc[1]) + (c[2] - tc[2]) * (c[2] - tc[2])
                    if diff < min_diff:
                        min_diff = diff
                        closest_color = c
                    if min_diff < 400:
                        break
            result.append(colors_name[colors.index(closest_color)])
        return result

    def get_held_piece(self):
        x, y = self.held_piece_xy
        image = self.screen_image.crop((
            x - self.pixel_area_half,
            y - self.pixel_area_half,
            x + self.pixel_area_half,
            y + self.pixel_area_half
        ))
        # image.save("board.png")
        target_colors = np.array(image)
        target_colors = target_colors.reshape(target_colors.shape[0] * target_colors.shape[1], target_colors.shape[2]).astype(np.int32)

        # find the closest color in target_colors that is in colors
        closest_color = (0, 0, 0)
        min_diff = float('inf')
        for target_color in target_colors:
            for color in colors:
                diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(color, target_color)))
                if diff < min_diff:
                    min_diff = diff
                    closest_color = color
                if min_diff < 20:
                    return colors_name[colors.index(closest_color)]
        return None

    def get_tetris_board(self):
        board_image = self.screen_image.crop((
            self.board_top_left[0],
            self.board_top_left[1],
            self.board_bottom_right[0],
            self.board_bottom_right[1]
        )).convert('L')
        # board_image.save("board.png")
        board = np.zeros((NUM_ROW, NUM_COL), dtype=np.int32)
        block_width = board_image.width / NUM_COL
        block_height = board_image.height / NUM_ROW

        for row in reversed(range(NUM_ROW)):
            empty_row = True
            for col in range(NUM_COL):
                total_darkness = 0
                num_pixels = 0
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        x = math.floor(col * block_width + block_width / 2) + dx
                        y = math.floor(row * block_height + block_height / 2) + dy
                        pixel_value = board_image.getpixel((x, y))
                        total_darkness += pixel_value
                        num_pixels += 1
                avg_darkness = total_darkness / num_pixels

                if avg_darkness < 30:
                    board[NUM_ROW - row - 1][col] = 0
                else:
                    empty_row = False
                    board[NUM_ROW - row - 1][col] = 1
            if empty_row:
                break
        return board

    def place_piece(self, best_position, rotations, need_hold):
        """Place a piece with configurable delays for more human-like inputs."""
        move_delay = lambda: get_delay_with_variance(self.move_delay_ms, self.delay_variance_percent)
        action_delay = lambda: get_delay_with_variance(self.action_delay_ms, self.delay_variance_percent)
        
        if need_hold:
            keyboard.press(hold_key)
            keyboard.release(hold_key)
            time.sleep(action_delay())
        if rotations[0] != 0:
            match rotations[0]:
                case 1:
                    key = rotate_clockwise_key
                case 2:
                    key = rotate_180_key
                case 3:
                    key = rotate_counterclockwise_key
                case _:
                    raise NotImplementedError
            keyboard.press(key)
            keyboard.release(key)
            time.sleep(action_delay())

        # press left arrow or right arrow to move to position
        if best_position < 3:
            for i in range(3 - best_position):
                keyboard.press(move_left_key)
                keyboard.release(move_left_key)
                time.sleep(move_delay())
        elif best_position > 3:
            for i in range(best_position - 3):
                keyboard.press(move_right_key)
                keyboard.release(move_right_key)
                time.sleep(move_delay())
        if len(rotations) > 1:
            keyboard.press('down')
            time.sleep(soft_drop_delay)
            for rot in rotations[1:]:
                match rot:
                    case 1:
                        key = rotate_clockwise_key
                    case 3:
                        key = rotate_counterclockwise_key
                    case 11:
                        key = move_left_key
                    case 12:
                        key = move_right_key
                    case _:
                        raise NotImplementedError
                keyboard.press(key)
                keyboard.release(key)
                time.sleep(move_delay())
            keyboard.release('down')
        # press space to drop piece
        keyboard.press('space')
        keyboard.release('space')
        time.sleep(action_delay())

    def run(self):
        combo = 0
        b2b = 0

        print("TetrioBot started. Waiting for game...", flush=True)
        last_next_pieces = self.get_next_pieces()
        print(f"Initial next pieces detected: {last_next_pieces}", flush=True)
        expected_board = np.zeros((NUM_ROW, NUM_COL), dtype=np.int32)
        # for _ in range(100):
        poll_count = 0
        while True:
            self.refresh_screen_image()
            next_pieces = self.get_next_pieces()
            while next_pieces == last_next_pieces:
                poll_count += 1
                if poll_count % 100 == 0:
                    print(f"Polling for piece change... ({poll_count} iterations, current: {next_pieces})", flush=True)
                time.sleep(wait_time)
                self.refresh_screen_image()
                next_pieces = self.get_next_pieces()
            poll_count = 0

            current_piece = last_next_pieces[0]
            last_next_pieces = next_pieces

            current_board = self.get_tetris_board()
            if not np.all(np.equal(current_board, expected_board)):
                print("Unexpected board", flush=True)
            held_piece = self.get_held_piece()
            if held_piece is None:
                print("Held is None!!!", flush=True)
                keyboard.press(hold_key)
                keyboard.release(hold_key)
                time.sleep(key_delay)
                continue
            t1 = time.time()
            score, (position, rotations, need_hold, combo, b2b, expected_board) = find_best_move(
                current_board, current_piece, next_pieces, held_piece, combo, b2b,
                self.pruning_moves,
                self.pruning_breadth,
                # mp_pool=None,
                mp_pool=self.mp_pool,
            )
            t2 = time.time()

            print(f"score: {round(score):6}   b2b: {b2b:2}    time: {t2-t1}", flush=True)
            if t2 - t1 < wait_time:
                time.sleep(wait_time - t2 + t1)

            if score < -50000:
                continue
            if need_hold:
                if held_piece is None:
                    current_piece = next_pieces[0]
                else:
                    current_piece = held_piece
            if current_piece in "SZI" and rotations[0] == 3:
                best_piece_pos_rot = tetris_pieces[current_piece][1]
            else:
                best_piece_pos_rot = tetris_pieces[current_piece][rotations[0]]
            # add offset depending on padded zeros on the left side of axis 1 only
            offset = 0
            for i in range(best_piece_pos_rot.shape[1]):
                if not any(best_piece_pos_rot[:, i]):
                    offset += 1
                else:
                    break
            # time.sleep(3)
            self.place_piece(position - offset, rotations, need_hold)
            # time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description='TetrioBot - An AI player for TETR.IO')
    parser.add_argument('--calibrate', action='store_true',
                        help='Run the calibration wizard to configure screen coordinates')
    parser.add_argument('--use-config', action='store_true',
                        help='Use saved configuration from config.json')
    parser.add_argument('--pruning-moves', type=int, default=5,
                        help='Number of moves for pruning (default: 5)')
    parser.add_argument('--pruning-breadth', type=int, default=5,
                        help='Breadth for pruning (default: 5)')
    parser.add_argument('--mp', type=int, default=16,
                        help='Number of multiprocessing workers (default: 16)')
    parser.add_argument('--delay', type=int, default=None,
                        help='Override move delay in milliseconds (e.g., --delay 50)')
    parser.add_argument('--action-delay', type=int, default=None,
                        help='Override action delay in milliseconds')
    parser.add_argument('--delay-variance', type=int, default=None,
                        help='Override delay variance percentage')
    
    args = parser.parse_args()
    
    if args.calibrate:
        run_calibration_wizard()
        return
    
    # Load configuration
    if args.use_config:
        config = load_config()
        if config is None:
            print("Error: No config.json found. Run with --calibrate first.")
            return
        
        # Apply CLI overrides for delay settings
        move_delay_ms = args.delay if args.delay is not None else config.get('move_delay_ms', DEFAULT_MOVE_DELAY_MS)
        action_delay_ms = args.action_delay if args.action_delay is not None else config.get('action_delay_ms', DEFAULT_ACTION_DELAY_MS)
        delay_variance_percent = args.delay_variance if args.delay_variance is not None else config.get('delay_variance_percent', DEFAULT_DELAY_VARIANCE_PERCENT)
        
        print(f"Loaded configuration from {CONFIG_FILE}")
        print(f"Delay settings: move={move_delay_ms}ms, action={action_delay_ms}ms, variance={delay_variance_percent}%")
        bot = TetrioBot(
            screen_offset=tuple(config['screen_offset']),
            screen_resolution=tuple(config['screen_resolution']),
            board_top_left=tuple(config['board_top_left']),
            board_bottom_right=tuple(config['board_bottom_right']),
            next_piece_xy_0=tuple(config['next_piece_xy_0']),
            next_piece_xy_4=tuple(config['next_piece_xy_4']),
            held_piece_xy=tuple(config['held_piece_xy']),
            pruning_moves=args.pruning_moves,
            pruning_breadth=args.pruning_breadth,
            mp=args.mp,
            move_delay_ms=move_delay_ms,
            action_delay_ms=action_delay_ms,
            delay_variance_percent=delay_variance_percent
        )
    else:
        # Use default/hardcoded values
        # Note: These values are based on a secondary-screen which has a TETR.IO window title bar(22px) but no windows-taskbar.
        #       If you have only 1 monitor, you may hide your windows-taskbar or measure the values for your own setting.
        
        # Apply CLI overrides for delay settings or use defaults
        move_delay_ms = args.delay if args.delay is not None else DEFAULT_MOVE_DELAY_MS
        action_delay_ms = args.action_delay if args.action_delay is not None else DEFAULT_ACTION_DELAY_MS
        delay_variance_percent = args.delay_variance if args.delay_variance is not None else DEFAULT_DELAY_VARIANCE_PERCENT
        
        print(f"Delay settings: move={move_delay_ms}ms, action={action_delay_ms}ms, variance={delay_variance_percent}%")
        bot = TetrioBot(
            # screen_offset=(0, 0),  # most common case
            screen_offset=(-1920, 0),
            screen_resolution=(1920, 1080),
            board_top_left=(787, 220),
            board_bottom_right=(1133, 899),
            next_piece_xy_0=(1260, 300),
            next_piece_xy_4=(1260, 721),
            held_piece_xy=(691, 300),
            pruning_moves=args.pruning_moves,
            pruning_breadth=args.pruning_breadth,
            mp=args.mp,
            move_delay_ms=move_delay_ms,
            action_delay_ms=action_delay_ms,
            delay_variance_percent=delay_variance_percent
        )
    
    time.sleep(1)
    bot.run()


if __name__ == "__main__":
    main()
