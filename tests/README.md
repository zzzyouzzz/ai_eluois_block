# Game framework tests

`test_game.c` includes `eluose.c` with its application `main` renamed, so every
test calls the real `place_block` and `clear_lines` functions.

Run from the repository root in PowerShell:

```powershell
gcc -std=c11 -Wall -Wextra -pedantic tests\test_game.c -lm -o tests\test_game.exe
.\tests\test_game.exe
gcc -std=c11 -Wall -Wextra -pedantic tests\test_training_update.c -lm -o tests\test_training_update.exe
.\tests\test_training_update.exe
gcc -std=c11 -Wall -Wextra -pedantic tests\test_extract_action_features.c -lm -o tests\test_extract_action_features.exe
.\tests\test_extract_action_features.exe
```

An alternate test-data file can be passed as the first argument.

## Test-data format

Every case has the same three sections:

```text
CASE case_name
BEFORE
EMPTY
ACTIONS 1
O 0 0 0
EXPECTED_SCORE 0
AFTER
<20 board rows, from top to bottom>
END
```

1. `BEFORE` is the starting board. Use `EMPTY` for a blank board; otherwise
   provide exactly 20 rows of 10 cells, from board row 19 down to row 0.
2. `ACTIONS` is the number of placements, followed by one `shape rotation x y`
   line per block. Shape names are `I`, `T`, `O`, `J`, `L`, `S`, and `Z`.
   The origin `(x, y)` follows the game board's bottom-left coordinate system.
3. `EXPECTED_SCORE` is the total score for the action list. `AFTER` contains
   the expected final board, using the same `EMPTY` or 20-row format.

Use `O` for an occupied cell and `.` for an empty cell. A rejected placement
returns a negative score; the driver does not add it to the total or clear
lines, so invalid-placement cases can use the same format.
Lines beginning with `#` and blank lines are ignored. Test data is the only
place to add or change scenarios; the driver applies each placement, clears
completed rows, accumulates scores, and compares all of `board[25][15]`.

## Coverage

The 30 cases cover:

- All 19 rotations enabled by `block_r`, with expected board pictures for each
  piece orientation.
- Zero through four lines cleared by a placement, checking scores 0, 100, 300,
  500, and 800 respectively.
- A cumulative six-line sequence: a vertical I clears four lines, then an O
  clears two more; expected total score is 1100.
- Horizontal and vertical bounds, collision, invalid shape, and invalid
  rotation. These cases verify that rejected placements leave the board
  unchanged.

The expected boards are explicit data, not reconstructed by the test driver,
and comparisons include all 25 rows and 15 columns of the board array.

## Training update tests

`test_training_update.c` includes `eluose.c` with the application entry point
renamed, then directly invokes `update_dqn_batch` with hand-built replay
transitions.

The cases verify:

- Terminal targets use only the immediate reward, even when next-state
  features are populated.
- Nonterminal targets bootstrap from the target weights.
- Large TD errors have a clipped Huber gradient.

These are focused update tests; they do not run a complete training episode or
inspect CSV trace formatting.

## Action feature tests

`test_extract_action_features.c` loops over `feature_cases.cases`. Each case
contains a 20-row board picture (`O` existing blocks, `X` the current piece,
`.` empty), one placement, whether feature extraction should succeed, and six
named expected values. The picture shows the piece already placed. The driver
checks that the `X` cells match the action's shape, rotation and coordinates,
uses only `O` cells as the input board, invokes `extract_action_features`,
checks each feature, and verifies that the temporary placement leaves the
input board unchanged. An alternate data file can be passed as the first
argument.

The Chinese feature labels in each case mean:

| Label | Feature meaning |
| --- | --- |
| `高度` | `y + block_height - 1`, the placed piece's top row |
| `消行特征` | Number of full rows times the placed piece's cells in those rows |
| `行方向变化` | Horizontal occupied/empty transitions across each board row |
| `列方向变化` | Vertical occupied/empty transitions down each board column |
| `洞数` | Empty cells below an occupied cell in the same column |
| `井深` | For each column, accumulated depth of empty cells whose horizontal neighbors are occupied; the running depth resets at an occupied cell |

Cases cover an empty board, completing one line with an O, a buried hole, a
well between walls, a vertical I over an uneven stack, a T above a terraced
board with holes and wells, and stacked placements with nonzero height. For
line-clear cases, geometry features are expected from the board after rows are
removed.
