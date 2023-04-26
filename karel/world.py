# Adapted from https://github.com/bunelr/GandRL_for_NPS/blob/master/karel/world.py

import os
import numpy as np

from config import Config

MAX_API_CALLS = 10000
MAX_MARKERS_PER_SQUARE = 10

STATE_TABLE = {
    0: 'Karel facing North',
    1: 'Karel facing East',
    2: 'Karel facing South',
    3: 'Karel facing West',
    4: 'Wall',
    5: '0 marker',
    6: '1 marker',
    7: '2 markers',
    8: '3 markers',
    9: '4 markers',
    10: '5 markers',
    11: '6 markers',
    12: '7 markers',
    13: '8 markers',
    14: '9 markers',
    15: '10 markers'
}

ACTION_TABLE = {
    0: 'Move',
    1: 'Turn left',
    2: 'Turn right',
    3: 'Pick up a marker',
    4: 'Put a marker'
}

class World:
    def __init__(self, s: np.ndarray = None):
        self.numAPICalls: int = 0
        self.crashed: bool = False
        self.crashable = Config.env_is_crashable
        self.leaps_behavior = Config.env_enable_leaps_behaviour 
        if s is not None:
            self.s = s.astype(bool)
        self.assets: dict[str, np.ndarray] = {}
        x, y, d = np.where(self.s[:, :, :4] > 0)
        self.hero_pos = [x[0], y[0], d[0]]
        self.markers_grid = self.s[:, :, 5:].argmax(axis=2)

    def get_state(self):
        return self.s

    @property
    def rows(self):
        return self.s.shape[0]

    @property
    def cols(self):
        return self.s.shape[1]

    # def state_rot90(self, n_times: int = 1):
    #     self.s = np.rot90(self.s, k=n_times)
    #     hero_r, hero_c, hero_d = self.get_hero_loc()
    #     new_d = (hero_d - n_times) % 4
    #     self.s[hero_r, hero_c, hero_d] = False
    #     self.s[hero_r, hero_c, new_d] = True

    # def center_and_pad(self, n_rows: int, n_cols: int):
    #     _, _, hero_d = self.get_hero_loc()
    #     self.state_rot90(hero_d)
    #     hero_r, hero_c, _ = self.get_hero_loc()
    #     # pad here

    def get_hero_loc(self):
        return self.hero_pos

    def set_new_state(self, s: np.ndarray = None):
        self.s = s.astype(bool)

    @classmethod
    def from_json(cls, json_object):
        rows = json_object['rows']
        cols = json_object['cols']
        s = np.zeros((rows, cols, 16), dtype=bool)
        hero = json_object['hero'].split(':')
        heroRow = int(hero[0])
        heroCol = int(hero[1])
        heroDir = World.get_dir_number(hero[2])
        s[rows - heroRow - 1,heroCol,heroDir] = True

        if json_object['blocked'] != '':
            for coord in json_object['blocked'].split(' '):
                coord_split = coord.split(':')
                r = int(coord_split[0])
                c = int(coord_split[1])
                s[rows - r - 1,c,4] = True # For some reason, the original program uses rows - r - 1

        s[:,:,5] = True
        if json_object['markers'] != '':
            for coord in json_object['markers'].split(' '):
                coord_split = coord.split(':')
                r = int(coord_split[0])
                c = int(coord_split[1])
                n = int(coord_split[2])
                s[rows - r - 1,c,n+5] = True
                s[rows - r - 1,c,5] = False

        return cls(s)

    # Function: Equals
    # ----------------
    # Checks if two worlds are equal. Does a deep check.
    def __eq__(self, other: "World") -> bool:
        if self.crashed != other.crashed: return False
        return (self.s == other.s).all()

    def __ne__(self, other: "World") -> bool:
        return not (self == other)

    # def hamming_dist(self, other: "World") -> int:
    #     dist = 0
    #     if self.heroRow != other.heroRow: dist += 1
    #     if self.heroCol != other.heroCol: dist += 1
    #     if self.heroDir != other.heroDir: dist += 1
    #     if self.crashed != other.crashed: dist += 1
    #     dist += np.sum(self.markers != other.markers)
    #     return dist

    @classmethod
    def from_string(cls, worldStr: str):
        lines = worldStr.replace('|', '').split('\n')
        # lines.reverse()
        rows = len(lines)
        cols = len(lines[0])
        s = np.zeros((rows, cols, 16), dtype=bool)
        for r in range(rows):
            for c in range(cols):
                if lines[r][c] == '*':
                    s[r][c][4] = True
                elif lines[r][c] == 'M': # TODO: could also be a number
                    s[r][c][6] = True
                else:
                    s[r][c][5] = True

                if lines[r][c] == '^':
                    s[r][c][0] = True
                elif lines[r][c] == '>':
                    s[r][c][1] = True
                elif lines[r][c] == 'v':
                    s[r][c][2] = True
                elif lines[r][c] == '<':
                    s[r][c][3] = True
        return cls(s)

    # Function: Equal Markers
    # ----------------
    # Are the markers the same in these two worlds?
    def equal_makers(self, other: "World") -> bool:
        return (self.s[:,:,5:] == other.s[:,:,5:]).all()

    def to_json(self) -> dict:
        obj = {}

        obj['rows'] = self.rows
        obj['cols'] = self.cols
        if self.crashed:
            obj['crashed'] = True
            return obj

        obj['crashed'] = False

        markers = []
        blocked = []
        hero = []
        for r in range(self.rows-1, -1, -1):
            for c in range(0, self.cols):
                if self.s[r][c][4]:
                    blocked.append("{0}:{1}".format(r, c))
                if self.hero_at_pos(r, c):
                    hero.append("{0}:{1}:{2}".format(r, c, self.heroDir))
                if np.sum(self.s[r, c, 6:]) > 0:
                    markers.append("{0}:{1}:{2}".format(r, c, np.sum(self.s[r, c, 6:])))

        obj['markers'] = " ".join(markers)
        obj['blocked'] = " ".join(blocked)
        obj['hero'] = " ".join(hero)

        return obj

    # Function: toString
    # ------------------
    # Returns a string version of the world. Uses a '>'
    # symbol for the hero, a '*' symbol for blocked and
    # in the case of markers, puts the number of markers.
    # If the hero is standing ontop of markers, the num
    # markers is not visible.
    def to_string(self) -> str:
        worldStr = ''
        #worldStr += str(self.heroRow) + ', ' + str(self.heroCol) + '\n'
        if self.crashed: worldStr += 'CRASHED\n'
        hero_r, hero_c, hero_d = self.get_hero_loc()
        for r in range(0, self.rows):
            rowStr = '|'
            for c in range(0, self.cols):
                if self.s[r][c][4] == 1:
                    rowStr += '*'
                elif r == hero_r and c == hero_c:
                    rowStr += self.get_hero_char(hero_d)
                elif np.sum(self.s[r, c, 6:]) > 0:
                    num_marker = self.s[r, c, 5:].argmax()
                    if num_marker > 9: rowStr += 'M'
                    else: rowStr += str(num_marker)
                else:
                    rowStr += ' '
            worldStr += rowStr + '|'
            if(r != self.rows-1): worldStr += '\n'
        return worldStr

    def to_image(self) -> np.ndarray:
        grid_size = 100
        if len(self.assets) == 0:
            from PIL import Image
            files = ['agent_0', 'agent_1', 'agent_2', 'agent_3', 'blank', 'marker', 'wall']
            for f in files:
                self.assets[f] = np.array(Image.open(os.path.join('assets', f'{f}.PNG')))

        img = np.ones((self.rows*grid_size, self.cols*grid_size))
        hero_r, hero_c, hero_d = self.get_hero_loc()
        for r in range(self.rows):
            for c in range(self.cols):
                if self.s[r][c][4] == 1:
                    asset = self.assets['wall']
                elif r == hero_r and c == hero_c:
                    if np.sum(self.s[r, c, 6:]) > 0:
                        asset = np.minimum(self.assets[f'agent_{hero_d}'], self.assets['marker'])
                    else:
                        asset = self.assets[f'agent_{hero_d}']
                elif np.sum(self.s[r, c, 6:]) > 0:
                    asset = self.assets['marker']
                else:
                    asset = self.assets['blank']
                img[(r)*grid_size:(r+1)*grid_size, c*grid_size:(c+1)*grid_size] = asset

        return img

    # Function: get hero char
    # ------------------
    # Returns a char that represents the hero (based on
    # the heros direction).
    def get_hero_char(self, dir) -> str:
        if(dir == 0): return '^'
        if(dir == 1): return '>'
        if(dir == 2): return 'v'
        if(dir == 3): return '<'
        raise("invalid dir")

    def get_dir_str(self, dir) -> str:
        if(dir == 0): return 'north'
        if(dir == 1): return 'east'
        if(dir == 2): return 'south'
        if(dir == 3): return 'west'
        raise('invalid dir')

    @staticmethod
    def get_dir_number(dir: str) -> int:
        if(dir == 'north'): return 0
        if(dir == 'east' ): return 1
        if(dir == 'south'): return 2
        if(dir == 'west' ): return 3
        raise('invalid dir')

    # Function: hero at pos
    # ------------------
    # Returns true or false if the hero is at a given location.
    def hero_at_pos(self, r: int, c: int) -> bool:
        row, col, _ = self.get_hero_loc()
        return row == r and col == c

    def is_crashed(self) -> bool:
        return self.crashed

    # Function: is clear
    # ------------------
    # Returns if the (r,c) is a valid and unblocked pos.
    def is_clear(self, r: int, c: int) -> bool:
        if(r < 0 or c < 0):
            return False
        if r >= self.rows or c >= self.cols:
            return False
        return not self.s[r, c, 4]

    # Function: front is clear
    # ------------------
    # Returns if the hero is facing an open cell.
    def front_is_clear(self) -> bool:
        if self.crashed: return
        self.note_api_call()
        r, c, d = self.get_hero_loc()
        if(d == 0):
            return self.is_clear(r - 1, c)
        elif(d == 1):
            return self.is_clear(r, c + 1)
        elif(d == 2):
            return self.is_clear(r + 1, c)
        elif(d == 3):
            return self.is_clear(r, c - 1)


    # Function: left is clear
    # ------------------
    # Returns if the left of the hero is an open cell.
    def left_is_clear(self) -> bool:
        if self.crashed: return
        self.note_api_call()
        r, c, d = self.get_hero_loc()
        if(d == 0):
            return self.is_clear(r, c - 1)
        elif(d == 1):
            return self.is_clear(r - 1, c)
        elif(d == 2):
            return self.is_clear(r, c + 1)
        elif(d == 3):
            return self.is_clear(r + 1, c)


    # Function: right is clear
    # ------------------
    # Returns if the right of the hero is an open cell.
    def right_is_clear(self) -> bool:
        if self.crashed: return
        self.note_api_call()
        r, c, d = self.get_hero_loc()
        if(d == 0):
            return self.is_clear(r, c + 1)
        elif(d == 1):
            return self.is_clear(r + 1, c)
        elif(d == 2):
            return self.is_clear(r, c - 1)
        elif(d == 3):
            return self.is_clear(r - 1, c)


    # Function: markers present
    # ------------------
    # Returns if there is one or more markers present at
    # the hero pos
    def markers_present(self) -> bool:
        self.note_api_call()
        r, c, _ = self.get_hero_loc()
        return self.markers_grid[r, c] > 0

    # Function: pick marker
    # ------------------
    # If there is a marker, pick it up. Otherwise crash the
    # program.
    def pick_marker(self) -> None:
        r, c, _ = self.get_hero_loc()
        num_marker = self.markers_grid[r, c]
        if num_marker == 0:
            if self.crashable:
                self.crashed = True
        else:
            self.s[r, c, 5 + num_marker] = False
            self.s[r, c, 4 + num_marker] = True
            self.markers_grid[r, c] -= 1
        self.note_api_call()

    # Function: put marker
    # ------------------
    # Adds a marker to the current location (can be > 1)
    def put_marker(self) -> None:
        r, c, _ = self.get_hero_loc()
        num_marker = self.markers_grid[r, c]
        if num_marker == MAX_MARKERS_PER_SQUARE:
            if self.crashable:
                self.crashed = True
        else:
            self.s[r, c, 5 + num_marker] = False
            self.s[r, c, 6 + num_marker] = True
            self.markers_grid[r, c] += 1
        self.note_api_call()

    # Function: move
    # ------------------
    # Move the hero in the direction she is facing. If the
    # world is not clear, the hero's move is undone.
    def move(self) -> None:
        if self.crashed: return
        r, c, d = self.get_hero_loc()
        new_r = r
        new_c = c
        if(d == 0): new_r = new_r - 1
        if(d == 1): new_c = new_c + 1
        if(d == 2): new_r = new_r + 1
        if(d == 3): new_c = new_c - 1
        if not self.is_clear(new_r, new_c) and self.crashable:
            self.crashed = True
        if not self.crashed and self.is_clear(new_r, new_c):
            self.s[r, c, d] = False
            self.s[new_r, new_c, d] = True
            self.hero_pos = [new_r, new_c, d]
        elif self.leaps_behavior:
            self.turn_left()
            self.turn_left()
        self.note_api_call()

    # Function: turn left
    # ------------------
    # Rotates the hero counter clock wise.
    def turn_left(self) -> None:
        if self.crashed: return
        r, c, d = self.get_hero_loc()
        new_d = (d - 1) % 4
        self.s[r, c, d] = False
        self.s[r, c, new_d] = True
        self.hero_pos = [r, c, new_d]
        self.note_api_call()

    # Function: turn left
    # ------------------
    # Rotates the hero clock wise.
    def turn_right(self) -> None:
        if self.crashed: return
        r, c, d = self.get_hero_loc()
        new_d = (d + 1) % 4
        self.s[r, c, d] = False
        self.s[r, c, new_d] = True
        self.hero_pos = [r, c, new_d]
        self.note_api_call()

    # Function: note api call
    # ------------------
    # To catch infinite loops, we limit the number of API calls.
    # If the num api calls exceeds a max, the program is crashed.
    def note_api_call(self) -> None:
        self.numAPICalls += 1
        if self.numAPICalls > MAX_API_CALLS:
            self.crashed = True

    def run_action(self, action: int):
        if action == 0: self.move()
        elif action == 1: self.turn_left()
        elif action == 2: self.turn_right()
        elif action == 3: self.pick_marker()
        elif action == 4: self.put_marker()
        else: raise NotImplementedError()


if __name__ == '__main__':
    world = World.from_string(
        '|  |\n' +
        '| *|\n' +
        '|  |\n' +
        '|  |\n' +
        '| *|\n' +
        '|  |\n' +
        '|  |\n' +
        '| *|\n' +
        '|  |\n' +
        '|^ |'
    )

    print(world.to_string())

    if (world.right_is_clear()):
        world.turn_right()
        world.move()
        world.put_marker()
        world.turn_left()
        world.turn_left()
        world.move()
        world.turn_right()
    while (world.front_is_clear()):
        world.move()
        if (world.right_is_clear()):
            world.turn_right()
            world.move()
            world.put_marker()
            world.turn_left()
            world.turn_left()
            world.move()
            world.turn_right()

    print(world.to_string())