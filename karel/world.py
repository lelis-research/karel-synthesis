# Adapted from https://github.com/bunelr/GandRL_for_NPS/blob/master/karel/world.py

import numpy as np

MAX_API_CALLS = 1e5
MAX_MARKERS_PER_SQUARE = 101

class World:
    # Function: Init
    # --------------
    # Creates a world from a json object. The json
    # must specify:
    # - rows and cols
    # - heroRow, heroCol and heroDir
    # - blocked cells
    # - markers.
    # See tasks/cs106a for examples
    def __init__(self, rows: int, cols: int, heroRow: int, heroCol: int,
                 heroDir: str, blocked: np.matrix, markers: np.matrix):
        self.numAPICalls: int = 0
        self.crashed: bool = False
        self.rows = rows
        self.cols = cols
        self.heroRow = heroRow
        self.heroCol = heroCol
        self.heroDir = heroDir
        self.blocked = blocked
        self.markers = markers

    @classmethod
    def from_json(cls, json_object):
        rows = json_object['rows']
        cols = json_object['cols']
        hero = json_object['hero'].split(':')
        heroRow = int(hero[0])
        heroCol = int(hero[1])
        heroDir = hero[2]

        blocked = np.zeros((rows, cols))
        if json_object['blocked'] != '':
            for coord in json_object['blocked'].split(' '):
                coord_split = coord.split(':')
                r = int(coord_split[0])
                c = int(coord_split[1])
                blocked[r][c] = 1 # For some reason, the original program uses rows - r - 1

        markers = np.zeros((rows, cols))
        if json_object['markers'] != '':
            for coord in json_object['markers'].split(' '):
                coord_split = coord.split(':')
                r = int(coord_split[0])
                c = int(coord_split[1])
                n = int(coord_split[2])
                markers[r][c] = n

        return cls(rows, cols, heroRow, heroCol, heroDir, blocked, markers)

    # Function: Equals
    # ----------------
    # Checks if two worlds are equal. Does a deep check.
    def __eq__(self, other: "World") -> bool:
        if self.heroRow != other.heroRow: return False
        if self.heroCol != other.heroCol: return False
        if self.heroDir != other.heroDir: return False
        if self.crashed != other.crashed: return False
        return self.equal_makers(other)

    def __ne__(self, other: "World") -> bool:
        return not (self == other)

    def hamming_dist(self, other: "World") -> int:
        dist = 0
        if self.heroRow != other.heroRow: dist += 1
        if self.heroCol != other.heroCol: dist += 1
        if self.heroDir != other.heroDir: dist += 1
        if self.crashed != other.crashed: dist += 1
        dist += np.sum(self.markers != other.markers)
        return dist

    @classmethod
    def from_string(cls, worldStr: str):
        lines = worldStr.replace('|', '').split('\n')
        lines.reverse()
        heroRow = None
        heroCol = None
        heroDir = None
        rows = len(lines)
        cols = len(lines[0])
        blocked = np.zeros((rows, cols))
        markers = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                if lines[r][c] == '*':
                    blocked[r][c] = 1
                elif lines[r][c] == 'M': # TODO: could also be a number
                    markers[r][c] = 1
                elif lines[r][c] == '>':
                    heroRow = r
                    heroCol = c
                    heroDir = 'east'
                elif lines[r][c] == '^':
                    heroRow = r
                    heroCol = c
                    heroDir = 'north'
                elif lines[r][c] == '<':
                    heroRow = r
                    heroCol = c
                    heroDir = 'west'
                elif lines[r][c] == 'v':
                    heroRow = r
                    heroCol = c
                    heroDir = 'south'
        if heroRow == None:
            raise Exception('No hero found in map')
        return cls(rows, cols, heroRow, heroCol, heroDir, blocked, markers)

    # Function: Equal Markers
    # ----------------
    # Are the markers the same in these two worlds?
    def equal_makers(self, other: "World") -> bool:
        return (self.markers == other.markers).all()

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
                if self.blocked[r][c] == 1:
                    blocked.append("{0}:{1}".format(r, c))
                if self.hero_at_pos(r, c):
                    hero.append("{0}:{1}:{2}".format(r, c, self.heroDir))
                if self.markers[r][c] > 0:
                    markers.append("{0}:{1}:{2}".format(r, c, int(self.markers[r][c])))

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
        for r in range(self.rows-1, -1, -1):
            rowStr = '|'
            for c in range(0, self.cols):
                if self.blocked[r][c] == 1 :
                    rowStr += '*'
                elif self.hero_at_pos(r, c):
                    rowStr += self.get_hero_char()
                elif self.markers[r][c] > 0:
                    numMarkers = int(self.markers[r][c])
                    if numMarkers > 9: rowStr += 'M'
                    else: rowStr += str(numMarkers)
                else:
                    rowStr += ' '
            worldStr += rowStr + '|'
            if(r != 0): worldStr += '\n'
        return worldStr

    # Function: get hero char
    # ------------------
    # Returns a char that represents the hero (based on
    # the heros direction).
    def get_hero_char(self) -> str:
        if(self.heroDir == 'north'): return '^'
        if(self.heroDir == 'south'): return 'v'
        if(self.heroDir == 'east'): return '>'
        if(self.heroDir == 'west'): return '<'
        raise("invalid dir")

    # Function: get hero dir value
    # ------------------
    # Returns a numeric representation of the hero direction.
    def get_hero_dir_value(self) -> int:
        if(self.heroDir == 'north'): return 1
        if(self.heroDir == 'south'): return 3
        if(self.heroDir == 'east'): return 2
        if(self.heroDir == 'west'): return 4
        raise("invalid dir")

    @classmethod
    def undo_hero_dir_value(cls, value: int) -> str:
        if(value == 1): return 'north'
        if(value == 3): return 'south'
        if(value == 2): return 'east'
        if(value == 4): return 'west'
        raise('invalid dir')

    # Function: hero at pos
    # ------------------
    # Returns true or false if the hero is at a given location.
    def hero_at_pos(self, r: int, c: int) -> bool:
        if self.heroRow != r: return False
        if self.heroCol != c: return False
        return True

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
        if self.blocked[r][c] != 0:
            return False
        return True

    # Function: front is clear
    # ------------------
    # Returns if the hero is facing an open cell.
    def front_is_clear(self) -> bool:
        if self.crashed: return
        self.note_api_call()
        if(self.heroDir == 'north'):
            return self.is_clear(self.heroRow + 1, self.heroCol)
        elif(self.heroDir == 'south'):
            return self.is_clear(self.heroRow - 1, self.heroCol)
        elif(self.heroDir == 'east'):
            return self.is_clear(self.heroRow, self.heroCol + 1)
        elif(self.heroDir == 'west'):
            return self.is_clear(self.heroRow, self.heroCol - 1)


    # Function: left is clear
    # ------------------
    # Returns if the left of the hero is an open cell.
    def left_is_clear(self) -> bool:
        if self.crashed: return
        self.note_api_call()
        if(self.heroDir == 'north'):
            return self.is_clear(self.heroRow, self.heroCol - 1)
        elif(self.heroDir == 'south'):
            return self.is_clear(self.heroRow, self.heroCol + 1)
        elif(self.heroDir == 'east'):
            return self.is_clear(self.heroRow + 1, self.heroCol)
        elif(self.heroDir == 'west'):
            return self.is_clear(self.heroRow - 1, self.heroCol)


    # Function: right is clear
    # ------------------
    # Returns if the right of the hero is an open cell.
    def right_is_clear(self) -> bool:
        if self.crashed: return
        self.note_api_call()
        if(self.heroDir == 'north'):
            return self.is_clear(self.heroRow, self.heroCol + 1)
        elif(self.heroDir == 'south'):
            return self.is_clear(self.heroRow, self.heroCol - 1)
        elif(self.heroDir == 'east'):
            return self.is_clear(self.heroRow - 1, self.heroCol)
        elif(self.heroDir == 'west'):
            return self.is_clear(self.heroRow + 1, self.heroCol)


    # Function: markers present
    # ------------------
    # Returns if there is one or more markers present at
    # the hero pos
    def markers_present(self) -> bool:
        self.note_api_call()
        return self.markers[self.heroRow][self.heroCol] > 0

    # Function: pick marker
    # ------------------
    # If there is a marker, pick it up. Otherwise crash the
    # program.
    def pick_marker(self) -> None:
        if not self.markers_present():
            self.crashed = True
        else:
            self.markers[self.heroRow][self.heroCol] -= 1
        self.note_api_call()

    # Function: put marker
    # ------------------
    # Adds a marker to the current location (can be > 1)
    def put_marker(self) -> None:
        self.markers[self.heroRow][self.heroCol] += 1
        if self.markers[self.heroRow][self.heroCol] > MAX_MARKERS_PER_SQUARE:
            self.crashed = True
        self.note_api_call()

    # Function: move
    # ------------------
    # Move the hero in the direction she is facing. If the
    # world is not clear, the hero's move is undone.
    def move(self) -> None:
        if self.crashed: return
        newRow = self.heroRow
        newCol = self.heroCol
        if(self.heroDir == 'north'): newRow = self.heroRow + 1
        if(self.heroDir == 'south'): newRow = self.heroRow - 1
        if(self.heroDir == 'east'): newCol = self.heroCol + 1
        if(self.heroDir == 'west'): newCol = self.heroCol - 1
        if not self.is_clear(newRow, newCol):
            self.crashed = True
        if not self.crashed:
            self.heroCol = newCol
            self.heroRow = newRow
        self.note_api_call()

    # Function: turn left
    # ------------------
    # Rotates the hero counter clock wise.
    def turn_left(self) -> None:
        if self.crashed: return
        if(self.heroDir == 'north'): self.heroDir = 'west'
        elif(self.heroDir == 'south'): self.heroDir = 'east'
        elif(self.heroDir == 'east'): self.heroDir = 'north'
        elif(self.heroDir == 'west'): self.heroDir = 'south'
        self.note_api_call()

    # Function: turn left
    # ------------------
    # Rotates the hero clock wise.
    def turn_right(self) -> None:
        if self.crashed: return
        if(self.heroDir == 'north'): self.heroDir = 'east'
        elif(self.heroDir == 'south'): self.heroDir = 'west'
        elif(self.heroDir == 'east'): self.heroDir = 'south'
        elif(self.heroDir == 'west'): self.heroDir = 'north'
        self.note_api_call()

    # Function: note api call
    # ------------------
    # To catch infinite loops, we limit the number of API calls.
    # If the num api calls exceeds a max, the program is crashed.
    def note_api_call(self) -> None:
        self.numAPICalls += 1
        if self.numAPICalls > MAX_API_CALLS:
            self.crashed = True