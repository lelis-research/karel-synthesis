from .task import Task
from .stair_climber import StairClimber
from .stair_climber_sparse import StairClimberSparse
from .maze import Maze
from .maze_sparse import MazeSparse
from .four_corners import FourCorners
from .four_corners_sparse import FourCornersSparse
from .harvester import Harvester
from .harvester_sparse import HarvesterSparse
from .clean_house import CleanHouse
from .clean_house_sparse import CleanHouseSparse
from .top_off import TopOff
from .top_off_sparse import TopOffSparse
from .door_key import DoorKey
from .find_marker import FindMarker
from .find_marker_sparse import FindMarkerSparse
from .seeder import Seeder
from .seeder_sparse import SeederSparse

def get_task_cls(task_cls_name: str) -> type[Task]:
    task_cls = globals()[task_cls_name]    
    assert issubclass(task_cls, Task)
    return task_cls