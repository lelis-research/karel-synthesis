from .task import Task
from .stair_climber import StairClimber

def get_task(task_cls_name: str) -> Task:
    task_cls = globals()[task_cls_name]    
    assert issubclass(task_cls, Task)
    return task_cls()