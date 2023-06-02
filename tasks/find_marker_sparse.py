from karel.world import World
from tasks.find_marker import FindMarker


class FindMarkerSparse(FindMarker):
        
    def get_reward(self, world_state: World):

        terminated = False
        reward = 0.
        num_markers = world_state.markers_grid.sum()

        if num_markers > 1:
            terminated = True
            reward = -1
        # Check if marker has been picked up
        elif world_state.markers_grid[self.marker_position[0], self.marker_position[1]] == 0:
            terminated = True
            reward = 1
        
        return terminated, reward