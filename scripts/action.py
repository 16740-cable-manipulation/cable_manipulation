import copy


class Action:
    def __init__(
        self,
        is_not_empty,
        pick_coord=[0, 0],
        pick_vec=[0, 0],
        place_coord=[0, 0],
        place_vec=[0, 0],
        z_2d=0.0,
    ):
        self.is_empty = not is_not_empty  # empty action does nothing
        # these are in cv coord
        self.pick_coord = copy.deepcopy(pick_coord)
        self.pick_vec = copy.deepcopy(pick_vec)
        self.place_coord = copy.deepcopy(place_coord)
        self.place_vec = copy.deepcopy(place_vec)
        self.z_2d = z_2d

        # these are np arrays
        self.pick_3d = None
        self.place_3d = None
        self.pick_vec_3d = None
        self.place_vec_3d = None
        self.z = None

    def print(self):
        print(
            f"Action: is_not_empty={not self.is_empty},\n"
            f"pick_coord={self.pick_coord}, place_coord={self.place_coord},\n"
            f"lift_z={self.z}"
        )
