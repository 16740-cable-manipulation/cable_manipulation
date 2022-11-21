class Action:
    def __init__(
        self,
        is_empty,
        pick_coord=[0, 0],
        pick_theta=0.0,
        place_coord=[0, 0],
        place_theta=0.0,
        z=0.0,
    ):
        self.is_empty = is_empty  # empty action does nothing
        self.pick_coord = pick_coord
        self.pick_theta = pick_theta
        self.place_coord = place_coord
        self.place_theta = place_theta
        self.z = z

        self.pick_3d = None
        self.place_3d = None
