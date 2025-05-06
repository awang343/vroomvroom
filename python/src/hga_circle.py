class CircleSector:
    def __init__(self, point=None):
        self.start = 0
        self.end = 0
        if point is not None:
            self.initialize(point)

    @staticmethod
    def positive_mod(i):
        # Equivalent to (i % 65536 + 65536) % 65536
        # but more efficiently written in Python
        return i & 0xFFFF  # i % 65536 using bitmask

    def initialize(self, point):
        self.start = point
        self.end = point

    def is_enclosed(self, point):
        return self.positive_mod(point - self.start) <= self.positive_mod(self.end - self.start)

    def extend(self, point):
        if not self.is_enclosed(point):
            if self.positive_mod(point - self.end) <= self.positive_mod(self.start - point):
                self.end = point
            else:
                self.start = point

    @staticmethod
    def overlap(sector1, sector2):
        return (
            CircleSector.positive_mod(sector2.start - sector1.start)
            <= CircleSector.positive_mod(sector1.end - sector1.start)
            or
            CircleSector.positive_mod(sector1.start - sector2.start)
            <= CircleSector.positive_mod(sector2.end - sector2.start)
        )
