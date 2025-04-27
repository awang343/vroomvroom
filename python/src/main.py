import sys
import os

from timer import Timer
from instance import VRPInstance


def main():
    if len(sys.argv) == 1:
        print("Usage: python main.py <file>")
        return

    input_file = sys.argv[1]
    filename = os.path.basename(input_file)
    # print(f"Instance: {input_file}")

    # Start timing the process
    timer = Timer()
    timer.start()

    # Load VRP instance
    instance = VRPInstance(input_file)
    obj_value, optimal, routes = instance.solve()

    # Stop timing
    timer.stop()

    # Output results in JSON-like format
    solution = f'{optimal} {" ".join(["0 "+" ".join([str(r) for r in route]) + " 0" for route in routes])}'
    print(
        f'{{"Instance": "{filename}","Time": {timer.getElapsed():.2f}, "Result": {obj_value}, "Solution": "{solution}"}}'
    )


if __name__ == "__main__":
    main()
