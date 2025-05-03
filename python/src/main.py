import sys
import os

from timer import Timer
from instance import VRPInstance

from solver_genetics import GeneticSolver
from solver_ipcw import IPCWSolver


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
    solver = GeneticSolver(instance)

    obj_value, optimal, routes = solver.solve()
    routes = [[0] + route + [0] for route in routes]

    # Stop timing
    timer.stop()

    with open(f"./sol/{filename}.sol", "w") as f:
        f.write(f"{obj_value} {optimal}\n")
        for route in routes:
            f.write(f'{" ".join([str(r) for r in route])}\n')

    solution = f'{optimal} {" ".join([" ".join([str(r) for r in route]) for route in routes])}'
    print(
        f'{{"Instance": "{filename}","Time": {timer.getElapsed():.2f}, "Result": {obj_value}, "Solution": "{solution}"}}'
    )


if __name__ == "__main__":
    main()
