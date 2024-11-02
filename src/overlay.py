from praxis.Utilities import *


def main(cfg: DictConfig) -> None:
    marker_2d_points = perspective_projection(marker_3[0], 500)
    print(marker_2d_points)


if __name__ == "__main__":
    main()


