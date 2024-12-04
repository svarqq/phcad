import __init__  # noqa

from phcad.data_handling import mvtec_dataset


if __name__ == "__main__":
    res = mvtec_dataset.generate_data(276)
    print(res)
