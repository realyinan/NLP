def n_gram(input_list):
    n_range = 2
    return set(zip(*[input_list[i:] for i in range(2)]))


if __name__ == "__main__":
    print(n_gram([1, 3, 2, 1, 5, 3]))


