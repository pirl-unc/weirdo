def f(*, first_arg : int = 1, second_arg : bool = False):
    print(first_arg)
    print(second_arg)


if __name__ == "__main__":
    import defopt
    defopt.run(f)
