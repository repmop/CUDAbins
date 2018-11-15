import argparse, random, json

def main():
    parser = argparse.ArgumentParser(description="Generate Bin Inputs.")
    parser.add_argument("-f", dest="out_file", metavar="FILE",
                        help="File to write to", required=True)
    parser.add_argument("-n", dest="num_objs", type=int,
                        help="Number of objects", required=True)
    parser.add_argument("-s", dest="bin_size", type=int,
                        help="Bin size", required=True)
    parser.add_argument("-m","--mode", dest="mode", choices=["uniform"],
                        default="uniform", help="Mode for input generation")
    args = parser.parse_args()

    num_objs = args.num_objs
    bin_size = args.bin_size
    objs = [0] * num_objs

    if(args.mode == "uniform"):
        for i in range(num_objs):
            objs[i] = random.randint(1, bin_size)

    inputs = {}
    inputs["num_objs"] = num_objs
    inputs["bin_size"] = bin_size
    inputs["objs"] = objs

    with open(args.out_file, "w") as outfile:
        json.dump(inputs, outfile)


if(__name__ == "__main__"):
    main()
