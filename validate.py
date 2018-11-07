import argparse, random, json

def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("-o", dest="obj_file",
                        help="Object file (input to program)", required=True)
    parser.add_argument("-b", dest="bin_file",
                        help="Bin file (output of program)", required=True)
    args = parser.parse_args()

    with open(args.obj_file) as objfile, open(args.bin_file) as binfile:
        objdata = json.load(objfile)
        bindata = json.load(binfile)

        # Same parameters
        assert(objdata["num_objs"] == bindata["num_objs"])
        assert(objdata["bin_size"] == bindata["bin_size"])
        num_objs = objdata["num_objs"]
        bin_size = objdata["bin_size"]

        # Same objects in the same order
        assert(len(objdata["objs"]) == num_objs)
        assert(len(bindata["objs"]) == num_objs)
        objs = objdata["objs"]
        for i in range(num_objs):
            assert(objs[i] == bindata["objs"][i])

        # Bins are valid
        bins = bindata["bins"]
        for b in bins:
            assert(sum(b) <= bin_size)

        # Bins contain the same objects as objects
        objdict = {}
        for o in objs:
            if(o in objdict):
                objdict[o] += 1
            else:
                objdict[o] = 1
        binobjdict = {}
        for b in bins:
            for o in b:
                if(o in binobjdict):
                    binobjdict[o] += 1
                else:
                    binobjdict[o] = 1
        for size in objdict.keys():
            assert (size in binobjdict and objdict[size] == binobjdict[size]), \
                    "Bins and object differ for size %d" % size
        for size in binobjdict.keys():
            assert (size in objdict and objdict[size] == binobjdict[size]), \
                    "Bins and object differ for size %d" % size

        # indices_by_obj is at least kind of correct
        idx_by_obj = bindata["indices_by_obj"]
        assert(len(idx_by_obj) == num_objs)
        for i in range(num_objs):
            size = objs[i]
            assert(size in bins[idx_by_obj[i]])



if(__name__ == "__main__"):
    main()
