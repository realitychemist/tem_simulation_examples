"""
This module reads files in .xyz format.  This file can also be run as a script to convert xyz files
into several other formats.  Currently supported formats are csv and cel.

If you are unfamiliar with command line programs, run the command:
    'python xyz-to-cel.py --help'
in a command prompt in the same directory as this file.

This is early-version software!  There are very likely bugs.  Make sure to take a look at your
output files and ensure they make sense.  If you encounter bugs, please open an issue on GitHub.

The comments in this file mention several planned features.  The version of this program hosted on
this repository is unlikely to see feature updates; message the author if you want to inquire about
newer versions.
"""
####################
# Importable Class #
####################


class XYZ:
    """
    A class to contain the data of an .xyz file in an accessible format.

    :Attributes:
        :num_sites (int): Number of atomic sites represented in the object (should be equal to
            len(sites))
        :comment_string (str): An arbitraty comment string
        :sites (list): Contians site information tuples (E, x, y, z) where E is the element
            shortname and x, y, and z are coordinate positions in angstroms
    """

    def __init__(self, num_sites, comment_string, sites):
        """Initialize an XYZ object."""
        self.num_sites = num_sites
        self.comment_string = comment_string
        self.sites = sites


########################
# Importable Functions #
########################


def readXYZ(file):
    """
    Reads in an .xyz formatted file, return an instance of XYZ class.

    :Params:
        :file (file object): An xyz-formatted file object, for example the result of
            open(/path/to/file.xyz); note that this method currently does not sanity check the file
            content, and if it is malformed this method may silently misbehave
    :Returns:
        :XYZ object: An initialized XYZ object containing the information from the file
    """
    # TODO: Sanity checking (file format is as expected) should be implemented
    lines = file.readlines()
    # .xyz files have two lines before the atomic positions are described:
    #   - the first line is the number of atoms described in the file
    #   - the second line is a comment string
    num_sites = int(lines.pop(0))
    comment_string = lines.pop(0)
    # The rest of the lines in the document contain identities and locations
    #  of atoms
    sites = []
    for line in lines:
        cols = line.split()
        # By default the x, y, z coords will be strings, but we want floats
        for i in range(1, 4):
            cols[i] = float(cols[i])
        site = tuple(cols)
        sites.append(site)
    return XYZ(num_sites, comment_string, sites)


def writeXYZ(xyz_obj, outfile):
    """
    Exports an xyz-formatted file.

    :Params:
        :xyz_obj (XYZ object): An initialized XYZ object containing the information to be written
        :outfile (str): The path to the file which will be written, including the file extension
    """
    with open(outfile, 'w') as file:
        file.write(xyz_obj.num_sites, "\n")
        file.write(xyz_obj.comment_string, "\n")
        for site in xyz_obj.sites:
            line = " ".join(site) + "\n"
            file.write(line)


#################
# CMD Mode Only #
#################

def _parse_args(args):
    """
    Command line argument parser.  This is almost always used like...

        >>> opts = _parse_args(sys.argv)

    :Params:
        :args (list): A list of strings to parse into arguments

    """
    parser = argparse.ArgumentParser(
        description="Convert from the .xyz file format.")
    parser.add_argument("infile", nargs="+", default="",
                        help="The path to the input file; currently only supports xyz files as " +
                             "input.  Multiple files can be passed at once for batch conversion.")
    parser.add_argument("--outfile", nargs="+",
                        help="The path to the output file.  If not specified, uses the same " +
                             "path as the input file with a new extension appropriate to " +
                             "--type.  If multuple infiles are specified, do not use this " +
                             "argument.")
    # Add additional filetypes to choices as they're implemented
    parser.add_argument("--type", nargs="+", choices=["cel", "csv"],
                        help="The type of file to output; multiple types may be listed at once, " +
                             "or type can be inferred from file extension of --outfile if " +
                             "provided.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow file overwriting on output if the output file already " +
                             "exists (default behavior is to not allow overwriting).")
    # TODO: figure out how to implement biso in a convenient way (for cel output)
    opts = parser.parse_args()
    return opts


def _write_csv(outfile, xyz_obj, overwrite=False, delim=","):
    """
    Writes out a CSV file based on the data in an XYZ object. See RFC 4180 for a description of
    the csv format.

    :Params:
        :outfile (str): The path to the outfile, including extension
        :xyz_obj (XYZ object): Contains the data to be written
        :overwrite (bool): Disables overwriting if False; should always be passed, but to be safe
            this defaults to False, which should prevent any accidental data loss
        :delim (str): The delimiter string to be used when writing the csv file
    """
    from csv import writer
    # Safety check outfile (no overwriting without permission!)
    _overwrite_check(outfile, overwrite)

    with open(outfile, 'w', newline='') as file:
        csvwriter = writer(file, delimiter=delim)  # This could fail for weird delimiters
        csvwriter.writerow(["Element", "x", "y", "z"])
        for site in xyz_obj.sites:
            csvwriter.writerow(site)


def _write_cel(outfile, xyz_obj, overwrite, deb_dict={}):
    """
    Writes out a CEL file based on the data in an XYZ object.  See
    https://er-c.org/barthel/drprobe/celfile.html for a description of the cel format.

    :Params:
        :outfile (str): The path to the outfile, including extension
        :xyz_obj (XYZ object): Contains the data to be written
        :overwrite (bool): Disables overwriting if False; should always be passed, but to be safe
            this defaults to False, which should prevent any accidental data loss
        :deb_dict (dict): A dictionary containing the Debeye-Waller parameters for each element in
            the XYZ object; NOTE THAT THIS IS CURRENTLY NOT FUNCITONING
    """
    # Safety check outfile (no overwriting without permission!)
    _overwrite_check(outfile, overwrite)

    cel_comment = "Number of atoms: " + str(xyz_obj.num_sites) \
        + "; " + xyz_obj.comment_string
    alpha, beta, gamma = 90, 90, 90  # Not sure if non-ortho coordinates are possible (or useful)

    a_max, a_min = (max([site[1] for site in xyz_obj.sites]),
                    min([site[1] for site in xyz_obj.sites]))
    b_max, b_min = (max([site[2] for site in xyz_obj.sites]),
                    min([site[2] for site in xyz_obj.sites]))
    c_max, c_min = (max([site[3] for site in xyz_obj.sites]),
                    min([site[3] for site in xyz_obj.sites]))
    # xyz files are in A, but we need spans in nm for cel files
    a_span, b_span, c_span = (round((a_max-a_min)/10, 6),
                              round((b_max-b_min)/10, 6),
                              round((c_max-c_min)/10, 6))

    header = (f"{'0' : <4}{a_span : ^9.6f}{b_span : ^9.6f}{c_span : ^9.6f}"
              + f"{alpha : ^4}{beta : ^4}{gamma : >4}\n")

    with open(outfile, 'w') as file:
        file.write(cel_comment)  # The first line in a cel file is a comment string
        file.write(header)  # The second line in a cel file defines its extent

        for site in xyz_obj.sites:
            e = site[0]
            x = f"{round((float(site[1])/10 + abs(a_min)) / a_span, 6) : .6f}"
            y = f"{round((float(site[2])/10 + abs(b_min)) / b_span, 6) : .6f}"
            z = f"{round((float(site[3])/10 + abs(c_min)) / c_span, 6) : .6f}"
            b = f"{deb_dict.get(site[0], 0.0) : .6f}"  # Default biso is 0

            line = (f"{e : <3}{x : ^9}{y : ^9}{z : ^9}" +
                    f"{'1.0' : ^5}{b : ^9}{' 0.0 0.0 0.0' : >11}\n")
            file.write(line)

        # EoF indicator
        file.write(r"*")


def _overwrite_check(file, flag):
    """
    Simply checks that either file does not exist or flag == True, otherwise exits with an error.
    """
    if os.path.isfile(file) and not flag:
        err_str = "File " + file + " already exists. Use the --overwrite option if you want " +\
            "to overwrite it."
        sys.exit(err_str)


def _main(opts):
    ################
    # Read Data In #
    ################
    # Handle underspecification error to do with multiple infiles
    if len(opts.infile) > 1 and opts.outfile is not None:
        sys.exit("When processing multiple input files, you must not specify any output files " +
                 "as this leads to ambiguous behavior (their names will be inferred from the " +
                 "list of input files).")
    for infile in opts.infile:
        infile_name, infile_ext = os.path.splitext(infile)
        # If no extension provided, assume .xyz
        # TODO: This will need to change when support for other input types is implemented
        if infile_ext == "":
            infile_ext = ".xyz"
        # Check that the file is not something other than .xyz
        if not infile_ext == ".xyz":
            sys.exit("This program currently only supports reading from .xyz files.  " +
                     "Conversion to .xyz from other filetypes is planned.")
        # Now ext must be correct, so rebuild infile string and make sure it exists
        infile = infile_name + infile_ext
        if not os.path.isfile(infile):
            err_str = "File " + infile + " does not exist."
            sys.exit(err_str)
        # Attempt to actually read in the file to create an XYZ object
        try:
            with open(infile) as file:
                print("Reading data from " + infile)
                xyz_obj = readXYZ(file)
        except Exception as ex:
            print("Something went wrong while reading from ", infile)
            sys.exit(ex.message)

        ##################
        # Write Data Out #
        ##################
        if opts.outfile is None:  # Outfile(s) not specified
            if opts.type is None:
                sys.exit("You must specify either output type (output file names will be " +
                         "inferred from input files), or else specify an output file name " +
                         "(output type will be inferred from the output file extension).")
            types = opts.type
            for out_type in types:
                # With no outfile provided, infer name from infile and extension from type
                outfile = infile_name + "." + out_type
                # Call the write function associated with out_type
                call = "_write_" + out_type
                # TODO: This doesn't allow specification of filetype-specific params, but it should
                #  e.g. for biso params in cel output
                print("Writing data to " + outfile)
                globals()[call](outfile, xyz_obj, opts.overwrite)

        else:  # Outfile(s) specified
            for outfile in opts.outfile:
                outfile_name, outfile_ext = os.path.splitext(outfile)

                # Types explicitly specified
                if opts.type is not None:
                    types = opts.type
                    for out_type in types:
                        # If --type specified, this takes precedence over given file extensions
                        outfile = outfile_name + "." + out_type
                        call = "_write_" + out_type
                        print("Writing data to " + outfile)
                        globals()[call](outfile, xyz_obj, opts.overwrite)

                else:  # Types not explicitly specified (attempt to infer from outfile ext)
                    out_type = outfile_ext.replace(".", "")
                    # Error out if type cannot be inferred from extension
                    if out_type not in ["csv", "cel"] and opts.type is None:
                        sys.exit("Output type could not be inferred from file extension.  " +
                                 "Either specify output format using --type or ensure that " +
                                 "path(s) passed to --outfile end with a valid extension.")
                    call = "_write_" + out_type
                    print("Writing data to " + outfile)
                    globals()[call](outfile, xyz_obj, opts.overwrite)
        print()  # Line breaks after output loops look nice
    # Everything was successful!
    print("Done!")
    sys.exit(0)  # 0 is the exit code for successful operation


# Run main loop iff called from the command line, but not when imported as module
if __name__ == "__main__":
    import os
    import sys
    import argparse
    opts = _parse_args(sys.argv)
    _main(opts)
