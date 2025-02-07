#codec.py

import os
import argparse
import pathlib
import preprocessors

def main():
    '''Main function
    input_file: path to the input file
    cli format: codec.py -i <input_file> -o <output_file> -p <preprocessor>
    input_file: path to the input file (required)
    output_file: path to the output file (defualt: same name as the input file, but with .bin extension)
    preprocessor: path to the preprocessor file (default: 4b)
    '''
    #validate the input arguments
    parser = argparse.ArgumentParser(prog='codec.py', description='Encode/Decode the input file')
    parser.add_argument('-i', '--input', help='path to the input file', required=True)
    parser.add_argument('-o', '--output', help='path to the output file', default=None)
    parser.add_argument('-p', '--preprocessor', help='preprocessor mode', default='byte')
    args = parser.parse_args()

    #validate the input file
    if not os.path.isfile(args.input):
        print(f"Error: input file {args.input} does not exist")
        return
    #check read permission
    if not os.access(args.input, os.R_OK):
        print(f"Error: input file {args.input} does not have read permission")
        return
    inputFile = args.input
    inputFileName = pathlib.Path(inputFile).name
    inputFilePath = pathlib.Path(inputFile).parent
    outputFile = None
    if args.output is None:
        outputFile = f"{inputFilePath}/{inputFileName}.bin"
    #if output is not none, check if output is valid filename/path
    else:
        outputFilePath = pathlib.Path(args.output).parent
        if os.path.isdir(outputFilePath):
            print(f"Error: output path/name {args.output} is not valid")
            return

    #check if write permission is available for the location
    if not os.access(outputFilePath, os.W_OK):
        print(f"Error: output path {outputFilePath} does not have write permission")
        return


    #validate the preprocessor
    if args.preprocessor not in ['byte', 'char']:
        print(f"Error: preprocessor mode {args.preprocessor} is not valid")
        return

    #get input file data
    data = None
    with open(inputFile, 'rb') as f:
        data = f.read()

    if args.preprocessor == 'byte':
        preprocessor = preprocessors.byte_preprocessor()
    else:
        preprocessor = preprocessors.charPreprocessor()

    #convert data to symbols
    symbols = preprocessor.convert_to_symbols(data)
    



if __name__ == "__main__":
    main()