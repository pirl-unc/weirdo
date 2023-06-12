import sys
import argparse

from .reduced_alphabet import alphabets
parser = argparse.ArgumentParser("weirdo")
subparsers = parser.add_subparsers(dest='subparser')


parser_translate = subparsers.add_parser('translate', help="Translate amino acid sequences to reduced alphabets")

parser_translate_inputs = parser_translate.add_mutually_exclusive_group(required=True)
parser_translate_inputs.add_argument("--input-fasta")
parser_translate_inputs.add_argument("--input-sequence")

parser_translate.add_argument(
    '-a', '--alphabet', dest='alphabet', help='Reduced alphabet name', choices=tuple(alphabets.keys()), required=True)


def run(args_list=None):
    if args_list is None:
        args_list = sys.argv[1:]
    args = parser.parse_args(args_list)   
    if args.subparser == "translate":
        alphabet = alphabets[args.alphabet]
        print("-> " + "".join([alphabet[aa] for aa in args.input_sequence]))
    print(args)
    

