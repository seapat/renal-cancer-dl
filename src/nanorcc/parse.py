from bs4 import BeautifulSoup
from collections import OrderedDict
import csv
from glob import glob
from itertools import chain
import pandas as pd
import warnings

def parse_tag(tag,data_tag='Code_Summary',message_tag='Messages') -> tuple[str, object]:
    """A function for parsing tags read by BeautifulSoup from RCC files"""
    # use case-insensitive match because html parser forces tags to lowercase
    if tag.name.casefold() == data_tag.casefold():
        data= list(csv.DictReader(tag.contents[0].strip().splitlines()))
    elif tag.name.casefold() == message_tag.casefold():
        data = tag.contents[0]
        if data == '\n':
            data = ''
    else:
        data = dict(csv.reader(tag.contents[0].strip().splitlines()))
        # Rename ID to match the lane or sample where applicable.
        if 'ID' in data:
            data[tag.name.split('_')[0].capitalize() + 'ID'] = data['ID']
            data.pop('ID')
    return(tag.name,data)

def parse_rcc_file(file):
    """RCC file parsing function; calls parse_tag on each tag in the file"""
    with open(file) as f:
        soup = BeautifulSoup(f,'html.parser')
    soup = filter(lambda x: x!='\n',soup.contents)
    data_pairs:list[tuple[str, dict]] = [parse_tag(tag) for tag in soup] #type: ignore
    data: dict[str, dict] = dict(data_pairs)
    # Convert the count value from str to float
    counts = dict(
        (d['Name'],float(d.pop('Count'))) for d in data['code_summary']
    )
    genes = data.pop('code_summary')
    sample_data = OrderedDict(
        **data['header'],
        **data['sample_attributes'],
        **data['lane_attributes'],
        **{'Messages':data['messages']},
        **counts,
    )
    return(sample_data,genes)

def get_rcc_data(files):
    """Read data from RCC files and return pandas dataframes of the data and
    the genes."""
    if type(files) == list:
        pass
    elif type(files) == str:
        files = glob(files)
    else:
        raise(
            TypeError(
                'Files must be a list or a valid string for the glob function.'
            )
        )
    exp,genes = zip(*map(parse_rcc_file,files))
    exp = pd.DataFrame(list(exp))
    # Make sure all numeric columns are numeric
    exp = exp.apply(pd.to_numeric,errors='ignore')
    # check codesets
    if exp['GeneRLF'].nunique() > 1:
        warnings.warn(
            "Multiple code sets detected. Process files from "\
                "different NanoString Platforms separately to "\
                "eliminate the risk of overlapping gene names causing "\
                "data omission.",
            UserWarning
        )
    # Get unique genes from the genes across all files
    # Convert dicts to tuples and then to a set to get unique elements
    genes = set(map(lambda x: tuple(x.items()),chain.from_iterable(genes)))
    genes = pd.DataFrame([dict(i) for i in genes])
    genes = genes.sort_values(by=['CodeClass','Name']).reset_index(drop=True)
    return(exp,genes)