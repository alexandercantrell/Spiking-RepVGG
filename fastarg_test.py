from argparse import ArgumentParser
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf
import json

Section('test','asdf').params(
    x=Param(int,'x',default=1)
)

@param('test.x')
def main(y,x):
    print(f'{x}*{y}={x*y}')

def make_config():
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    #config.add_entry(Section('test'),'dir','asdf')
    config.asdf=3
    print(config.__dict__)
    print(config.summary())
    print(dir(config))
    config.summary()

if __name__=="__main__":
    make_config()
    main(3)