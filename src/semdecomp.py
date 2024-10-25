import argparse
import sys
import json
import logging

from pydantic import BaseModel
from outlines import models, generate, samplers


DEFAULT_PROMPT_INFO = {
    'system_prompt': "Often a single sentence conveys multiple ideas/meanings. Here, we decompose one or a few sentences into their component meanings, each phrased as a stand-alone sentence.",
    'prompt_template': """## Example {n}. 

> {original}

We can *maximally decompose* this into the following meaning components, each rephrased as an independent sentence:

{response}

""",
    'examples': [], # TODO add default examples
}


class Components(BaseModel):
    components: list[str]


def main():

    argparser = argparse.ArgumentParser(description='SemDecomp: Separating and paraphrasing meaning components.')
    argparser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='Input file, one (composite) question per line; when omitted stdin.')
    argparser.add_argument('--json', action='store_true', help='To output JSON lists.')
    argparser.add_argument('--prompt', required=False, type=argparse.FileType('r'), default=None, help='.jsonl file with system prompt, prompt template, and examples (keys original, components)')
    argparser.add_argument('--model', required=False, default="unsloth/llama-3-70b-bnb-4bit", type=str)
    argparser.add_argument('--temp', required=False, type=float, help='Temperature', default=None)
    argparser.add_argument('--topp', required=False, type=float, help='Sample only from top p portion of probability distribution', default=None)
    argparser.add_argument('--topk', required=False, type=int, help='Sample only from top k tokens with max probability', default=None)
    argparser.add_argument('--beams', required=False, type=int, help='number of beams to search', default=1)
    argparser.add_argument('-v', '--verbose', action='store_true', help='To show debug messages.')

    args = argparser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not args.prompt:
        logging.warning('Are you sure you don\'t want to specify a custom prompt .json file (--prompt), perhaps containing few-shot examples?')

    prompt_template = create_prompt_template(**(json.load(args.prompt) if args.prompt else DEFAULT_PROMPT_INFO))

    logging.info(f'Prompt template: {prompt_template}')

    model = models.transformers(args.model)
    sampler = samplers.multinomial(samples=args.beams, top_k=args.topk, top_p=args.topp, temperature=args.temp)
    generator = generate.json(model, Components, sampler=sampler)


    for n, line in enumerate(args.file):
        if n and not args.json:  # separate multiple lines of output belonging to a single input
            print()

        original_text = line.strip()
        prompt = prompt_template.format(original=original_text)
        result = generator(prompt)

        if args.json:
            print(json.dumps(result.components))
        else:
            for res in result.components:
                print(res)


def create_prompt_template(system_prompt: str, prompt_template: str, examples: list[dict]) -> str:
    prompt_lines = [system_prompt]
    n_example = 0
    for n_example, example in enumerate(examples, start=1):
        example_prompt = prompt_template.format(n=n_example, original=example['original'], response=json.dumps({'components': example['components']}).replace('{', '{{').replace('}', '}}'))
        prompt_lines.append(example_prompt)
    prompt_lines.append(prompt_template.format(n=n_example+1, original='{original}', response=''))

    full_prompt_template = '\n'.join(prompt_lines)
    return full_prompt_template


if __name__ == '__main__':
    main()