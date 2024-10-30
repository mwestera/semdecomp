import argparse
import sys
import json
import logging
import numpy
import itertools

from pydantic import BaseModel
from outlines import models, generate, samplers

import csv

import re
import functools

MAX_TOKENS=500

DEFAULT_PROMPT_INFO = {
    'system_prompt': "Often a single sentence conveys multiple ideas/meanings. Here, we decompose one or a few sentences into their component meanings, each phrased as a stand-alone sentence.",
    'prompt_template': """## Example {n}. 

> {original}

We can *maximally decompose* this into the following meaning components, each rephrased as an independent sentence:

{response}

""",
    'examples': [], # TODO add default examples
}

DEFAULT_PROMPT_INFO_CONTEXT = {
    'system_prompt': "Often a single sentence conveys multiple ideas/meanings. Here, we decompose one or a few sentences into their component meanings, each phrased as a stand-alone sentence.",
    'prompt_template': """## Example {n}. 

Some prior context: "{context}"

Target sentence to decompose: "{original}"

Target sentence's components: 

{response}

""",
    'examples': [], # TODO add default examples
}


class Components(BaseModel):
    components: list[str]


def main():

    argparser = argparse.ArgumentParser(description='SemDecomp: Separating and paraphrasing meaning components.')
    argparser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='Input file, one (composite) question per line, or context,question csv; when omitted stdin.')
    argparser.add_argument('--json_out', action='store_true', help='To output JSON lists.')
    argparser.add_argument('--prompt', required=False, type=argparse.FileType('r'), default=None, help='.jsonl file with system prompt, prompt template, and examples (keys original, components)')
    argparser.add_argument('--model', required=False, default="unsloth/llama-3-70b-bnb-4bit", type=str)  #  debug: xiaodongguaAIGC/llama-3-debug
    argparser.add_argument('--temp', required=False, type=float, help='Temperature to use for sampling; greedy (deterministic) if not given', default=None)
    argparser.add_argument('--topp', required=False, type=float, help='Sample only from top p portion of probability distribution', default=.9)
    argparser.add_argument('--topk', required=False, type=int, help='Sample only from top k tokens with max probability', default=None)
    argparser.add_argument('--beams', required=False, type=int, help='number of beams to search; only with sampling enabled (--temp)', default=1)

    argparser.add_argument('--retry', required=False, type=int, help='How often to retry; only with unconstrained generation', default=1)
    argparser.add_argument('--retry_hotter', required=False, type=float, help='Temperature increment for retries', default=0.1)

    argparser.add_argument('--context', action='store_true', help='To distinguish context from target sentence in prompt and examples (keys: context, original, components).')
    argparser.add_argument('--json', action='store_true', help='Whether to constrain generation to JSON (detrimental: https://arxiv.org/abs/2408.02442v1), or a plain bullet list.')

    argparser.add_argument('-v', '--verbose', action='store_true', help='To show debug messages.')

    args = argparser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='SemDecomp %(levelname)s: %(message)s')
    logging.info(json.dumps({k: v for k, v in args.__dict__.items() if k not in ['file', 'prompt']}, indent='  '))
    if not args.prompt:
        logging.warning('Are you sure you don\'t want to specify a custom prompt .json file (--prompt), perhaps containing few-shot examples?')

    prompt_info = json.load(args.prompt) if args.prompt else DEFAULT_PROMPT_INFO if not args.context else DEFAULT_PROMPT_INFO_CONTEXT
    prompt_template = create_prompt_template(**prompt_info, request_json=args.json)

    model = models.transformers(args.model)

    if not args.temp:
        sampler = samplers.GreedySampler()
    else:
        sampler = samplers.multinomial(samples=args.beams, top_k=args.topk, top_p=args.topp, temperature=args.temp)

    if args.json:
        generator = generate.json(model, Components, sampler=sampler)
    else:
        generator = functools.partial(retry_until_parse, model=model, samples=args.beams, top_k=args.topk, top_p=args.topp, temperature=args.temp, increase_temp=args.retry_hotter, parser=parse_json_or_itemized_list_of_strings, n_retries=args.retry, fail_ok=True)

    stats_keeper = []

    if args.context:
        rows = csv.reader(args.file)
    else:
        rows = ([line.strip()] for line in args.file)

    for n, row in enumerate(rows):
        if not args.json_out and n:  # separate multiple lines of output belonging to a single input
            print()

        original = row[-1]
        if args.context:
            prompt = prompt_template.format(context=row[0], original=original)
        else:
            prompt = prompt_template.format(original=original)

        components = generator(prompt, max_tokens=200)
        success = True if components else False
        if not success:
            components = [original]

        logging.debug(f'Original: {original}')
        newline = "\n"
        logging.debug(f'Decomposed: {json.dumps(components)}')
        
        stats_keeper.append(stats_to_record(original, components, success))

        if args.json_out:
            print(json.dumps(components))
        else:
            for res in components:
                print(res)

    log_stats_summary(stats_keeper)


def create_prompt_template(system_prompt: str, prompt_template: str, examples: list[dict], request_json: bool = False) -> str:
    prompt_lines = [system_prompt]
    n_example = 0
    for n_example, example in enumerate(examples, start=1):
        if request_json:
            example_response = json.dumps({'components': example['components']}).replace('{', '{{').replace('}', '}}')
        else:
            example_response = '\n'.join('- ' + comp for comp in example['components'])
        if '{context}' in prompt_template:
            example_prompt = prompt_template.format(n=n_example, original=example['original'], context=example['context'], response=example_response)
        else:
            example_prompt = prompt_template.format(n=n_example, original=example['original'], response=example_response)
        prompt_lines.append(example_prompt)

    if '{context}' in prompt_template:
        actual_prompt = prompt_template.format(n=n_example + 1, original='{original}', context='{context}', response='')
    else:
        actual_prompt = prompt_template.format(n=n_example + 1, original='{original}', response='')

    prompt_lines.append(actual_prompt)

    full_prompt_template = '\n'.join(prompt_lines)

    logging.info(f'Prompt template: {full_prompt_template}')

    return full_prompt_template


def stats_to_record(original_text, components, success):
    return {
        'successful': success,
        'n_components': len(components),
        'components_length_abs': [len(component) for component in components],
        'components_length_rel': [len(component) / len(original_text) for component in components],
    }


def log_stats_summary(stats_keeper: list[dict]) -> None:
    stats_lists = {
        'successful': [s['successful'] for s in stats_keeper], # TODO Also restricted to successful ones
        'n_components': [s['n_components'] for s in stats_keeper],
        'components_length_abs': list(itertools.chain(*(s['components_length_abs'] for s in stats_keeper))),
        'components_length_rel': list(itertools.chain(*(s['components_length_rel'] for s in stats_keeper))),
    }
    for key, stats_list in stats_lists.items():
        logging.info(f'{key}: {numpy.mean(stats_list)} (std: {numpy.std(stats_list)})')


#### Parsing stuff below

def retry_until_parse(prompt, model, parser, samples=None, top_k=None, top_p=None, temperature=None, n_retries=None, fail_ok=False, try_skip_first_line=True, increase_temp=.1, return_upon_fail=None, max_tokens=None):
    """
    :param try_skip_first_line: Sometimes LLMs preface their (otherwise fine) answer by "Here is the answer:" etc.
    """
    n_try = 0
    result = None
    errors = []

    starting_sampler = samplers.multinomial(samples=samples, top_k=top_k, top_p=top_p, temperature=temperature) if temperature else samplers.GreedySampler()
    retry_temp = temperature or 0.0
    while result is None and (n_retries is None or n_try < n_retries):
        if n_try == 0:
            generator = generate.text(model, sampler=starting_sampler)
        else:
            retry_temp += increase_temp
            generator = generate.text(model, sampler=samplers.multinomial(samples=samples, top_k=top_k, top_p=top_p, temperature=retry_temp))
        n_try += 1
        raw = generator(prompt, max_tokens=max_tokens)
        logging.debug(f'(Attempt {n_try}): Model says: {raw}'.replace('\n', '//'))
        try:
            result = parser(raw)
        except ValueError as e1:    # TODO: refactor
            if try_skip_first_line and (raw_lines := raw.splitlines()) and len(raw_lines) > 1:
                try:
                    result = parser('\n'.join(raw_lines[1:]))
                except ValueError as e2:
                    errors.append(str(e1) + '; ' + str(e2))
                    continue
            else:
                errors.append(str(e1))
                continue
        if result:
            return result

    if not fail_ok:
        raise ValueError(f'Max number of retries ({"; ".join(errors)})')
    else:
        logging.warning(f'Max number of retries ({"; ".join(errors)})')
        return return_upon_fail


def parse_json_or_itemized_list_of_strings(raw) -> list[str]:
    try:
        return parse_json_list_of_strings(raw)
    except ValueError as e1:
        try:
            return parse_itemized_list_of_strings(raw)
        except ValueError as e2:
            raise ValueError(f'{e1}; {e2}')

def parse_json_list_of_strings(raw) -> list[str]:
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError('Not a json string')
    if not isinstance(result, list):
        raise ValueError('Not a list')
    if any(not isinstance(x, str) for x in result):
        raise ValueError('List contains a non-string')
    return result


enum_regex = re.compile(r'[ \t]*\d+. +([^\n]+)')
item_regex = re.compile(r'[ \t]*- +([^\n]+)')

def parse_itemized_list_of_strings(raw) -> list[str]:
    if len(result := enum_regex.findall(raw)) <= 1 and len(result := item_regex.findall(raw)) <= 1:
        raise ValueError('Not an itemized/enumerated list of strings')
    return [s.strip('"\'').strip() for s in result]



if __name__ == '__main__':
    main()