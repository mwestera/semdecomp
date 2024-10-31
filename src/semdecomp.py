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

# TODO add default examples

DEFAULT_PROMPT_INFO = {
    'system_prompt': "Often a single sentence conveys multiple ideas/meanings. Here, we decompose one or a few sentences into their component meanings, each phrased as a stand-alone sentence.",
    'prompt_template': """## Example {n}. 
Original: "{original}"
Components, rephrased independently:{response}""",
    'examples': [],
}

DEFAULT_PROMPT_INFO_CONTEXT = {
    'system_prompt': "Often a single sentence conveys multiple ideas/meanings. Here, we decompose one or a few sentences into their component meanings, each phrased as a stand-alone sentence.",
    'prompt_template': """## Example {n}. 
Prior context: "{context}"
Target sentence: "{original}"
Components of target sentence, rephrased independently:{response}""",
    'examples': [],
}


class Components(BaseModel):
    components: list[str]


def main():

    argparser = argparse.ArgumentParser(description='SemDecomp: Separating and paraphrasing meaning components.')
    argparser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='Input file, one (composite) question per line, or context,question csv (with --context).')
    argparser.add_argument('--prompt', required=False, type=argparse.FileType('r'), default=None, help='.jsonl file with system prompt, prompt template, and examples (each with keys "original" and "response"; also "context" if --context is given)')
    argparser.add_argument('--json_out', action='store_true', help='To output JSON lists instead of plaintext lines.')
    argparser.add_argument('--context', action='store_true', help='To distinguish context from target sentence in prompt and examples (keys: context, original, components).')
    argparser.add_argument('--model', required=False, default="unsloth/llama-3-70b-bnb-4bit", type=str, help='Huggingface identifier.')  #  debug: xiaodongguaAIGC/llama-3-debug

    # sampling params
    argparser.add_argument('--temp', required=False, type=float, help='Temperature to use for sampling; greedy (deterministic) if 0', default=0.0)
    argparser.add_argument('--topp', required=False, type=float, help='Sample only from top p portion of probability distribution', default=.9)
    argparser.add_argument('--topk', required=False, type=int, help='Sample only from top k tokens with max probability', default=None)
    argparser.add_argument('--beams', required=False, type=int, help='number of beams to search; only with sampling enabled (--temp)', default=1)
    argparser.add_argument('--json', action='store_true', help='Whether to constrain generation to JSON (detrimental: https://arxiv.org/abs/2408.02442v1); otherwise it will mostly respond with plain bullet lists, which will then be parsed.')

    # retrying params
    argparser.add_argument('--retry_hotter', required=False, type=float, help='Temperature increment for retries', default=0.1)
    argparser.add_argument('--tries', required=False, type=int, help='How often to try; only used with unconstrained generation', default=1)

    argparser.add_argument('-v', '--verbose', action='store_true', help='To show debug messages.')

    args = argparser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='SemDecomp %(levelname)s: %(message)s')

    logging.info(json.dumps({k: v for k, v in args.__dict__.items() if k not in ['file', 'prompt']}, indent='  '))

    if not args.prompt:
        logging.warning('Are you sure you don\'t want to specify a custom prompt .json file (--prompt), perhaps containing few-shot examples?')

    if args.tries <= 1 and not args.json:
        logging.warning('Not using constrained generation (to enable, include --json). This may be a wise choice, but do consider increasing the number of --tries.')

    prompt_info = json.load(args.prompt) if args.prompt else DEFAULT_PROMPT_INFO if not args.context else DEFAULT_PROMPT_INFO_CONTEXT
    prompt_template = create_prompt_template(**prompt_info, request_json=args.json)

    model = models.transformers(args.model)

    if args.json:
        generator = generate.json(model, Components, sampler=samplers.GreedySampler() if not args.temp else samplers.MultinomialSampler(beams=args.beams, top_k=args.topk, top_p=args.topp, temperature=args.temperature))
    else:
        sampling_params = {'beams': args.beams, 'top_k': args.topk, 'top_p': args.topp, 'temperature': args.temp, 'increase_temp': args.retry_hotter}
        generator = functools.partial(retry_until_parse, model=model, parser=parse_itemized_list_of_strings, fail_ok=True, n_tries=args.tries, **sampling_params)


    stats_keeper = []
    for n, item in enumerate(read_items(args.file, args.context)):
        if not args.json_out and n:
            print()  # separate multiple lines of output belonging to a single input

        prompt = prompt_template.format(**item)
        logging.debug(f'Prompt: {prompt[-200:]}')
        response = generator(prompt, max_tokens=200)
        components = response.components if args.json else (response or [item['original']])

        logging.debug(f'{n}\nOriginal: {item["original"]}\nDecomposed: {json.dumps(components)}')
        stats_keeper.append(stats_to_record(item['original'], components, success=bool(response)))
        print(json.dumps(components) if args.json_out else '\n'.join(components))

    log_stats_summary(stats_keeper)


def create_prompt_template(system_prompt: str, prompt_template: str, examples: list[dict], request_json: bool = False) -> str:
    prompt_lines = [system_prompt]
    n_example = 0

    for n_example, example in enumerate(examples, start=1):
        prompt_values = {'n': n_example, **example}
        if request_json:
            prompt_values['response'] = json.dumps({'components': example['response']}).replace('{', '{{').replace('}', '}}')
        else:
            prompt_values['response'] = '\n' + ('\n'.join('- ' + comp for comp in example['response']))
        prompt_lines.append(
            prompt_template.format(**prompt_values)
        )

    prompt_values = {'n': n_example+1, 'original': '{original}', 'response': ' ' if request_json else '\n -'}   # don't forget to add this dash to generated text
    if '{context}' in prompt_template:
        prompt_values['context'] = '{context}'
    final_prompt_line = prompt_template.format(**prompt_values)
    if final_prompt_line.startswith('## Example'):
        final_prompt_line = final_prompt_line.replace('\n', ' (Final example!)\n', 1)  # To help it stop hallucinating more and more examples
    prompt_lines.append(final_prompt_line)

    full_prompt_template = '\n\n'.join(prompt_lines)
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
    stats_keeper_successful = [s for s in stats_keeper if s['successful']]
    for stats, label in [(stats_keeper, 'all'), (stats_keeper_successful, 'successful')]:
        stats_lists = {
            'successful': [s['successful'] for s in stats],
            'n_components': [s['n_components'] for s in stats],
            'components_length_abs': list(itertools.chain(*(s['components_length_abs'] for s in stats))),
            'components_length_rel': list(itertools.chain(*(s['components_length_rel'] for s in stats))),
        }
        for key, stats_list in stats_lists.items():
            logging.info(f'{key} ({label}): {numpy.mean(stats_list)} (std: {numpy.std(stats_list)})')


#### Parsing stuff below

def retry_until_parse(prompt, model, parser, n_tries=None, fail_ok=False, try_skip_first_line=True, return_upon_fail=None, max_tokens=None, **kwargs):
    """
    :param try_skip_first_line: Sometimes LLMs preface their (otherwise fine) answer by "Here is the answer:" etc.
    """
    beams, top_p, top_k = kwargs.get('beams'), kwargs.get('top_p'), kwargs.get('top_k')
    current_temp = kwargs.get('temperature', 0.0)
    delta_temp = kwargs.get('increase_temp', 0.0)

    n_try = 0
    result = None
    collected_error_messages = []

    while n_tries is None or n_try < n_tries:
        generator = generate.text(
            model,
            sampler=samplers.GreedySampler() if not current_temp else samplers.multinomial(beams, top_p=top_p, top_k=top_k, temperature=current_temp)
        )

        current_temp += delta_temp
        n_try += 1

        raw = generator(prompt, max_tokens=max_tokens)
        logging.debug(f'(Attempt {n_try}): Model says: {raw}'.replace('\n', '//'))
        try:
            result = parser(raw)
        except ValueError as e1:
            if try_skip_first_line and (raw_lines := raw.splitlines()) and len(raw_lines) > 1:
                try:
                    result = parser('\n'.join(raw_lines[1:]))
                except ValueError as e2:
                    collected_error_messages.append(str(e1) + ' & ' + str(e2))
                    continue
                else:
                    break
            else:
                collected_error_messages.append(str(e1))
                continue
        else:
            break

    if result:
        return result

    error_message = f'Max number of retries ({"; ".join(collected_error_messages)})'
    if not fail_ok:
        raise ValueError(error_message)
    else:
        logging.warning(error_message)
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
    result = []
    raw = f'- {raw}'    # Because the first dash was given in prompt.
    for line in raw.splitlines():
        if line.startswith('## '):  # Because sometimes the generation continues hallucinating more examples :D
            break
        if item_regex.match(line):
            result.append(line.strip('"\'').strip())
    if not result:
        raise ValueError('Not an itemized/enumerated list of strings')
    return result


def read_items(file, with_context):
    if with_context:
        rows = csv.DictReader(file, fieldnames=['context', 'original'])
    else:
        rows = ({'original': line.strip()} for line in file)
    return rows


if __name__ == '__main__':
    main()