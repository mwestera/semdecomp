# SemDecomp: Decomposing a sentence into its meaning components #

It uses a local LLM to, for example, go from this composite question:

> Hoe duur was die wasmachine, van welke winkel en bevalt hij?

to its three component questions:

> Hoe duur was die wasmachine?
> Waar heb je die wasmachine gekocht?
> Bevalt die wasmachine?


## Install ##

```bash
pip install git+https://github.com/mwestera/semdecomp
```

This will make available the command `semdecomp`, which tasks an LLM with splitting a sentence or two into the intents expressed, listed as independent paraphrases.

## Usage ##

Given a text file `questions.txt` containing composite questions to break down, one per line:

```text
Hoe werkt dat en waarom?
Wie ben je en hoe oud ben je?
```

You can feed it into `semdecomp` like this:

```bash
semdecomp questions.txt --temp .3
```

Or pipe into it:

```bash
cat questions.txt | semdecomp
```

This will output one subquestion per line, the potentially multiple outputs for different inputs separated by empty lines.

Alternatively, add `--json_out` to get a single-line JSON list per input, instead of potentially multiple plain text lines.

For more info about the options, do: 

```bash
semdecomp --help
```

## Usage with a custom prompt (recommended)

You can specify a custom prompt, with few-shot examples, from a separate `.json` file:

```bash
semdecomp questions.txt --prompt questions_prompt.json
```

Example of a prompt specification:

```json
{"system_prompt": "Often a single sentence asks multiple questions. Here, for the Dutch language, we decompose each sentence into its component questions, each phrased as a stand-alone question.",
  "prompt_template": "## Example {n}.\n\n> {original}\n\nWe can *maximally decompose* this into the following subquestions, each rephrased as an independent question:\n\n{response}\n\n",
  "examples": [
  {
    "original": "Wat deze maatregel betreft, sinds wanneer geldt deze en wat was destijds de motivatie?",
    "components": [
      "Sinds wanneer geldt deze maatregel?",
      "Wat was destijds de motivatie voor deze maatregel?"
    ]
  },
  {
    "original": "Heeft u de brief van de Indonesische overheid gelezen, en zoja, wat is uw reactie?",
    "components": [
      "Heeft u de brief van de Indonesische overheid gelezen?",
      "Wat is uw reactie op de brief van de Indonesische overheid?"
    ]
  },
  {
    "original": "Wat is de staatrechtelijke grondslag van deze maatregel? Is dit onderzocht (en door wie)?",
    "components": [
      "Wat is de staatrechtelijke grondslag van deze maatregel?",
      "Is de staatrechtelijke grondslag van deze maatregel onderzocht?",
      "Door wie is de staatsrechtelijke grondslag van deze maatregel onderzocht?"
    ]
  }
]}
```
