# kani-multimodal-core

This package contains core extensions for using Kani with multimodal language models.

## Installation

kani-multimodal-core should be installed alongside the core kani install using an extra:

```shell
$ pip install "kani[multimodal]"
```

However, you can also explicitly specify a version and install the core package itself:

```shell
$ pip install kani-multimodal-core
```

## Features

This package provides the core multimodal extensions that engine implementations can use -- it does not provide any
engine implementations on its own.

The package adds support for:

- Images (`kani.ext.multimodal_core.ImagePart`)
- Audio (`kani.ext.multimodal_core.AudioPart`)
- Video (WIP)

When installed, these core kani engines will automatically use the multimodal parts:

- OpenAIEngine
- AnthropicEngine

Additionally, the core kani `chat_in_terminal` method will support attaching multimodal data from a local drive or
from the internet using `@/path/to/media` or `@https://example.com/media`.

### Message Parts

The main feature you need to be familiar with is the `MessagePart`, the core way of sending messages to the engine.
To do this, when you call the kani round methods (i.e. `Kani.chat_round` or `Kani.full_round` or their str variants),
pass a *list* of multimodal parts rather than a string:

```python
from kani import Kani
from kani.engines.openai import OpenAIEngine
from kani.ext.multimodal_core import ImagePart

engine = OpenAIEngine(model="gpt-4.1-nano")
ai = Kani(engine)

# notice how the arg is a list of parts rather than a single str!
msg = await ai.chat_round_str([
    "Please describe this image:",
    ImagePart.from_file("path/to/image.png")
])
print(msg)
```

See the docs (TODO) for more information about the provided message parts.

### Terminal Utility

When installed, kani-multimodal-core augments the `chat_in_terminal` utility provided by kani.

This utility allows you to provide multimodal media on your disk or on the internet inline by prepending it with an
@ symbol:

```pycon
>>> from kani import chat_in_terminal
>>> chat_in_terminal(ai)
USER: Please describe this image: @path/to/image.png and also this one: @https://example.com/image.png
```
