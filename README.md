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
- Video (`kani.ext.multimodal_core.VideoPart`)
- Other binary files, such as PDFs (`kani.ext.multimodal_core.BinaryFilePart`)

When installed, these core kani engines will automatically use the multimodal parts:

- OpenAIEngine
- AnthropicEngine
- GoogleAIEngine

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

See the docs (https://kani-multimodal-core.readthedocs.io) for more information about the provided message parts.

### Terminal Utility

When installed, kani-multimodal-core augments the `chat_in_terminal` utility provided by kani.

This utility allows you to provide multimodal media on your disk or on the internet inline by prepending it with an
@ symbol:

```pycon
>>> from kani import chat_in_terminal
>>> chat_in_terminal(ai)
USER: Please describe this image: @path/to/image.png and also this one: @https://example.com/image.png
```

## Returning Multimodal Content from AIFunctions

What if an AI function returns multimodal content? By default, Kani applies certain transformations to the return value
of an AIFunction in order:

* if it is a :class:`.ChatMessage`, do not modify it
* if it is a list of :class:`.MessagePart`, do not modify it
* if it is a JSON-serializable Python dict, serialize it to JSON
* if it is a Pydantic model, serialize it to JSON
* otherwise, cast it to a string

Thus, in order to return multimodal content, the AIFunction must return a FUNCTION-role chat message. For example, the
following code snippet defines a function which returns an image:

```python
class MyKani(Kani):
    @ai_function()
    async def get_image(self, url: str):
        """Download the image at a certain URL, and view it."""
        return [await ImagePart.from_url(url)]
```

However, most API-based LLMs currently do not support multimodal returns from tools.
