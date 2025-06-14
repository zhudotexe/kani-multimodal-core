"""
CLI helpers for chatting with kani from terminal. Gets conditionally imported by kani.utils.cli if this package is
installed to provide multimodal support.
"""

import logging
import mimetypes
import pathlib
import re
import sys
import warnings

from kani.models import MessagePartType

from .audio import AudioPart
from .base import TextPart
from .image import ImagePart
from .utils import get_mime_type

_is_notebook = "ipykernel" in sys.modules

log = logging.getLogger(__name__)

# it's time to get s p i c y
# please don't do this in prod, this is just for a dev helper
# @formatter:off
# https://gist.github.com/gruber/8891611
WEB_REGEX = r"""((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""  # noqa: E501
# @formatter:on
MEDIA_RE = re.compile(
    r"(?<!\S)@(?:"  # not after a non-WS character (to prevent catching email addresses etc)
    rf"(?P<url>{WEB_REGEX})"  # URL
    r"|(?P<path>/?(\S+?/)*([^/\s]+\.[^/\s]+))"  # path, no quotes
    r"|(?P<path_quot>\"/?([^\"]+?/)*([^/\"]+\.[^/\s\"]+)\")"  # path with quotes
    r")",
    re.IGNORECASE,
)


# ==== parsing helpers ====
async def parts_from_cli_query(query: str) -> list[MessagePartType]:
    """Parse a string with paths to media prepended by ``@`` into the right messageparts."""
    query_parts = []
    last_idx = 0
    for media_match in MEDIA_RE.finditer(query):
        # push everything between the end of the last path and the start of this one to the parts
        query_parts.append(query[last_idx : media_match.start()])
        last_idx = media_match.end()

        # if a path:
        if not media_match["url"]:
            # ensure the path is valid
            if path := media_match["path"]:
                fp = pathlib.Path(path)
            else:
                fp = pathlib.Path(media_match["path_quot"].strip('"'))

            # if not valid, push the string to parts
            log.debug(f"Found path: {fp}")
            if not (fp.exists() and fp.is_file()):
                warnings.warn(f"The given path ({fp}) either does not exist or is not a valid file.")
                query_parts.append(media_match[0])
            # otherwise, push a media part depending on the filetype
            else:
                mime, _ = mimetypes.guess_type(fp.name)
                if mime.startswith("image/"):
                    query_parts.append(ImagePart.from_file(fp))
                elif mime.startswith("audio/"):
                    query_parts.append(AudioPart.from_file(fp))
                # TODO
                # elif mime.startswith("video/"):
                #     query_parts.append(VideoPart.from_file(fp))
                else:
                    warnings.warn(
                        f"I could not understand the filetype of the given file: {fp}\n(expected MIME type to be one of"
                        f" image/*, audio/*, or video/*, but got {mime})"
                    )
                    query_parts.append(media_match[0])
        # if a url:
        else:
            url = media_match["url"]
            # try getting the MIME type
            mime = await get_mime_type(url)

            # TODO special case - we can handle youtube videos

            if mime.startswith("image/"):
                query_parts.append(await ImagePart.from_url(url))
            elif mime.startswith("audio/"):
                query_parts.append(await AudioPart.from_url(url))
            # TODO
            # elif mime.startswith("video/"):
            #     query_parts.append(await VideoPart.from_url(url))
            else:
                warnings.warn(
                    f"I could not understand the filetype of the given URL: {url}\n(expected MIME type to be one of"
                    f" image/*, audio/*, or video/*, but got {mime})"
                )
                query_parts.append(media_match[0])

    # and make sure the rest of the query is in the parts
    query_parts.append(query[last_idx:])
    return [part for part in query_parts if part]


# ==== media display helpers ====
def display_media_ipython(
    parts: list[MessagePartType],
    *,
    show_text=False,
    media_height=350,
):
    """
    Display a list of media parts using IPython display.

    :param parts: The list of parts to display.
    :param show_text: Whether to echo text parts or only display media parts.
    :param media_height: The height to display image and video media parts at.
    """
    from IPython.display import Audio, Image, display

    # show each part in an IPython display
    for part in parts:
        if isinstance(part, (str, TextPart)) and show_text:
            print(str(part))
        elif isinstance(part, ImagePart):
            display(Image(part.as_bytes(), height=media_height))
        elif isinstance(part, AudioPart):
            display(Audio(data=part.raw, rate=part.sample_rate))
        # TODO
        # elif isinstance(part, VideoPart):
        #     display(Video(filename=part.as_tempfile(), height=media_height))
        else:
            print(str(part))
